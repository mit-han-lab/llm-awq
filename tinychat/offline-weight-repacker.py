import argparse
import torch
import numpy as np
from typing import List
from collections import OrderedDict


def qweight_unpack(qweight):
    assert qweight.dtype == torch.int32
    n = qweight.shape[0]
    k = qweight.shape[1] * 8
    unpacked_qweight = torch.zeros((n, k), dtype=torch.int32, device=qweight.device)
    mask = 0x0000000F
    for kk in range(k):
        ele_offset = kk // 8
        bit_offset = (kk % 8) * 4
        unpacked_qweight[:, kk] = (qweight[:, ele_offset] >> bit_offset) & mask

    return unpacked_qweight


def packing_v2_from_unpacked(unpacked_qweight, interleave, kstride):
    # unpacked_qweight: [N, K]
    N = unpacked_qweight.shape[0]
    K = unpacked_qweight.shape[1]

    Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
    Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    Packed_Kernel = Packed_Kernel.reshape(N, K)

    # interleaving every four rows
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
    Packed_Kernel = Packed_Kernel.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    Packed_Kernel = (
        Packed_Kernel[..., 0]
        | (Packed_Kernel[..., 1] << 4)
        | (Packed_Kernel[..., 2] << 8)
        | (Packed_Kernel[..., 3] << 12)
    )
    # reshape to (N // 4, K), FP16 format
    Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
    qweight_v2 = (
        torch.tensor(Packed_Kernel.astype("int16"))
        .to(unpacked_qweight.device)
        .contiguous()
    )
    return qweight_v2


def multiply_scale_qzero_negative(scales, qzeros, zp_shift=-8):
    pack_size = 8
    k_groups = scales.shape[1]
    scaled_zeros = torch.zeros_like(scales)
    for group_idx in range(k_groups):
        zero_idx = group_idx // pack_size
        zero_offset = group_idx % pack_size
        zero = qzeros[:, zero_idx] >> (4 * zero_offset) & 0x0000000F
        scaled_zeros[:, group_idx] = scales[:, group_idx] * zero.to(scales.dtype)
    return -(scaled_zeros + (zp_shift * scales))


def qweight_pack_v1_to_v2(qweight, interleave, kstride):
    unpacked_qweight = qweight_unpack(qweight)
    qweight_v2 = packing_v2_from_unpacked(unpacked_qweight, interleave, kstride)
    return qweight_v2


def ckpt_check():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", type=str, default="./vicuna-7b-w4-g128-awq-v2-1.pt")
    parser.add_argument("--input2", type=str, default="./vicuna-7b-w4-g128-awq-v2-2.pt")
    args = parser.parse_args()

    model_dict1 = torch.load(args.input1)
    model_dict2 = torch.load(args.input2)

    keys = model_dict1.keys()
    for key in keys:
        param = model_dict1[key]
        assert type(param) == torch.Tensor
        if (
            "qweight" in key
            or "scales" in key
            or "qzeros" in key
            or "scaled_zeros" in key
        ):
            print("=" * 50)
            print(key)
            # print(model_dict1[key])
            # print(model_dict2[key])
            diff = torch.max(torch.abs(model_dict2[key] - model_dict1[key]))
            print(diff)
            assert diff < 1e-6
            print("=" * 50)


def offline_repacker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./vicuna-7b-w4-g128-awq.pt")
    parser.add_argument("--output", type=str, default="./vicuna-7b-w4-g128-awq-v2.pt")
    args = parser.parse_args()

    model_dict = torch.load(args.input)
    model_dict_v2 = OrderedDict()

    keys = model_dict.keys()
    for key in keys:
        param = model_dict[key]
        assert type(param) == torch.Tensor
        if "qweight" in key:
            print("repacking:", key)
            qweight = param
            qweight_v2 = qweight_pack_v1_to_v2(qweight, interleave=4, kstride=64)
            model_dict_v2[key] = qweight_v2
        elif "scales" in key:
            print("repacking:", key)
            scales = param
            # print(scales.shape)
            scales_v2 = scales.transpose(1, 0).contiguous()
            model_dict_v2[key] = scales_v2

            # deal with corresponding zero points
            zeros_key = key.replace("scales", "qzeros")
            print("repacking:", zeros_key)

            zeros_key_v2 = key.replace("scales", "scaled_zeros")
            qzeros = model_dict[zeros_key]
            scaled_zeros_v2 = multiply_scale_qzero_negative(scales, qzeros, zp_shift=0)
            # K // G, N
            scaled_zeros_v2 = scaled_zeros_v2.transpose(1, 0).contiguous()
            model_dict_v2[zeros_key_v2] = scaled_zeros_v2
        elif "qzeros" in key:
            pass
        else:
            print("copying:", key)
            model_dict_v2[key] = param

    torch.save(model_dict_v2, args.output)


if __name__ == "__main__":
    offline_repacker()
    # ckpt_check()
