# Adapted from qserve (https://github.com/mit-han-lab/qserve/tree/main) and modified by Yuming Lou


from typing import Optional, Union
from torch.nn import Parameter
import awq_inference_engine
import torch
import gc
from awq.utils.module import set_op_by_name
from tqdm import tqdm


class W8A8OF16LinearStaticScale(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale: Union[torch.tensor, float] = 1.0,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        # size [1] or size [oc]
        self.register_buffer(
            "dequant_scale", torch.ones(out_features, dtype=torch.half)
        )
        # Parameters.
        # NOTE: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.create_weights()

        if bias:
            self.bias = torch.empty(
                self.out_features,
                device=torch.cuda.current_device(),
                dtype=torch.float16,
            )
        else:
            self.register_parameter("bias", None)

    def create_weights(self) -> None:
        self.register_buffer(
            "weight",
            torch.empty(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input_):
        # Matrix multiply.
        output = self.apply_weights(input_, self.bias)
        output_bias = self.bias
        return output, output_bias


class W8A8OF16LinearDynamicInputScale(W8A8OF16LinearStaticScale):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale: Union[torch.tensor, float] = 1.0,
        params_dtype: Optional[torch.dtype] = None,
    ):            
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            scale=scale,
            params_dtype=params_dtype,
        )
        if bias:
            self.apply_weights = self.apply_weights_bias
        else:
            self.apply_weights = self.apply_weights_no_bias
    
    #W bias. Fused bias and W8A8 GEMM
    def apply_weights_bias(
        self,
        # [batch, tokens, channels]
        x: torch.Tensor,
        # [batch * tokens]
        input_scale: torch.Tensor,
        output_buffer: torch.Tensor,
        bias: torch.Tensor = None,
    ):
        x_shape = x.shape
        if len(x.shape) > 2:
            assert 0, "Not implemented"
            x = x.view(-1, x_shape[-1])
        # If use awq_inference_engine.w8a8_gemm_fuse_bias_forward_cuda
        awq_inference_engine.w8a8_gemm_fuse_bias_forward_cuda(
        x, self.weight, self.dequant_scale, input_scale, output_buffer, bias
        )                    
        if len(x.shape) > 2:
            assert 0, "Not implemented 2"
            output_buffer = output_buffer.view(*x_shape[:-1], -1)
    
    #W/H bias. W8A8 GEMM
    def apply_weights_no_bias(
            self,
            # [batch, tokens, channels]
            x: torch.Tensor,
            # [batch * tokens]
            input_scale: torch.Tensor,
            output_buffer: torch.Tensor,
            bias: torch.Tensor = None,
        ):
            x_shape = x.shape
            if len(x.shape) > 2:
                assert 0, "Not implemented"
                x = x.view(-1, x_shape[-1])
            # If use awq_inference_engine.w8a8_gemm_forward_cuda
            awq_inference_engine.w8a8_gemm_forward_cuda(
                x, self.weight, self.dequant_scale, input_scale, output_buffer
            )
            if len(x.shape) > 2:
                assert 0, "Not implemented 2"
                output_buffer = output_buffer.view(*x_shape[:-1], -1)

    def forward(self, input_, input_scale, output_buffer):
        # Matrix multiply.
        self.apply_weights(input_, input_scale, output_buffer, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear,
        init_only=False,
        s1_scale=None,
        fc1=False,
    ):
        q_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
        )
        if init_only:  # just prepare for loading sd
            return q_linear
        if s1_scale is None:
            s1_scale, _ = torch.max(abs(linear.weight.data), dim=-1, keepdim=True)
            s1_scale = s1_scale.clamp_(min=1e-5).div_(127)

        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().half().contiguous().cuda()
        ## Quantize the weights
        # ---- Quantize the weights to int8 ---- #
        linear_weight = linear.weight.data  # OC, IC
        linear_weight = linear_weight.div_(s1_scale.to(linear_weight.device))
        linear_weight = linear_weight.round_().to(torch.int8)
            
        q_linear.weight.data[:, :] = linear_weight.half().contiguous().cuda()

        # ---- Pack the scales ---- #
        q_linear.dequant_scale.data[:] = (
            s1_scale.reshape(-1).half().contiguous().cuda()
        )
        return q_linear.cuda()

    @classmethod
    def from_qkv(
        cls,
        q,
        k,
        v,
        init_only=False,
        s1_scale=None,
    ):
        q_linear = cls(
            q.in_features,
            q.out_features + k.out_features + v.out_features,
            q.bias is not None,
        )
        if init_only:  # just prepare for loading sd
            return q_linear
        weight = torch.cat([q.weight.data, k.weight.data, v.weight.data], dim=0)

        if s1_scale is None:
            s1_scale, _ = torch.max(abs(weight), dim=-1, keepdim=True)
            s1_scale = s1_scale.clamp_(min=1e-5).div_(127)

        if q.bias is not None:
            bias = torch.cat([q.bias, k.bias, v.bias], dim=0)
            q_linear.bias = bias.clone().half().contiguous().cuda()
        # ---- Quantize the weights to int8 ---- #
        weight = weight.div_(s1_scale.to(weight.device))
        weight = weight.round_().to(torch.int8)

        q_linear.weight.data[:, :] = weight.contiguous().cuda()

        # ---- Pack the scales ---- #
        q_linear.dequant_scale.data[:] = (
            s1_scale.reshape(q.out_features + k.out_features + v.out_features)
            .half()
            .contiguous().cuda()
        )
        return q_linear.cuda()


class FakeW8A8Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, wbit: int = 8
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.half)
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(1, out_features, dtype=torch.half)
            )
        else:
            self.bias = None
        self.wbit = wbit
        self.maxv = 2 ** (wbit - 1) - 1

    def forward(self, input):
        t_shape = input.shape
        input.view(-1, t_shape[-1])
        scales = input.abs().max(dim=-1, keepdim=True)[0]
        scales.clamp_(min=1e-5).div_(self.maxv)
        input.div_(scales).round_().mul_(scales)
        output = torch.functional.F.linear(input, self.weight, self.bias)
        return output

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, wbit=8):
        fake_linear = cls(
            linear.in_features, linear.out_features, linear.bias is not None, wbit
        )
        maxv = 2 ** (wbit - 1) - 1
        scale = (
            torch.max(abs(linear.weight.data.detach()), -1, keepdim=True)[0]
            .clamp_(min=1e-5)
            .div_(maxv)
        )
        weight = linear.weight.data / scale
        weight = weight.round_()
        weight = weight * scale
        fake_linear.weight.copy_(weight.contiguous())
        if linear.bias is not None:
            fake_linear.bias.copy_(
                linear.bias.detach().half().reshape(1, linear.out_features).contiguous()
            )
        else:
            linear.bias = None
        del linear, scale, weight
        torch.cuda.empty_cache()
        return fake_linear


def fake_quant(model, wbit=8):
    for name, m in tqdm(
        model.named_modules(),
        desc="Fake quantizing",
        total=len(list(model.named_modules())),
    ):
        if isinstance(m, torch.nn.Linear):
            FQlinear = FakeW8A8Linear.from_linear(m, wbit)
            del m
            torch.cuda.empty_cache()
            set_op_by_name(model, name, FQlinear)
