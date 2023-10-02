import os
import re
import torch
import argparse


def split(
    ckpt_path: str,
    out_folder_path: str,
):
    os.system(f"mkdir -p {out_folder_path}")
    ckpt = torch.load(ckpt_path)
    count = 0
    for key, value in ckpt.items():
        output_dict = {key: value}
        output_name = out_folder_path + "/" + key + ".pt"
        torch.save(output_dict, output_name)
        count += 1
    print(f"Finished splitting the original checkpoint into {count} shards.")


def ckpt_folder_reader(ckpt_folder_path: str):
    file_list = [f for f in os.listdir(ckpt_folder_path) if f.endswith(".pt")]
    for ckpt in file_list:
        print(ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to the original checkpoint (ends with *.pt)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Folder to store the splitted checkpoint shards",
    )

    args = parser.parse_args()
    assert (
        args.input_path is not None
    ), "Please specify the path to the original checkpoint."
    if args.output_path is None:
        suffix = r"\.pt$"
        args.output_path = re.sub(suffix, "", args.input_path)

    split(args.input_path, args.output_path)
