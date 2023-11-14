import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import awq_inference_engine


class FTLlamaRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        output = torch.empty_like(x)
        awq_inference_engine.layernorm_forward_cuda(
            x, self.weight, output, self.variance_epsilon
        )
        return output


def make_quant_norm(model):
    """
    Replace all LlamaRMSNorm modules with FTLlamaRMSNorm modules
    """

    for name, m in model.named_modules():
        if not isinstance(m, LlamaRMSNorm):
            continue

        norm = FTLlamaRMSNorm(m.weight, m.variance_epsilon)

        if "." in name:
            parent_name = name.rsplit(".", 1)[0]
            child_name = name[len(parent_name) + 1 :]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ""
            parent = model
            child_name = name

        # print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, norm)
