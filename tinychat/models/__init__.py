from .falcon import FalconForCausalLM
from .llama import LlamaForCausalLM
from .mpt import MPTForCausalLM
from .llava_llama import LlavaLlamaForCausalLM
from .qwen2 import Qwen2ForCausalLM
try:
    from .internvl3 import InternVL3
except ImportError as e:
    print("InternVL3 model import failure. To activate, please install VILA at https://github.com/NVlabs/VILA.")

