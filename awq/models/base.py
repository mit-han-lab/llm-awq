import os
import gc
import json
import torch
import functools
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

from huggingface_hub import snapshot_download
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.quantizer import pseudo_quantize_tensor
from awq.quantize.qmodule import WQLinear, ScaledActivation
from awq.quantize.auto_clip import auto_clip_block, apply_clip
from awq.quantize.auto_scale import auto_scale_block, apply_scale
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from awq.utils.module import append_str_prefix, get_op_name, get_named_linears, set_op_by_name

class BaseAWQForCausalLM(nn.Module):
    def __init__(self, model, model_type, is_quantized, quant_config):
        super().__init__()
        self.model:PreTrainedModel = model
        self.model_type:str = model_type
        self.is_quantized:bool = is_quantized
        self.search_result = None
        self.quant_config:dict = quant_config
    
    def to(self, device: str):
        return self.model.to(device)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @torch.no_grad()
    def quantize(self, tokenizer=None, quant_config={}, n_samples=128, seqlen=512,
                       auto_scale=True, mse_range=True, run_search=True, run_quant=True,
                       calib_data="pileval"):
        self.quant_config = quant_config

        if run_search:
            self.search_result = self._awq_search(tokenizer, quant_config, n_samples=n_samples, seqlen=seqlen,
                       auto_scale=auto_scale, mse_range=mse_range, calib_data=calib_data)
        
        if run_quant:
            self._awq_quant()
            self.is_quantized = True
    
    
    def _awq_quant(self):
        assert self.quant_config["zero_point"], "We only support zero_point quantization now."
        layers = self.get_model_layers(self.model)

        # Run AWQ quantization
        for i in tqdm(range(len(layers)), desc="AWQ Quantization"):
            layer = layers[i]
            named_linears = get_named_linears(layer)
            self._scale_activations(self, layer)

            for name, module in named_linears.items():
                module.cuda()

                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, 
                    get_scale_zp=True, 
                    **self.quant_config
                )

                scales = scales.t().contiguous()
                zeros = zeros.t().contiguous()

                q_linear = WQLinear.from_linear(
                    module, 
                    self.quant_config['w_bit'], 
                    self.quant_config['q_group_size'], 
                    False, 
                    scales, 
                    zeros
                )

                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()
            
            torch.cuda.empty_cache()
            gc.collect()
    
    def _awq_search(self, tokenizer, quant_config, n_samples=128, seqlen=512,
                       auto_scale=True, mse_range=True, calib_data="pileval"):
        layers = self.get_model_layers(self.model)

        samples = get_calib_dataset(
            data=calib_data, tokenizer=tokenizer, n_samples=n_samples, block_size=seqlen)
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        layers[0] = layers[0].cuda()
        self.move_embed(self.model, "cuda")
        
        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps.append(inp)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        layers[0] = Catcher(layers[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        del samples
        layers[0] = layers[0].module  # restore
        inps = inps[0]

        layers[0] = layers[0].cpu()
        self.move_embed(self.model, "cpu")
        
        gc.collect()
        torch.cuda.empty_cache()
        awq_results = {
            "scale": [],
            "clip": [],
        }

        # Run AWQ search layer by layer
        for i in tqdm(range(len(layers)), desc="AWQ Search"):
            layer = layers[i]
            layer = layer.cuda()
            named_linears = get_named_linears(layer)

            # firstly, get input features of all linear layers
            def cache_input_hook(m, x, y, name, feat_dict):
                x = x[0]
                x = x.detach().cpu()
                feat_dict[name].append(x)

            input_feat = defaultdict(list)
            handles = []
            for name in named_linears:
                handles.append(named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name,
                                    feat_dict=input_feat)))
            inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
            # get output as next layer's input
            inps = layer(inps, **layer_kwargs)[0]
            for h in handles:
                h.remove()
            # now solve for scaling and clipping
            input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

            # Clear GPU memory
            torch.cuda.empty_cache()

            if auto_scale:  # if it applies, we should also modify the input_feat with scales
                scales_list = auto_scale_block(
                    self,
                    layer,
                    layer_kwargs,
                    quant_config=quant_config,
                    input_feat=input_feat,
                )

                apply_scale(layers[i], scales_list, input_feat_dict=input_feat)

                # append prefix to make names global
                awq_results["scale"] += append_str_prefix(scales_list, get_op_name(self.model, layer) + ".")

            # Clear GPU memory
            torch.cuda.empty_cache()
            
            if mse_range:
                clip_list = auto_clip_block(
                    layer,
                    quant_config=quant_config,
                    input_feat=input_feat
                )

                apply_clip(layer, clip_list)
                # append prefix to make names global
                awq_results["clip"] += append_str_prefix(clip_list, get_op_name(self.model, layer) + ".")

            layer = layer.cpu()
            # Haotian: check activation replacement
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()
        
        return awq_results

    def save_quantized(self, save_dir):
        def _save_files(save_dir, model_name, model):
            class EmptyModule(nn.Module):
                def __init__(self): super(EmptyModule, self).__init__()
                def forward(self, x): return x

            # Save model fiels without search results
            self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())

            # Remove empty module
            os.remove(f'{save_dir}/pytorch_model.bin')

            # Save search results
            torch.save(model, f'{save_dir}/{model_name}')

            # Save config
            with open(f'{save_dir}/quant_config.json', 'w+') as file:
                file.write(json.dumps(self.quant_config, indent=4))

        save_dir = save_dir[:-1] if save_dir[-1] == '/' else save_dir

        # Save model
        if self.search_result is None or self.is_quantized:
            model_name = f'awq_model_w{self.quant_config["w_bit"]}_g{self.quant_config["q_group_size"]}.pt'
            _save_files(save_dir, model_name, self.model.state_dict())
        else:
            model_name = 'awq_model_search_result.pt'
            _save_files(save_dir, model_name, self.search_result)
        
    @classmethod
    def from_pretrained(self, model_path, model_type, torch_dtype: torch.dtype = torch.float16, 
                        trust_remote_code=True, safetensors=False):
        return self.from_quantized(
            model_path, 
            model_type, 
            model_filename='', 
            max_new_tokens=None,
            device='balanced', 
            torch_dtype=torch_dtype, 
            trust_remote_code=trust_remote_code, 
            safetensors=safetensors,
            is_quantized=False
        )

    @classmethod
    def from_quantized(self, model_path, model_type, model_filename, max_new_tokens=None,
                       device='balanced', torch_dtype=torch.float16, trust_remote_code=True, 
                       safetensors=False, is_quantized=True, fuse_layers=False):
        # [STEP 1] Download model if path is not a directory
        if not os.path.isdir(model_path):
            ignore_patterns = ["*msgpack*", "*h5*"]
            if safetensors:
                ignore_patterns.extend(["*.pt", "*.bin"])
            else:
                ignore_patterns.append("*safetensors*")

            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
        
        # TODO: Better naming, model_filename becomes a directory
        model_filename = model_path + f'/{model_filename}'

        # [STEP 2] Load config and set sequence length
        # TODO: Create BaseAWQConfig class
        quant_config_path = f'{model_path}/quant_config.json'
        if os.path.exists(quant_config_path):
            with open(quant_config_path, 'r') as file:
                quant_config = json.loads(file.read())
        else:
            # Default config that works for most models
            quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4}
        
        # Load model config and set max generation length
        if max_new_tokens is None and hasattr(self, 'max_new_tokens_key'):
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            config.max_new_tokens = getattr(config, self.max_new_tokens_key)
        else:
            max_new_tokens = 2048 if max_new_tokens is None else max_new_tokens
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            config.max_new_tokens = max_new_tokens
        
        # [STEP 3] Load model
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        
        # Only need to replace layers if a model is AWQ quantized
        if is_quantized:
            # Prepare WQLinear layers, replace nn.Linear
            self._load_quantized_modules(self, model, quant_config)
        
        model.tie_weights()

        # Load model weights
        if is_quantized:
            model = load_checkpoint_and_dispatch(model, model_filename, device_map=device, no_split_module_classes=[self.layer_type])

            if fuse_layers:
                self.fuse_layers(model)

        else:
            # If not quantized, must load with AutoModelForCausalLM
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=[self.layer_type], 
                dtype=torch_dtype
            )
            
            del model
            
            # Load model weights
            model = AutoModelForCausalLM.from_pretrained(
                model_filename, device_map=device_map, offload_folder="offload", offload_state_dict=True, torch_dtype=torch_dtype, use_safetensors=safetensors
            )
            model.eval()

        return self(model, model_type, is_quantized=is_quantized, quant_config=quant_config)

    def _load_quantized_modules(self, model, quant_config):
        # Real quantization of weights
        assert quant_config["zero_point"], "We only support zero_point quantization now."
        
        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # Replace activation functions
            self._scale_activations(self, layer)

            # Replace nn.Linear with WQLinear
            for name, module in named_linears.items():
                q_linear = WQLinear.from_linear(
                    module, quant_config['w_bit'], quant_config['q_group_size'], True)
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer)

        if scale_dict['is_scalable']:
            if not isinstance(scale_dict['scale_layer'], ScaledActivation):
                param = next(layer.parameters())

                # get activation scale
                scale_like = torch.ones(scale_dict['scale_shape'], dtype=param.dtype, device=param.device)

                # scale activation
                scaled_act = ScaledActivation(scale_dict['scale_layer'], scale_like)
                set_op_by_name(layer, scale_dict['scale_name'], scaled_act)