import gc
import torch
import functools
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

from awq.utils.calib_data import get_calib_dataset
from transformers import AutoModelForCausalLM, AutoConfig
from awq.quantize.quantizer import pseudo_quantize_tensor
from awq.quantize.qmodule import WQLinear, ScaledActivation
from awq.quantize.auto_clip import auto_clip_block, apply_clip
from awq.quantize.auto_scale import auto_scale_block, apply_scale
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from awq.utils.module import append_str_prefix, get_op_name, get_named_linears, set_op_by_name

class BaseAWQForCausalLM:
    @torch.no_grad()
    def quantize(self, model, tokenizer=None, w_bit=4, q_config={}, n_samples=128, seqlen=512,
                       auto_scale=True, mse_range=True, run_search=False, run_quant=True,
                       calib_data="pileval"):
        search_result = None

        if run_search:
            search_result = self._awq_search(model, tokenizer, w_bit, q_config, n_samples=n_samples, seqlen=seqlen,
                       auto_scale=auto_scale, mse_range=mse_range, calib_data=calib_data)
        
        if run_quant:
            self._awq_quant(model, w_bit, q_config)
        
        return search_result
    
    
    def _awq_quant(self, model, w_bit, q_config):
        assert q_config["zero_point"], "We only support zero_point quantization now."
        layers = self.get_model_layers(model)

        # Run AWQ quantization
        for i in tqdm(range(len(layers)), desc="AWQ Quantization"):
            layer = layers[i]
            named_linears = get_named_linears(layer)
            self._scale_activations(layer)

            for name, module in named_linears.items():
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config)
                scales = scales.t().contiguous()
                zeros = zeros.t().contiguous()
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config['q_group_size'], False, scales, zeros)
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()
            
            torch.cuda.empty_cache()
            gc.collect()
    
    def _awq_search(self, model, tokenizer, w_bit, q_config, n_samples=128, seqlen=512,
                       auto_scale=True, mse_range=True, calib_data="pileval"):
        layers = self.get_model_layers(model)

        samples = get_calib_dataset(
            data=calib_data, tokenizer=tokenizer, n_samples=n_samples, block_size=seqlen)
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        layers[0] = layers[0].cuda()
        self.move_embed(model, "cuda")
        
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
            model(samples.to(next(model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        del samples
        layers[0] = layers[0].module  # restore
        inps = inps[0]

        layers[0] = layers[0].cpu()
        self.move_embed(model, "cpu")
        
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
                    layer, layer_kwargs,
                    w_bit=w_bit, q_config=q_config,
                    input_feat=input_feat,
                )
                # apply_scale(layer, scales_list, input_feat_dict=input_feat)
                apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
                # append prefix to make names global
                awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

            # Clear GPU memory
            torch.cuda.empty_cache()
            
            if mse_range:
                clip_list = auto_clip_block(layer,
                                w_bit=w_bit, q_config=q_config,
                                input_feat=input_feat,)
                apply_clip(layer, clip_list)
                # append prefix to make names global
                awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

            layer = layer.cpu()
            # Haotian: check activation replacement
            del input_feat
            gc.collect()
            torch.cuda.empty_cache()
        
        return awq_results

    def save_quantized():
        pass

    def from_pretrained():
        pass

    def from_quantized(self, model_path, quant_path, w_bit, q_config, device, trust_remote_code=True):
        # Load config
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16, trust_remote_code=True)

        # Initialize layers
        assert q_config["zero_point"], "We only support zero_point quantization now."
        layers = self.get_model_layers(model)
        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]
            named_linears = get_named_linears(layer)
            self._scale_activations(layer)

            for name, module in named_linears.items():
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config['q_group_size'], True)
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            
            torch.cuda.empty_cache()
            gc.collect()
        
        model.tie_weights()

        model = load_checkpoint_and_dispatch(model, quant_path, device_map="balanced")

        return model
    
    def _scale_activations(self, layer):
        act_function = self.get_act_from_layer(layer)

        if act_function is not None and not isinstance(act_function, ScaledActivation):
            param = next(layer.parameters())

            # get activation scale
            scale_dict = self.get_act_for_scaling(layer)
            scale_like = torch.ones(scale_dict['scale_shape'], dtype=param.dtype, device=param.device)

            # scale activation
            scaled_act = ScaledActivation(scale_dict['scale_layer'], scale_like)
            set_op_by_name(layer, scale_dict['scale_name'], scaled_act)