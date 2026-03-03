from typing import Callable, Iterable
import torch
import gc
from xpmir.utils.functools import cache as cache

import logging

from utils import untuple

logging.basicConfig(level=logging.INFO)

class OutputsExtractor(torch.nn.Module):
    # SAme as InputsExtractor but to extract outputs instead
    def __init__(self, model: torch.nn.Module, layer_names: Iterable[str]):
        super().__init__()
        self.model = model
        self.outputs_store = dict()
        self.hooks_handles = list()

        for layer_name in layer_names:
            layer = dict([*self.model.named_modules()])[layer_name]
            self.hooks_handles.append(layer.register_forward_hook(self.save_outputs_hooks(layer_name)))

    @property
    def items(self):
        return self.outputs

    @property
    def items_store(self):
        return self.outputs_store
    
    def save_outputs_hooks(self, name) -> Callable:
        def hook(_, __, output):
            self.outputs_store[name] = untuple(output.detach().cpu())
        return hook
    
    def remove_hooks(self):
        for handle in self.hooks_handles:
            handle.remove()
    
    def clear_items(self):
        del self.outputs_store
        gc.collect()
        torch.cuda.empty_cache()
        self.outputs_store = dict()
    
    def forward(self, inputs):
        model_outputs = self.model(**inputs)
        return model_outputs
    

class OutputsExtractorWithResiduals(torch.nn.Module):
    # SAme as InputsExtractor but to extract outputs instead
    def __init__(self, model: torch.nn.Module, layer_names: Iterable[str]):
        super().__init__()
        self.model = model
        self.outputs_store = dict()
        self.hooks_handles = list()

        for layer_name in layer_names:
            layer = dict([*self.model.named_modules()])[layer_name]
            if "LayerNorm" in layer_name:
                residual_name = layer_name.replace("output.LayerNorm", "input.LayerNorm")
            else:
                residual_name = None
            self.hooks_handles.append(layer.register_forward_hook(self.save_outputs_hooks(layer_name, residual_name)))

    @property
    def items(self):
        return self.outputs_store
    
    def save_outputs_hooks(self, layer_name, residual_name) -> Callable:
        def hook(_, inputs, output):
            if residual_name is not None:
                self.outputs_store[residual_name] = untuple(inputs[0])
            self.outputs_store[layer_name] = untuple(output) # Store it and prepares it for backprop
        return hook
    
    def remove_hooks(self):
        for handle in self.hooks_handles:
            handle.remove()
    
    def clear_items(self):
        del self.outputs_store
        gc.collect()
        torch.cuda.empty_cache()
        self.outputs_store = dict()
    
    def forward(self, inputs):
        model_outputs = self.model(**inputs)
        return model_outputs
    
class OutputsExtractorInfoNCE(torch.nn.Module):
    # SAme as InputsExtractor but to extract outputs instead
    def __init__(self, model: torch.nn.Module, layer_names: Iterable[str]):
        super().__init__()
        self.model = model
        self.pos_outputs = dict()
        self.neg_outputs = dict()
        self.previous_pos_outputs = dict()
        self.previous_neg_outputs = dict()
        self.hooks_handles = list()
        self.tag_pos_neg: str = None

        for layer_name in layer_names:
            layer = dict([*self.model.named_modules()])[layer_name]
            if "dropout" in layer_name:
                self.hooks_handles.append(layer.register_forward_hook(self.get_attention_probs(layer_name)))
            else:
                self.hooks_handles.append(layer.register_forward_hook(self.save_outputs_hooks(layer_name)))
    
    def save_outputs_hooks(self, name) -> Callable:
        def hook(_, __, output):
            nonlocal name
            if self.tag_pos_neg == "pos":
                if not name in self.pos_outputs.keys():
                    # If self.pos_outputs is empty, it means we are at the first forward pass
                    self.pos_outputs[name] = untuple(output) # Store it and prepares it for backprop
                    self.pos_outputs[name].retain_grad()
                else:
                    # Else, we store the previous output and the current one
                    self.previous_pos_outputs[name] = self.pos_outputs[name].clone().detach().cpu()
                    self.pos_outputs[name] = untuple(output) # Store it and prepares it for backprop
                    self.pos_outputs[name].retain_grad()
                
            elif self.tag_pos_neg == "neg":
                if not name in self.neg_outputs.keys():
                    # If self.pos_outputs is empty, it means we are at the first forward pass
                    self.neg_outputs[name] = untuple(output) # Store it and prepares it for backprop
                    self.neg_outputs[name].retain_grad()
                else:
                    # Else, we store the previous output and the current one
                    self.previous_neg_outputs[name] = self.neg_outputs[name].clone().detach().cpu()
                    self.neg_outputs[name] = untuple(output) # Store it and prepares it for backprop
                    self.neg_outputs[name].retain_grad()
            else:
                raise ValueError("The tag_pos_neg attribute has not been set.")
            
        return hook
    
    def get_attention_probs(self, name) -> Callable:
        # This hook is used to get to the `attention_probs`` (see SelfAttention module in modeling_bert.py in
        # the transformers library). As this is the input of the self.dropout, we can access it through the hook.
        def hook(_, input, __):
            nonlocal name
            new_name = name.replace("dropout", "attention_probs")
            if self.tag_pos_neg == "pos":
                if not new_name in self.pos_outputs.keys():
                    # If self.pos_outputs is empty, it means we are at the first forward pass
                    self.pos_outputs[new_name] = untuple(input) # Store it and prepares it for backprop
                    self.pos_outputs[new_name].retain_grad()
                else:
                    # Else, we store the previous output and the current one
                    self.previous_pos_outputs[new_name] = self.pos_outputs[new_name].clone().detach().cpu()
                    self.pos_outputs[new_name] = untuple(input) # Store it and prepares it for backprop
                    self.pos_outputs[new_name].retain_grad()
            elif self.tag_pos_neg == "neg":
                if not new_name in self.neg_outputs.keys():
                    # If self.pos_outputs is empty, it means we are at the first forward pass
                    self.neg_outputs[new_name] = untuple(input) # Store it and prepares it for backprop
                    self.neg_outputs[new_name].retain_grad()
                else:
                    # Else, we store the previous output and the current one
                    self.previous_neg_outputs[new_name] = self.neg_outputs[new_name].clone().detach().cpu()
                    self.neg_outputs[new_name] = untuple(input) # Store it and prepares it for backprop
                    self.neg_outputs[new_name].retain_grad()
            else:
                raise ValueError("The tag_pos_neg attribute has not been set.")
            
        return hook
    
    def remove_hooks(self):
        for handle in self.hooks_handles:
            handle.remove()
    
    def clear_items(self):
        del self.neg_outputs
        del self.pos_outputs
        gc.collect()
        torch.cuda.empty_cache()
        self.neg_outputs = dict()
        self.pos_outputs = dict()
    
    def pos_forward(self, inputs_embeddings, token_type_ids, attention_mask):
        self.tag_pos_neg = "pos"
        model_outputs = self.model(inputs_embeds=inputs_embeddings, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return model_outputs

    def neg_forward(self, inputs_embeddings, token_type_ids, attention_mask):
        self.tag_pos_neg = "neg"
        model_outputs = self.model(inputs_embeds=inputs_embeddings, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return model_outputs