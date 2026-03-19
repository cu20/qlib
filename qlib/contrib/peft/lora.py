#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def add_loras_(parent_module: nn.Module, lora_rank: int, scaling: float = 1, merge_weights: bool = True):
    """
    Examples:
        backbone = add_loras(backbone)

    Parameters
    ----------
    parent_module: nn.Module
        your backbone
    lora_rank: int
        the dimension of the bottleneck, a hyperparameter
    scaling: float
        scaling factor, a hyperparameter
    merge_weights: bool
        whether to merge lora into the backbone (to reduce the inference cost)

    Returns
    -------
    A new backbone with (multiple) lora layers

    """
    for name, module in parent_module.named_children():
        # Target at MASTER
        if isinstance(module, nn.Linear) and 'qtrans' in name or 'vtrans' in name:
            add_lora_(parent_module, name.split('.')[-1], r=lora_rank, scaling=scaling,
                      load_weights=True, merge_weights=merge_weights)
        else:
            add_loras_(module, lora_rank, scaling, merge_weights)
    return parent_module


def add_lora_(parent_module: nn.Module, module_name: str, r: int, scaling: float,
              merge_weights=True, load_weights=True, **kwargs):
    old_module = getattr(parent_module, module_name)
    if isinstance(old_module, nn.Linear):
        new_module = Linear(in_features=old_module.in_features, out_features=old_module.out_features,
                            bias=old_module.bias is not None, r=r, scaling=scaling,
                            device=old_module.weight.device, dtype=old_module.weight.dtype,
                            merge_weights=merge_weights, **kwargs)
    else:
        raise NotImplementedError

    if load_weights:
        new_module.load_state_dict(old_module.state_dict(), strict=False)
    setattr(parent_module, module_name, new_module)


class LoRALayer():
    def __init__(
            self,
            r: int,
            scaling: float,
            merge_weights: bool,
    ):
        self.r = r
        self.scaling = scaling
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None, dtype=None,
            r: int = 0,
            scaling: float = 1,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        LoRALayer.__init__(self, r=r, scaling=scaling, merge_weights=merge_weights)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # Freezing the pre-trained weight matrix while the bias term is tunable with a little cost
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= self.lora_B @ self.lora_A * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += self.lora_B @ self.lora_A * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.linear(x, self.weight + self.lora_B @ self.lora_A * self.scaling, bias=self.bias)
        else:
            return F.linear(x, self.weight, bias=self.bias)

