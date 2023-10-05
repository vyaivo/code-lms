# Adapted from Megatron-LM code, which is:
#    Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Classification model."""

import glob
import os
import torch

from functools import partial
from megatron import mpu, print_rank_0
# from megatron.training import get_batch_pipe
from megatron.checkpointing import load_checkpoint

# from megatron.model.enums import AttnMaskType
# from megatron.model.bert_model import bert_extended_attention_mask, bert_position_ids
# from megatron.model.utils import get_linear_layer
# from megatron.model.utils import init_method_normal
# from megatron.model.utils import scaled_init_method_normal
# from .module import MegatronModule

from deepspeed.pipe import LayerSpec

# From GPTNeoX
from megatron.model import (
    GPT2Finetune,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
)


# def get_language_model(neox_args, inference=False, get_key_value=True):
#     """Build the model."""
#
#     print_rank_0("building GPT2 model ...")
#
#     # Build model on cpu.
#     model = GPT2Finetune(
#         neox_args=neox_args,
#         num_tokentypes=0,
#         parallel_output=False,  # VAV DEBUG TRAIN ERROR
#         topology=mpu.get_topology(),
#         inference=inference,
#         get_key_value=get_key_value,
#     )
#
#     ### soft prompt tuning stuff ###
#     if neox_args.soft_prompt_tuning is not None and neox_args.soft_prompt_tuning.get(
#         "enabled", False
#     ):
#         raise NotImplementedError
#
#     if not neox_args.is_pipe_parallel:
#         # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
#         model = model.to_sequential()
#     else:
#         # This is a hack to give us a reference to get_batch_pipe from within training.py
#         # We need to call model.set_batch_fn after deepspeed.initialize
#         model._megatron_batch_fn = partial(get_batch_pipe, neox_args=neox_args)
#
#     if neox_args.deepspeed:
#         # DeepSpeed handles CUDA, FP16, and DDP components.
#         return model
#     else:
#         raise ValueError("Must be using deepspeed to run neox")


class ClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size, num_classes, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, lm_output):
        dropped = self.dropout(lm_output)
        classif_output = self.head(dropped)
        return classif_output


class Classification(GPT2Finetune):

    def __init__(self,
                 neox_args,
                 num_classes,
                 post_process=True, **kwargs):
                 # load_lm_checkpoint=None):
        self.num_classes = num_classes
        self.post_process = post_process

        super().__init__(neox_args, **kwargs)
        self.add_classif_specs()

    def add_classif_specs(self):
        if self.post_process:
            # Linear layer for pooling
            self.specs.append(LayerSpec(
                    torch.nn.Linear(self.neox_args.hidden_size, self.neox_args.hidden_size))
            )
        # Actual classification head
        self.specs.append(
            LayerSpec(
                ClassificationHead,
                self.neox_args.hidden_size, self.num_classes, self.neox_args.hidden_dropout
            )
        )

    # def load_checkpoint(self, neox_args):
    #     load_dir = neox_args.load
    #     latest_path = os.path.join(load_dir, 'latest')
    #     if os.path.isfile(latest_path):
    #         with open(latest_path, 'r') as fd:
    #             tag = fd.read().strip()
    #
    #     curr_ckpt_path = os.path.join(load_dir, tag)
    #     ckpt_list = glob.glob(curr_ckpt_path + '/layer_*-model_00-model_states.pt')
    #
    #     for idx, layer in enumerate(self.language_model.sequential):
    #         layer_ckpt_path = curr_ckpt_path + f'/layer_{idx:02d}-model_00-model_states.pt'
    #         if layer_ckpt_path in ckpt_list:
    #             layer.load_state_dict(torch.load(layer_ckpt_path, map_location=lambda storage, loc: storage), strict=True)
    #             print(f"Loaded {layer_ckpt_path} to language model!")

    def forward_step(self, lm_output, pool_seq_index=0):

        if self.post_process:
            # Pools over a specific token -- default is start of the sequence
            # print_rank_0("VAV DEBUG:", lm_output.shape)  # batch x seq x emb_size
            pooled_output = self.pooler(lm_output[:, pool_seq_index, :])
            classif_input = torch.tanh(pooled_output)
        else:
            classif_input = lm_output

        classification_output = self.classification_dropout(classif_input)
        classification_logits = self.classification_head(classification_output)

        # Reshape back to separate choices.
        # classification_logits = classification_logits.view(-1, self.num_classes)
        return classification_logits

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        if self.post_process:
            state_dict_[self._classification_head_key] \
                = self.classification_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        if state_dict is not None:
            self.language_model.load_state_dict(state_dict, strict=strict)
        if self.post_process:
            if self._classification_head_key in state_dict:
                self.classification_head.load_state_dict(
                    state_dict[self._classification_head_key], strict=strict)
            else:
                print_rank_0('***WARNING*** could not find {} in the checkpoint, '
                             'initializing to random'.format(
                             self._classification_head_key))




