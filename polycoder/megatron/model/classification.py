# Adapted from Megatron-LM code, which is:
#    Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Classification model."""

import torch

from functools import partial
from megatron import mpu, print_rank_0
from megatron.training import get_batch_pipe
# from megatron.model.enums import AttnMaskType
# from megatron.model.bert_model import bert_extended_attention_mask, bert_position_ids
# from megatron.model.utils import get_linear_layer
# from megatron.model.utils import init_method_normal
# from megatron.model.utils import scaled_init_method_normal
# from .module import MegatronModule

# From GPTNeoX
from megatron.model import (
    GPT2Finetune,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
)


def get_language_model(neox_args, inference=False, get_key_value=True):
    """Build the model."""

    print_rank_0("building GPT2 model ...")

    # Build model on cpu.
    model = GPT2Finetune(
        neox_args=neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        inference=inference,
        get_key_value=get_key_value,
    )

    ### soft prompt tuning stuff ###
    if neox_args.soft_prompt_tuning is not None and neox_args.soft_prompt_tuning.get(
        "enabled", False
    ):
        soft_prompt = SoftEmbedding(
            neox_args,
            wte=getattr(model, "0").word_embeddings,
            n_tokens=neox_args.soft_prompt_tuning.get("n_tokens", 10),
            init_string=neox_args.soft_prompt_tuning.get("init_string", ""),
            init_range=neox_args.soft_prompt_tuning.get("init_range", 0.5),
        )
        model.insert_layers(
            layers=soft_prompt, idx=1
        )  # insert the soft prompt layer directly after the word embeddings

        # freeze everything but the soft prompt
        for name, param in model.named_parameters():
            if not "soft_embedding" in name:
                param.requires_grad = False

    if not neox_args.is_pipe_parallel:
        # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
        model = model.to_sequential()
    else:
        # This is a hack to give us a reference to get_batch_pipe from within training.py
        # We need to call model.set_batch_fn after deepspeed.initialize
        model._megatron_batch_fn = partial(get_batch_pipe, neox_args=neox_args)

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


class Classification(torch.nn.Module):

    def __init__(self,
                 neox_args,
                 num_classes,
                 post_process=True):
        super().__init__()

        self.num_classes = num_classes
        self.post_process = post_process

        self.language_model = get_language_model(neox_args)
        self._language_model_key = 'polycoder'

        # Multi-choice head.
        if self.post_process:
            self.pooler = torch.nn.Linear(neox_args.hidden_size, neox_args.hidden_size)
            self._pooler_key = 'pooler'

            self.classification_dropout = torch.nn.Dropout(neox_args.hidden_dropout)
            # Next 4 lines taken from Megatron-LM model.utils.get_linear_layer
            self.classification_head = torch.nn.Linear(neox_args.hidden_size,
                                                       self.num_classes)
            with torch.no_grad():
                self.classification_head.bias.zero_()
            self._classification_head_key = 'classification_head'

    # def set_input_tensor(self, input_tensor):
    #     """See megatron.model.transformer.set_input_tensor()"""
    #     self.language_model.set_input_tensor(input_tensor)

    def post_process_step(self, lm_output, pool_seq_index=0):

        assert self.post_process, "Called post_process_step but those layers don't exist!"

        # Pools over a specific token -- default is start of the sequence
        print_rank_0("VAV DEBUG:", lm_output.shape)  # batch x seq x emb_size
        pooled_output = self.pooler(lm_output[:, pool_seq_index, :])
        pooled_output = torch.tanh(pooled_output)

        classification_output = self.classification_dropout(pooled_output)
        classification_logits = self.classification_head(classification_output)

        # Reshape back to separate choices.
        classification_logits = classification_logits.view(-1, self.num_classes)
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

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            if self._classification_head_key in state_dict:
                self.classification_head.load_state_dict(
                    state_dict[self._classification_head_key], strict=strict)
            else:
                print_rank_0('***WARNING*** could not find {} in the checkpoint, '
                             'initializing to random'.format(
                             self._classification_head_key))




