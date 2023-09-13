# Adapted from Megatron-LM
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Classification model."""

import torch

from megatron import get_args, print_rank_last
from megatron.model.enums import AttnMaskType
from megatron.model.bert_model import bert_extended_attention_mask, bert_position_ids
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule

# From GPTNeoX
from megatron.model import (
    GPT2ModelPipe,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
)


class Classification(MegatronModule):

    def __init__(self,
                 config,
                 num_classes,
                 num_tokentypes=2,
                 pre_process=True,
                 post_process=True):
        super().__init__(config=config, share_embeddings_and_output_weights=False)
        args = get_args()

        self.num_classes = num_classes
        self.post_process = post_process

        self.language_model = get_model(neox_args)
        self._language_model_key = 'polycoder'

        # Multi-choice head.
        if self.post_process:
            self.classification_dropout = torch.nn.Dropout(args.hidden_dropout)
            # Next 4 lines taken from Megatron-LM model.utils.get_linear_layer
            self.classification_head = torch.nn.Linear(args.hidden_size,
                                                       self.num_classes)
            with torch.no_grad():
                self.classification_head.bias.zero_()
            self._classification_head_key = 'classification_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, model_input, attention_mask, tokentype_ids=None):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)
        input_ids = model_input
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids
        )

        if self.post_process:
            _, pooled_output = lm_output
            classification_output = self.classification_dropout(pooled_output)
            classification_logits = self.classification_head(classification_output)

            # Reshape back to separate choices.
            classification_logits = classification_logits.view(-1, self.num_classes)

            return classification_logits
        return lm_output

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
                print_rank_last('***WARNING*** could not find {} in the checkpoint, '
                                'initializing to random'.format(
                                    self._classification_head_key))


def init_model(model, neox_args):
    print_rank_0(model)
    for name, p in model.named_parameters():
        if p.requires_grad:
            print_rank_0(name, p.shape)

    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        _model_params = None
        _lr_scheduler = None

        model, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=None,
            args=neox_args,
            lr_scheduler=_lr_scheduler,
            dist_init_required=False,
            model_parameters=_model_params,
            config_params=neox_args.deepspeed_config,
            mpu=mpu if not neox_args.is_pipe_parallel else None,
        )
        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')
#        raise ValueError("Stopping here so we can check!")  ## VAV DEBUG
        if neox_args.is_pipe_parallel:
            model.set_has_attention_mask(True)
            model.set_batch_fn(model.module._megatron_batch_fn)
    else:
        raise ValueError("Must be using deepspeed to run neox")

    if neox_args.load is not None:
        neox_args.iteration = load_checkpoint(
            neox_args=neox_args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            inference=inference,
            iteration=iteration,
        )
        print_rank_0(
            f"Loading checkpoint and starting from iteration {neox_args.iteration}"
        )
    else:
        raise ValueError("Must specify a checkpoint to load to finetune a downstream task")
    return model


def get_model(neox_args, inference=True, get_key_value=True):
    """Build the model."""

    print_rank_0("building GPT2 model ...")

    # Build model on cpu.
    model = GPT2ModelPipe(
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

    model = init_model(model, neox_args)

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


