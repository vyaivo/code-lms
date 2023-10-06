"""LORE loop dataset"""

import deepspeed
from functools import partial
import math
import torch

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from megatron.neox_arguments import NeoXArgs
from megatron import mpu, print_rank_0
from megatron.training import evaluate, get_optimizer, get_learning_rate_scheduler
from megatron.model import (
    GPT2ModelPipe,
    SoftEmbedding,
)
from megatron.utils import get_total_params

from megatron.checkpointing import load_checkpoint
from tasks.finetune_utils import finetune
from tasks.data_utils import build_data_iterators, get_batch

from data_lore_lm import build_lore_dataset


def extract_lore_representations(neox_args):
    
    output_dir = "/lm_data/extracted/"
    get_lore_dataset = partial(build_lore_dataset, data_path=neox_args.finetune_data_path,
                               tokenizer=neox_args.tokenizer)
    data_iterators = build_data_iterators(neox_args, get_dataset_fn=get_lore_dataset,
                                          pad_sequences=True, length_bins=[2048, 1024, 512, 0])
    data_keys = ["input_ids", "source"]
    eval_batch_fn = partial(get_batch, keys=data_keys, custom_batch_fn=lore_batch_fn)
    model = setup_model(neox_args)

    extract_loop(neox_args, timers, model, data_iterator, eval_batch_fn, output_dir)


def lore_batch_fn(neox_args, tokenizer, keys, data, datatype=torch.int64):
    """Support function for get_batch / get_batch pipe (to avoid code repetition). See tasks/data_utils.py."""
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["input_ids"].long().squeeze()

    if tokens_.ndim > 2:
        # this is to debug an intermittent error that we're getting in batching
        print(data)
        print(tokens_.shape)
        import pdb; pdb.set_trace()

    labels = tokens_[:, 1:]
    tokens = tokens_[:, :-1].contiguous()
    
    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, 0, True
    )

    return tokens, (labels, loss_mask), attention_mask, position_ids


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
        raise ValueError("Please disable pipe parallelism when extracting representations")

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def setup_model(neox_args, inference=True, get_key_value=True, iteration=None):
    """Setup model and optimizer."""
    model = get_model(
        neox_args=neox_args, inference=inference, get_key_value=get_key_value
    )
    ## VAV DEBUG ##
    for name, p in model.named_parameters():
        if p.requires_grad:
            print_rank_0(name, p.shape)
    ## VAV DEBUG ##

    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        assert optimizer is None
        _model_params = None
        _lr_scheduler = None

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=neox_args,
            lr_scheduler=_lr_scheduler,
            dist_init_required=False,
            model_parameters=_model_params,
            config_params=neox_args.deepspeed_config,
            mpu=None,
        )
        model.total_params = get_total_params(model.module)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')
    else:
        raise ValueError("Must be using deepspeed to run neox")

    if neox_args.load is not None:
        _ = load_checkpoint(
            neox_args=neox_args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            inference=True,
            iteration=None,
        )
        print_rank_0(
            f"Loading checkpoint")
    else:
        raise ValueError("Need a checkpoint to load from to extract representations!") 

    return model


if __name__ == "__main__":

    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

    extract_lore_representations(neox_args)
