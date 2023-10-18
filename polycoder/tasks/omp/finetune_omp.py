"""OpenMP pragma completion task -- finetuning."""

import deepspeed
from functools import partial
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
from tasks.data_utils import build_dataloaders, get_batch, get_batch_pipe

from data_omp import build_omp_dataset, omp_batch_fn


def omp_generation(neox_args):

    data_iter_fn = partial(build_dataloaders,
                           get_dataset_fn=partial(build_omp_dataset, rebuild=True),
                           pad_sequences=True, length_bins=[2048, 1024, 0])

    data_keys = ["input_ids"]
    pipe_batch_fn = partial(get_batch_pipe, keys=data_keys, custom_batch_fn=omp_batch_fn, neox_args=neox_args)
    eval_batch_fn = partial(get_batch, keys=data_keys, custom_batch_fn=omp_batch_fn)
    model_setup = partial(setup_model_and_optimizer, train_batch_fn=pipe_batch_fn)

    """Finetune/evaluate."""
    finetune(neox_args, model_setup, data_iter_fn, loss_fn, evaluate, custom_batch_fn=eval_batch_fn)


def loss_fn(logits, labels):
    if isinstance(labels, tuple) and len(labels) == 2:
        lm_labels, _ = labels
    else:
        lm_labels = labels
    return torch.nn.functional.cross_entropy(logits.permute((0, 2, 1)), lm_labels, ignore_index=0)


def get_model(neox_args, pipeline_batch_fn, inference=False, get_key_value=True):
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
        model._megatron_batch_fn = partial(pipeline_batch_fn, neox_args=neox_args)

    if neox_args.deepspeed:
        # DeepSpeed handles CUDA, FP16, and DDP components.
        return model
    else:
        raise ValueError("Must be using deepspeed to run neox")


def setup_model_and_optimizer(neox_args, train_batch_fn, inference=False, get_key_value=True, iteration=None):
    """Setup model and optimizer."""
    model = get_model(
        neox_args=neox_args, pipeline_batch_fn=train_batch_fn, inference=inference, get_key_value=get_key_value
    )
    print_rank_0(model)  # VAV DEBUG
    ## VAV DEBUG ##
    for name, p in model.named_parameters():
        if p.requires_grad:
            print_rank_0(name, p.shape)
    ## VAV DEBUG ##

    optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
    lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)

    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        if neox_args.no_load_optim:
            assert optimizer is None
            _model_params = None
            _lr_scheduler = None
        else:
            _model_params = param_groups if optimizer is None else None
            _lr_scheduler = lr_scheduler

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
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
            inference=False,
            iteration=None,
        )
        print_rank_0(
            f"Loading checkpoint and starting from iteration {neox_args.iteration}"
        )
    else:
        neox_args.iteration = 0

    return model, optimizer, lr_scheduler


if __name__ == "__main__":

    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined

    omp_generation(neox_args)
