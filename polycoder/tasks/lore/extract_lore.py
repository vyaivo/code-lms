"""LORE loop dataset"""

import deepspeed
from functools import partial
import torch

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from megatron import mpu, print_rank_0
from megatron.initialize import initialize_megatron
from megatron.neox_arguments import NeoXArgs
from megatron.model import (
    GPT2Finetune,
    GPT2ModelPipe,
    SoftEmbedding,
)
from megatron.utils import get_total_params, get_ltor_masks_and_position_ids, Timers
from megatron.checkpointing import load_checkpoint

from tasks.data_utils import build_dataloaders, get_batch, get_batch_pipe, broadcast_data_list
from tasks.extract_utils import extract_loop

print('VAV DEBUG:', os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_lore_lm import build_lore_dataset


def extract_lore_representations(neox_args):
    # Find and load the dataset
    output_path = "/lm_data/extracted/small-bpe/"
    get_lore_dataset = partial(build_lore_dataset, data_path=neox_args.data_path,
                               tokenizer=neox_args.tokenizer)
    # Initialize GPT-NeoX/Megatron
    timers = Timers(use_wandb=False, tensorboard_writer=neox_args.tensorboard_writer)
    initialize_megatron(neox_args=neox_args)
    # Do the data loader and model setup
    neox_args.iteration = 0
    timers("train/valid/test data iterators").start()
    data_split_names = ["train", "val", "test"]
    seq_length_bins = [2048, 1024, 768, 512, 0]
    data_loaders = build_dataloaders(neox_args, get_dataset_fn=get_lore_dataset,
                                     pad_sequences=True, length_bins=seq_length_bins,
                                     drop_last=False, shuffle=False)
    # Grab n samples in each length bin
    bin_n = [{}, {}, {}]
    for i, d in enumerate(data_loaders):
        db_dict = d.batch_sampler.sampler.data_buckets
        for n in range(len(seq_length_bins) - 1):
            bin_n[i][seq_length_bins[n]] = len(db_dict[n+1])
    timers("train/valid/test data iterators").stop()
    timers.log(["train/valid/test data iterators"])

    timers("model setup").start()
    data_keys = ["input_ids", "source"]
    # eval_batch_fn = partial(get_batch, keys=data_keys, custom_batch_fn=lore_batch_fn)
    eval_batch_fn = partial(get_batch_pipe, keys=data_keys, custom_batch_fn=lore_batch_fn)
    model = setup_model(neox_args, pipeline_batch_fn=eval_batch_fn)
    timers("model setup").stop()
    # print(model.use_cache)

    print_rank_0("done with setups ...")
    timers.log(["model setup"])

    # Maybe we should just keep two iterators???
    # When we run the attention layer, get an error: CUDA error:
    # CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`

    # Extract for each split in the dataset
    for i, (split, loader) in enumerate(zip(data_split_names, data_loaders)):
        output_path = os.path.join(output_path, split)
        os.makedirs(output_path, exist_ok=True)
        extract_loop(neox_args, timers, model, iter(loader), iter(loader), eval_batch_fn,
                     output_dir=output_path, batch_data_key="source",
                     samples_per_split_bin=bin_n[i])


def lore_batch_fn(neox_args, keys, data, datatype=torch.int64, tokenizer=None):
    """Support function for get_batch / get_batch pipe (to avoid code repetition). See tasks/data_utils.py."""
    data_b = mpu.broadcast_data(["input_ids"], data, datatype)
    # data_x = broadcast_data_list(["source"], data, is_tensor=False, data_type=str)

    # Unpack.
    tokens = data_b["input_ids"].long().squeeze(1).contiguous()

    if tokens.ndim > 2:
        # this is to debug an intermittent error that we're getting in batching
        try:
            tokens = data_b["input_ids"].long().squeeze().contiguous()
        except Exception as e:
            print(e)

    # labels = data_x["source"]
    labels = torch.zeros((1, 2)).to(datatype)  # dummy values
    data_2 = mpu.broadcast_data(["labels"], {'labels': labels}, datatype)

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, 0, True
    )
    return tokens, (labels, loss_mask), attention_mask, position_ids


def get_model(neox_args, pipeline_batch_fn, inference=True, get_key_value=False):
    """Build the model."""

    print_rank_0("building GPT2 model ...")

    # Build model on cpu.
    # model = GPT2ModelPipe(
    model = GPT2Finetune(
        neox_args=neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=mpu.get_topology(),
        inference=inference,
        get_key_value=get_key_value,
    )

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


def setup_model(neox_args, pipeline_batch_fn=None, inference=True, get_key_value=False):
    """Setup model and optimizer."""
    model = get_model(
        neox_args=neox_args, pipeline_batch_fn=pipeline_batch_fn,
        inference=inference, get_key_value=get_key_value
    )
    ## VAV DEBUG ##
    for name, p in model.named_parameters():
        if p.requires_grad:
            print_rank_0(name, p.shape)
    ## VAV DEBUG ##

    if neox_args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        optimizer = None
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
        if neox_args.is_pipe_parallel:
            model.set_has_attention_mask(True)
            model.set_batch_fn(model.module._megatron_batch_fn)
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
        print_rank_0(f"Loaded checkpoint")
    else:
        # pass
        raise ValueError("Need a checkpoint to load from to extract representations!")

    return model


if __name__ == "__main__":

    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

    extract_lore_representations(neox_args)
