import glob
import numpy as np
import os
import torch

from megatron import mpu, print_rank_0
from megatron.utils import get_ltor_masks_and_position_ids
    #, average_losses_across_data_parallel_group
# from megatron.data.gpt2_dataset import _build_index_mappings

from tasks.data_utils import make_data_loader_with_padding, broadcast_data_list
from tasks.varmisuse.preprocess_data_varmisuse import build_varmisuse_datasets
import datasets as trd

# DATASET KEYS:
# 'has_bug'
# 'error_location'
# 'repair_targets'
# 'repair_candidates'
# 'provenances' - drop
# 'source_code'


def get_varmisuse_datasets(neox_args, dummy_arg=None):
    if not os.path.exists(os.path.join(neox_args.finetune_data_path, "test/dataset.arrow")):
        # Build the dataset
        build_varmisuse_datasets(neox_args)

    datasets = trd.load_from_disk(neox_args.finetune_data_path)

    for split, ds in datasets.items():
        ds.set_format(type="torch",
                      columns=['input_ids', 'error_inds', 'repair_targets',
                               'repair_candidates', 'has_bug', 'length'],
                      output_all_columns=False)
    # print(f"VAV DEBUG:", datasets["train"].features)
    tmp = datasets["validation"]

    return datasets["train"], datasets["validation"], datasets["test"]


def get_varmisuse_batch(neox_args, tokenizer, keys, data):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    # print(f"VAV DEBUG: {data['repair_targets']}")
    # print(f"VAV DEBUG: {data['repair_candidates']}")
    # print(f"VAV DEBUG: {data['input_ids'].shape}")
    data_b = broadcast_data_list(keys, data)

    # Unpack.
    tokens = data_b["input_ids"].long().contiguous()

    nbatch, seq_length = tokens.shape
    # Create masks for loss function. Mask values of 1 are kept. Default 0.

    # TODO: use torch.scatter for more efficient mask creation
    # Create mask for the error tokens
    err_mask = torch.zeros((nbatch, seq_length), dtype=torch.long, device=tokens.device)
    einds = data_b["error_inds"]
    if len(data_b["error_inds"]) != nbatch:
        assert len(einds) == 1, f"Not sure how to handle batch! Error indices: {data_b['error_inds']}"
        assert data_b["error_inds"][0].shape[0] == nbatch, \
            f"Error indices are unexpected shape {[x.shape for x in data_b['error_inds']]} for batch size {nbatch}"
        # Catch the rare case where all of these are the same length
        einds = torch.split(data_b["error_inds"][0], 1)
        einds = [r.squeeze() for r in einds]
    try:
        for b in range(nbatch):
            err_mask[b, einds[b]] = 1.
    except Exception as e:
        print(e)
        print(f'maximum index value is {einds[b].max()}')
        import pdb; pdb.set_trace()

    # Create mask for the repair candidates
    cand_mask = torch.zeros((nbatch, seq_length), dtype=torch.long, device=tokens.device)
    rcand = data_b["repair_candidates"]
    if len(data_b["repair_candidates"]) != nbatch:
        assert len(rcand) == 1, f"Not sure how to handle batch! Repair candidates: {data_b['repair_candidates']}"
        assert data_b["repair_candidates"][0].shape[0] == nbatch,\
            f"Repair candidates are unexpected shape {[x.shape for x in data_b['repair_candidates']]} for batch size {nbatch}"
        # Catch the rare case where all of these are the same length
        rcand = torch.split(data_b["repair_candidates"][0], 1)
        rcand = [r.squeeze() for r in rcand]
    try:
        for b in range(nbatch):
            cand_mask[b, rcand[b]] = 1.
    except Exception as e:
        print(e)
        print(f'maximum index value is {rcand[b].max()}')
        import pdb; pdb.set_trace()

    # Create mask for the repair targets
    if torch.all(~data_b["has_bug"]):
        target_mask = torch.zeros((nbatch, seq_length), device=tokens.device)
    else:
        rtarg = data_b["repair_targets"]
        if len(data_b["repair_targets"]) != nbatch:
            assert len(rtarg) == 1, f"Not sure how to handle batch! Repair targets: {data_b['repair_targets']}"
            # Catch the rare case where all of these are the same length
            rtarg = torch.split(data_b["repair_targets"][0], 1)
            rtarg = [r.squeeze() for r in rtarg]
        target_mask = []
        for rlab in rtarg:
            if torch.numel(rlab) == 0:
                target_mask.append(torch.zeros(seq_length, device=tokens.device))
            elif torch.numel(rlab) == 1:
                target_mask.append(torch.nn.functional.one_hot(rlab, num_classes=seq_length))
            else:
                target_mask.append(torch.nn.functional.one_hot(rlab, num_classes=seq_length).sum(dim=0))
        target_mask = torch.vstack(target_mask)
        assert target_mask.shape == (nbatch, seq_length), f"target mask has unexpected dimensions {target_mask.shape}!"

    # (error_locations, target_mask, mask, has_bug)
    labels = (err_mask, target_mask, cand_mask, data_b["has_bug"])
    # Get the masks and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens, tokenizer.eod, neox_args.eod_mask_loss
    )

    return tokens, (labels, None), attention_mask, position_ids


# def get_batch(neox_args, data_iterator):
#     """Generate a batch"""
#
#     # Items and their type.
#     keys = ['input_ids', 'error_inds', 'repair_targets', 'repair_candidates', 'has_bug']
#
#     # Broadcast data.
#     if data_iterator is not None:
#         data = next(data_iterator)
#     else:
#         data = None
#     return _get_batch(
#         neox_args=neox_args,
#         tokenizer=neox_args.tokenizer,
#         keys=keys,
#         data=data
#     )


# def get_batch_pipe(data, neox_args):
#     """A modification of get_batch() to work with the latest batch instead of an iterator. """
#     # Items and their type.
#     keys = ['input_ids', 'error_inds', 'repair_targets', 'repair_candidates', 'has_bug']
#
#     tokens, labels, _, attention_mask, position_ids = _get_batch(
#         neox_args, neox_args.tokenizer, keys, data
#     )
#     # VAV: need to return data in this order to be compatible with deepspeed pipelining code
#     # unpack data
#     return (tokens, position_ids, attention_mask), labels
#     # return tokens, labels, loss_mask, attention_mask, position_ids


# def build_train_valid_test_data_iterators(neox_args):
#     """
#     Edited from megatron.training to simply use VarMisuse dataset instead
#     """
#
#     (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)
#
#     print_rank_0('> building train, validation, and test datasets ...')
#
#     # Ensure only the first/last pipeline stages have data loaders
#     if neox_args.is_pipe_parallel:
#         is_first_stage = mpu.get_pipe_parallel_rank() == 0
#         is_last_stage = mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
#         pipe_load = is_first_stage or is_last_stage
#     else:
#         pipe_load = True
#
#     # Data loader only on rank 0 of each model parallel group.
#     if mpu.get_model_parallel_rank() == 0 and pipe_load:
#         # Number of train/valid/test samples.
#         train_iters = neox_args.train_iters
#         eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
#         test_iters = neox_args.eval_iters
#         train_val_test_num_samples = [train_iters * neox_args.train_batch_size,
#                                       eval_iters * neox_args.train_batch_size,
#                                       test_iters * neox_args.train_batch_size]
#
#         if neox_args.train_data_paths:
#             raise NotImplementedError
#         else:
#             # when just data_path is provided
#             # split dataset into train, valid and test from data_path
#             train_ds, valid_ds, test_ds = get_varmisuse_datasets(neox_args)
#
#         length_bins = [2048, 2000, 512, 128, 64, 0]
#         # Build dataloaders.
#         train_dataloader = make_data_loader_with_padding(train_ds, neox_args, length_bins)
#         valid_dataloader = make_data_loader_with_padding(valid_ds, neox_args, length_bins)
#         test_dataloader = make_data_loader_with_padding(test_ds, neox_args, length_bins)
#
#         # Flags to know if we need to do training/validation/testing.
#         do_train = train_dataloader is not None and neox_args.train_iters > 0
#         do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
#         do_test = test_dataloader is not None and neox_args.eval_iters > 0
#         # Need to broadcast num_tokens and num_type_tokens.
#         flags = torch.cuda.LongTensor(
#             [int(do_train), int(do_valid), int(do_test)])
#     else:
#         flags = torch.cuda.LongTensor([0, 0, 0])
#
#     # Broadcast num tokens.
#     if neox_args.is_pipe_parallel:
#         # Only first/last pipeline stages have data loaders, so pipeline parallelism should
#         # broadcast globally instead of just the model parallel group.
#         torch.distributed.broadcast(flags, src=0)
#     else:
#         torch.distributed.broadcast(flags,
#                                     mpu.get_model_parallel_src_rank(),
#                                     group=mpu.get_model_parallel_group())
#     neox_args.do_train = flags[0].item()
#     neox_args.do_valid = flags[1].item()
#     neox_args.do_test = flags[2].item()
#
#     # Shift the start iterations.
#     if train_dataloader is not None:
#         train_dataloader.batch_sampler.start_iter = (neox_args.iteration * neox_args.gradient_accumulation_steps) % \
#                                                     len(train_dataloader)
#         print_rank_0('setting training data start iteration to {}'.
#                      format(train_dataloader.batch_sampler.start_iter))
#     if valid_dataloader is not None:
#         start_iter_val = ((neox_args.iteration * neox_args.gradient_accumulation_steps) // neox_args.eval_interval) * \
#                          neox_args.eval_iters
#         valid_dataloader.batch_sampler.start_iter = start_iter_val % \
#                                                     len(valid_dataloader)
#         print_rank_0('setting validation data start iteration to {}'.
#                      format(valid_dataloader.batch_sampler.start_iter))
#
#     # Build iterators.
#     if train_dataloader is not None:
#         train_data_iterator = iter(train_dataloader)
#     else:
#         train_data_iterator = None
#
#     if valid_dataloader is not None:
#         valid_data_iterator = iter(valid_dataloader)
#     else:
#         valid_data_iterator = None
#
#     if test_dataloader is not None:
#         test_data_iterator = iter(test_dataloader)
#     else:
#         test_data_iterator = None
#
#     return train_data_iterator, valid_data_iterator, test_data_iterator