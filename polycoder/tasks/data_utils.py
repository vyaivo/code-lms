# VAV taken from current megatron repo
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

""" Tasks data utility."""

from functools import partial
import numpy as np
import torch
import re

from megatron import mpu


def collate_fn_pad(batch, data_keys_to_collate=[]):
    from torch.utils.data._utils.collate import default_collate

    elem = batch[0]
    elem_type = type(elem)

    try:
        out_dict = {}
        for key in elem:
            item_list = [d[key] for d in batch]
            if key in ['text', 'input_ids', 'attention_mask']:
                # Custom behavior: pad this baby!
                lengths = torch.IntTensor([sample.size(dim=0) for sample in item_list])
                padded_item = torch.nn.utils.rnn.pad_sequence(item_list, batch_first=True)
                out_dict.update({'lengths': lengths})
                out_dict.update({key: padded_item})
            elif key in data_keys_to_collate:
                # Default collate behavior for a dictionary, according to pytorch 2.0.0
                out_dict.update({key: default_collate(item_list)})
            else:
                # Custom behavior for fields that are lists of lists
                out_dict.update({key: item_list})
        return elem_type(out_dict)
    except TypeError:
        raise ValueError(f"This mapping type {elem_type} may not support `__init__(iterable)`.")


def make_data_loader_with_padding(dataset, neox_args):
    """Build dataloader given an input dataset. Minor modification of megatron.data.data_utils"""
    from megatron.data.samplers import DistributedBatchSampler

    if dataset is None:
        return None
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = neox_args.batch_size * world_size
    num_workers = neox_args.num_workers

    collate_fn = partial(collate_fn_pad, data_keys_to_collate=neox_args.data_keys_collate)

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn,
                                       pin_memory=True)


def clean_text(text):
    """Remove new lines and multiple spaces and adjust end of sentence dot."""

    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    for _ in range(3):
        text = text.replace(' . ', '. ')

    return text


def build_sample(ids, types, paddings, label, unique_id):
    """Convert to numpy and return a sample consumed by the batch producer."""

    ids_np = np.array(ids, dtype=np.int64)
    types_np = np.array(types, dtype=np.int64)
    paddings_np = np.array(paddings, dtype=np.int64)
    sample = ({'text': ids_np,
               'types': types_np,
               'padding_mask': paddings_np,
               'label': int(label),
               'uid': int(unique_id)})

    return sample


def build_tokens_types_paddings_from_text(text_a, text_b,
                                          tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    text_a_ids = tokenizer.tokenize(text_a)
    text_b_ids = None
    if text_b is not None:
        text_b_ids = tokenizer.tokenize(text_b)

    return build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids,
                                                max_seq_length, tokenizer.cls,
                                                tokenizer.sep, tokenizer.pad)


def build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids, max_seq_length,
                                         cls_id, sep_id, pad_id):
    """Build token types and paddings, trim if needed, and pad if needed."""

    ids = []
    types = []
    paddings = []

    # [CLS].
    ids.append(cls_id)
    types.append(0)
    paddings.append(1)

    # A.
    len_text_a = len(text_a_ids)
    ids.extend(text_a_ids)
    types.extend([0] * len_text_a)
    paddings.extend([1] * len_text_a)

    # [SEP].
    ids.append(sep_id)
    types.append(0)
    paddings.append(1)

    # B.
    if text_b_ids is not None:
        len_text_b = len(text_b_ids)
        ids.extend(text_b_ids)
        types.extend([1] * len_text_b)
        paddings.extend([1] * len_text_b)

    # Cap the size.
    trimmed = False
    if len(ids) >= max_seq_length:
        max_seq_length_m1 = max_seq_length - 1
        ids = ids[0:max_seq_length_m1]
        types = types[0:max_seq_length_m1]
        paddings = paddings[0:max_seq_length_m1]
        trimmed = True

    # [SEP].
    if (text_b_ids is not None) or trimmed:
        ids.append(sep_id)
        if text_b_ids is None:
            types.append(0)
        else:
            types.append(1)
        paddings.append(1)

    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([pad_id] * padding_length)
        types.extend([pad_id] * padding_length)
        paddings.extend([0] * padding_length)

    return ids, types, paddings
