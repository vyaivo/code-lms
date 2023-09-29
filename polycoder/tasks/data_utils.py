""" Tasks data utilities."""

from functools import partial
import numpy as np
import torch
import re

from megatron import mpu


def unwrap_lists(list_data, data_type=torch.Tensor):
    # Handle possibly nested lists.
    if not isinstance(list_data, list):
        return list_data
    else:
        if isinstance(list_data[0], data_type):
            return list_data
        else:
            return unwrap_lists(list_data[0])


def broadcast_data_list(keys, data):
    """Broadcast data from rank zero of each model parallel group to the
        members of the same model parallel group.

        Arguments:
            keys: list of keys in the data dictionary to be broadcasted
            data: data dictionary of string keys and cpu tensor values.
        """
    # Pack on rank zero.
    if mpu.get_model_parallel_rank() == 0:
        data_list = []
        # Convert dict to list
        for k in keys:
            if isinstance(data[k], list):
                # Handle nested lists
                data[k] = unwrap_lists(data[k])
                data[k] = [d.squeeze().cuda() for d in data[k]]
            else:
                data[k] = data[k].squeeze().cuda()
            data_list.append(data[k])
    else:
        data_list = [None] * len(keys)

    # Broadcast
    torch.distributed.broadcast_object_list(data_list,
                                            mpu.get_model_parallel_src_rank(),
                                            group=mpu.get_model_parallel_group())

    output = {k: data for k, data in zip(keys, data_list)}
    return output


def collate_fn_pad(batch, data_keys_to_collate=[], report_lengths=False):
    from torch.utils.data._utils.collate import default_collate

    elem = batch[0]
    elem_type = type(elem)

    try:
        out_dict = {}
        for key in elem:
            item_list = [d[key] for d in batch]
            # Custom behavior: pad this baby!
            if key in ['text', 'input_ids', 'attention_mask']:
                # Handle nested list[list[Tensor]]
                item_list = unwrap_lists(item_list)
                padded_item = torch.nn.utils.rnn.pad_sequence(item_list, batch_first=True)
                out_dict.update({key: padded_item})
                if report_lengths:
                    lengths = torch.IntTensor([sample.size(dim=0) for sample in item_list])
                    out_dict.update({'lengths': lengths})
            elif key in data_keys_to_collate:
                # Default collate behavior for a dictionary, according to pytorch 2.0.0
                out_dict.update({key: default_collate(item_list)})
            else:
                # Custom behavior for fields that are lists of lists
                out_dict.update({key: item_list})
        return elem_type(out_dict)
    except TypeError:
        raise ValueError(f"This mapping type {elem_type} may not support `__init__(iterable)`.")


class SeqLengthSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size, bucket_boundaries):
        super().__init__(data_source)
        self.ind_n_len = data_source['length'].numpy()
        self.bucket_boundaries = bucket_boundaries
        bin_inds = np.digitize(self.ind_n_len, self.bucket_boundaries)
        outside_bins = np.sum(bin_inds == 0)
        self.data_len = len(data_source) - outside_bins
        if outside_bins > 0:
            print(f'Excluding {outside_bins} samples because they fall outside length bins')
        uniq_bins = np.unique(bin_inds)
        data_buckets = {}
        for b in uniq_bins:
            if b == 0:
                continue
            data_buckets[b] = np.where(bin_inds == b)[0]
        self.data_buckets = data_buckets
        self.batch_size = batch_size

    def __iter__(self):
        iter_list = []
        for k in self.data_buckets.keys():
            np.random.shuffle(self.data_buckets[k])
            iter_list += (np.array_split(self.data_buckets[k],
                                         int(self.data_buckets[k].shape[0] / self.batch_size)))
        # shuffle(iter_list)  # shuffle batches so that they aren't ordered by bucket size
        for i in iter_list:
            yield i.tolist()  # as it was stored in an array

    def __len__(self):
        return self.data_len


def make_data_loader_with_padding(dataset, neox_args, seq_length_bins=None):
    """Build dataloader given an input dataset. Minor modification of megatron.data.data_utils"""
    from megatron.data.samplers import DistributedBatchSampler

    if dataset is None:
        return None
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = neox_args.batch_size * world_size
    num_workers = neox_args.num_workers

    collate_fn = partial(collate_fn_pad,
                         data_keys_to_collate=neox_args.data_keys_collate)

    if seq_length_bins:
        sampler = SeqLengthSampler(dataset, global_batch_size, seq_length_bins)
        batch_sampler = DistributedBatchSampler(sampler=sampler,
                                                batch_size=1,
                                                drop_last=True,
                                                rank=rank,
                                                world_size=world_size)
    else:
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


# def clean_text(text):
#     """Remove new lines and multiple spaces and adjust end of sentence dot."""
#
#     text = text.replace("\n", " ")
#     text = re.sub(r'\s+', ' ', text)
#     for _ in range(3):
#         text = text.replace(' . ', '. ')
#
#     return text
#
#
# def build_sample(ids, types, paddings, label, unique_id):
#     """Convert to numpy and return a sample consumed by the batch producer."""
#
#     ids_np = np.array(ids, dtype=np.int64)
#     types_np = np.array(types, dtype=np.int64)
#     paddings_np = np.array(paddings, dtype=np.int64)
#     sample = ({'text': ids_np,
#                'types': types_np,
#                'padding_mask': paddings_np,
#                'label': int(label),
#                'uid': int(unique_id)})
#
#     return sample
#
#
# def build_tokens_types_paddings_from_text(text_a, text_b,
#                                           tokenizer, max_seq_length):
#     """Build token types and paddings, trim if needed, and pad if needed."""
#
#     text_a_ids = tokenizer.tokenize(text_a)
#     text_b_ids = None
#     if text_b is not None:
#         text_b_ids = tokenizer.tokenize(text_b)
#
#     return build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids,
#                                                 max_seq_length, tokenizer.cls,
#                                                 tokenizer.sep, tokenizer.pad)
#
#
# def build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids, max_seq_length,
#                                          cls_id, sep_id, pad_id):
#     """Build token types and paddings, trim if needed, and pad if needed."""
#
#     ids = []
#     types = []
#     paddings = []
#
#     # [CLS].
#     ids.append(cls_id)
#     types.append(0)
#     paddings.append(1)
#
#     # A.
#     len_text_a = len(text_a_ids)
#     ids.extend(text_a_ids)
#     types.extend([0] * len_text_a)
#     paddings.extend([1] * len_text_a)
#
#     # [SEP].
#     ids.append(sep_id)
#     types.append(0)
#     paddings.append(1)
#
#     # B.
#     if text_b_ids is not None:
#         len_text_b = len(text_b_ids)
#         ids.extend(text_b_ids)
#         types.extend([1] * len_text_b)
#         paddings.extend([1] * len_text_b)
#
#     # Cap the size.
#     trimmed = False
#     if len(ids) >= max_seq_length:
#         max_seq_length_m1 = max_seq_length - 1
#         ids = ids[0:max_seq_length_m1]
#         types = types[0:max_seq_length_m1]
#         paddings = paddings[0:max_seq_length_m1]
#         trimmed = True
#
#     # [SEP].
#     if (text_b_ids is not None) or trimmed:
#         ids.append(sep_id)
#         if text_b_ids is None:
#             types.append(0)
#         else:
#             types.append(1)
#         paddings.append(1)
#
#     # Padding.
#     padding_length = max_seq_length - len(ids)
#     if padding_length > 0:
#         ids.extend([pad_id] * padding_length)
#         types.extend([pad_id] * padding_length)
#         paddings.extend([0] * padding_length)
#
#     return ids, types, paddings
