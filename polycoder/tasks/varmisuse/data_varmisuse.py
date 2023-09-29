import glob
import numpy as np
import os
import torch

from megatron import mpu, print_rank_0
from megatron.utils import get_ltor_masks_and_position_ids
    #, average_losses_across_data_parallel_group
# from megatron.data.gpt2_dataset import _build_index_mappings

from tasks.data_utils import make_data_loader_with_padding, broadcast_data_list
import datasets as trd

# DATASET KEYS:
# 'has_bug'
# 'error_location'
# 'repair_targets'
# 'repair_candidates'
# 'provenances' - drop
# 'source_code'


def build_varmisuse_datasets(neox_args):
    # if os.path.exists(os.path.join(neox_args.finetune_data_path, "test/data.arrow")):
    #     tokenized_datasets = trd.load_from_disk(neox_args.finetune_data_path)
    #     # tokenized_datasets = tokenized_datasets.rename_column("input_ids", "text")
    # else:
    # Build the dataset
    from datasets import Sequence, Value
    from megatron.tokenizer import build_tokenizer
    tokenizer = build_tokenizer(neox_args)

    dataset_dir = os.path.dirname(os.path.dirname(neox_args.finetune_data_path))
    splits = ['train', 'dev', 'eval']
    split_rename = ['train', 'validation', 'test']
    split_dict = {}
    for s, sr in zip(splits, split_rename):
        sfiles = glob.glob(f'{dataset_dir}/{s}/*.jsonl')
        assert isinstance(sfiles, list), f"Did not get a list of {s} files for {dataset_dir}/{s}/"
        assert len(sfiles) > 0, f"Did not get a list of {s} files for {dataset_dir}/{s}/"
        split_dict[sr] = sfiles
    d = trd.load_dataset('json', data_files=split_dict)

    # TODO: map the repair_targets and repair_candidates onto the LM TOKENS vs the original tokens :X
    def tokenize(example):
        example["input_ids"] = tokenizer.tokenize(example["source_code"])
        example["length"] = len(example["input_ids"])
        return example

    tokenized_datasets = d.map(tokenize, batched=False)
    feature_types = trd.Features({
        "input_ids": Sequence(Value("int32")),
        "error_location": Sequence(Value("int32")),
        "repair_targets": Sequence(Value("int32")),
        "repair_candidates": Sequence(Value("int32")),
        "length": Sequence(Value("int32")),
        "has_bug": Sequence(Value("bool"))
    })
    for split, ds in tokenized_datasets.items():
        ds.set_format(type="torch",
                      columns=['input_ids', 'error_location', 'repair_targets',
                               'repair_candidates', 'has_bug', 'length'])
    # tokenized_datasets.save_to_disk(neox_args.finetune_data_path)

    return tokenized_datasets["train"], tokenized_datasets["validation"], tokenized_datasets["test"]


def build_train_valid_test_data_iterators(neox_args):
    """
    Edited from megatron.training to simply use VarMisuse dataset instead
    """

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Ensure only the first/last pipeline stages have data loaders
    if neox_args.is_pipe_parallel:
        is_first_stage = mpu.get_pipe_parallel_rank() == 0
        is_last_stage = mpu.get_pipe_parallel_rank() == mpu.get_pipe_parallel_world_size() - 1
        pipe_load = is_first_stage or is_last_stage
    else:
        pipe_load = True

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0 and pipe_load:
        # Number of train/valid/test samples.
        train_iters = neox_args.train_iters
        eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
        test_iters = neox_args.eval_iters
        train_val_test_num_samples = [train_iters * neox_args.train_batch_size,
                                      eval_iters * neox_args.train_batch_size,
                                      test_iters * neox_args.train_batch_size]

        if neox_args.train_data_paths:
            raise NotImplementedError
        else:
            # when just data_path is provided
            # split dataset into train, valid and test from data_path
            train_ds, valid_ds, test_ds = build_varmisuse_datasets(neox_args)

        length_bins = [2048, 2000, 512, 128, 64, 0]
        # Build dataloaders.
        train_dataloader = make_data_loader_with_padding(train_ds, neox_args, length_bins)
        valid_dataloader = make_data_loader_with_padding(valid_ds, neox_args, length_bins)
        test_dataloader = make_data_loader_with_padding(test_ds, neox_args, length_bins)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and neox_args.train_iters > 0
        do_valid = valid_dataloader is not None and neox_args.eval_iters > 0
        do_test = test_dataloader is not None and neox_args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    if neox_args.is_pipe_parallel:
        # Only first/last pipeline stages have data loaders, so pipeline parallelism should
        # broadcast globally instead of just the model parallel group.
        torch.distributed.broadcast(flags, src=0)
    else:
        torch.distributed.broadcast(flags,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
    neox_args.do_train = flags[0].item()
    neox_args.do_valid = flags[1].item()
    neox_args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = (neox_args.iteration * neox_args.gradient_accumulation_steps) % \
                                                    len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = ((neox_args.iteration * neox_args.gradient_accumulation_steps) // neox_args.eval_interval) * \
                         neox_args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
                                                    len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def _get_batch(neox_args, tokenizer, keys, data):
    """Support function for get_batch / get_batch pipe (to avoid code repetition)"""
    data_b = broadcast_data_list(keys, data)

    # Unpack.
    tokens = data_b["input_ids"].long().contiguous()

    nbatch, seq_length = tokens.shape
    # Create masks for loss function. Mask values of 1 are kept. Default 0.

    # Create mask for the repair candidates
    mask = torch.zeros((nbatch, seq_length), dtype=torch.long)
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
            mask[b, rcand[b]] = 1.
    except Exception as e:
        print(e)
        print(f'maximum index value is {rcand[b].max()}')
        import pdb; pdb.set_trace()

    # Create mask for the repair targets
    if data_b["error_location"].sum() == 0:
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
    labels = (data_b["error_location"], target_mask, mask, data_b["has_bug"])
    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, tokenizer.eod, neox_args.eod_mask_loss
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch(neox_args, data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ['input_ids', 'error_location', 'repair_targets', 'repair_candidates', 'has_bug']

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    return _get_batch(
        neox_args=neox_args,
        tokenizer=neox_args.tokenizer,
        keys=keys,
        data=data
    )


def get_batch_pipe(data, neox_args):
    """A modification of get_batch() to work with the latest batch instead of an iterator. """
    # Items and their type.
    keys = ['input_ids', 'error_location', 'repair_targets', 'repair_candidates', 'has_bug']

    tokens, labels, _, attention_mask, position_ids = _get_batch(
        neox_args, neox_args.tokenizer, keys, data
    )
    # VAV: need to return data in this order to be compatible with deepspeed pipelining code
    # unpack data
    return (tokens, position_ids, attention_mask), labels
    # return tokens, labels, loss_mask, attention_mask, position_ids


############# DO NOT USE ##############
# class VarMisuseDataset(torch.utils.data.Dataset):
#
#     def __init__(self, name, data_path, all_datasets, all_documents,
#                  num_samples, seq_length, seed):
#         """
#
#         :param name: the name of the data split.
#         :param data_path: the parent path containing all the *.idx and *.bin files for each dataset.
#         :param all_datasets: a dictionary for each dataset key. Each value is an IndexedDataset.
#         :param all_documents: a dictionary for each dataset key. Each value is an array containing the
#                               document indices for that data split and dataset.
#                               NOTE that the number of documents is actually dependent on the size of the dataset.
#         :param num_samples: number of samples to use from that dataset split. this is dependent on the
#                             batch size, train_iters/eval_iters
#         :param seq_length: the sequence length for the text data.
#         :param seed: the random seed which generates the indices to shuffle the dataset.
#         """
#         self.name = name
#         self.datasets = all_datasets
#         # data_keys = list(self.datasets.keys())
#         self.data_keys = list(self.datasets.keys())
#         # self.indexed_dataset = indexed_dataset
#
#         # Checks
#         # self.text_key = data_key
#         self.doc_idx, self.sample_idx, self.shuffle_idx = {}, {}, {}
#         # data_prefix = data_path + data_key + "_document"
#         # print(f"VAV DEBUG: building {data_prefix}")
#         # # Build index mappings.
#         # self.doc_idx[data_key], self.sample_idx[data_key], self.shuffle_idx[data_key] = _build_index_mappings(
#         #     self.name, data_prefix, documents, self.indexed_dataset.sizes,
#         #     num_samples, seq_length, seed)
#         # self.shuffle_idx_len = self.shuffle_idx.shape[0] - 1
#         # self.sample_idx_len = self.sample_idx.shape[0] - 1
#         # if self.shuffle_idx_len != self.sample_idx_len:
#         #     print(f'WARNING: shuffle index length ({self.shuffle_idx_len}) is not equal to sample index length ({self.sample_idx_len})')
#
#         # Sanity checks for the other dataset keys
#         for key in self.data_keys:  #list(self.datasets.keys()):
#             documents = all_documents[key]
#             indexed_dataset = self.datasets[key]
#             assert np.min(documents) >= 0
#             assert np.max(documents) < indexed_dataset.sizes.shape[0]
#             data_prefix = data_path + key + "_document"
#             if key == 'text':
#                 self.doc_idx[key], self.sample_idx[key], self.shuffle_idx[key] = _build_index_mappings(
#                     self.name, data_prefix, documents, indexed_dataset.sizes,
#                     num_samples, seq_length, seed)
#                 self.shuffle_idx_len = self.shuffle_idx[key].shape[0] - 1
#                 self.sample_idx_len = self.sample_idx[key].shape[0] - 1
#             else:
#                 if key in ['has_bug', 'error_location']:
#                     # Do something better here
#                     doc_idx, sample_idx, shuffle_idx = _build_nontext_index_mappings(
#                         self.name, data_prefix, documents, indexed_dataset.sizes, num_samples, 1, seed)
#             # assert np.all(np.array(shuffle_idx) == np.array(self.shuffle_idx)), "shuffle indices are not the same!"
#             # assert np.all(np.array(sample_idx) == np.array(self.sample_idx)), "sample indices are not the same!"
#             # assert np.all(np.array(doc_idx) == np.array(self.doc_idx)), "sample indices are not the same!"
#             # raise ValueError
#         print("VAV DEBUG shuf_idx:", self.shuffle_idx)
#         print("VAV DEBUG doc_idx:", self.doc_idx)
#         # for train, doc_idx is either 108 -- for error_location, has_bug
#         # or it's 1802538 for everyone else.
#         # for indexed_dataset.sizes, it's number of docs for error_location, has_bug
#         raise ValueError("VAV DEBUG: Stopping here to examine the indices")
#
#     def __len__(self):
#         return min(self.shuffle_idx_len, self.sample_idx_len)
#
#     def __getitem__(self, idx):
#         for key in self.data_keys:
#             try:
#                 # Get the shuffled index.
#                 idx = self.shuffle_idx[key][idx]
#                 # Start and end documents and offsets.
#                 doc_index_f = self.sample_idx[key][idx][0]
#                 doc_index_l = self.sample_idx[key][idx + 1][0]
#                 offset_f = self.sample_idx[key][idx][1]
#                 offset_l = self.sample_idx[key][idx + 1][1]
#                 # If we are within the same document, just extract the chunk.
#                 output_sample = {}
#                 if doc_index_f == doc_index_l:
#                     sample = self.datasets[key].get(self.doc_idx[key][doc_index_f], offset=offset_f,
#                                                     length=offset_l - offset_f + 1)
#                     output_sample[key] = np.array(sample, dtype=np.int64)
#                 else:
#                     # Otherwise, get the rest of the initial document.
#                     sample_list = [self.datasets[key].get(self.doc_idx[key][doc_index_f], offset=offset_f)]
#                     # Loop over all in between documents and add the entire document.
#                     for i in range(doc_index_f + 1, doc_index_l):
#                         sample_list.append(self.datasets[key].get(self.doc_idx[key][i]))
#                     # And finally add the relevant portion of last document.
#                     sample_list.append(self.datasets[key].get(self.doc_idx[key][doc_index_l], length=offset_l + 1))
#                     sample = np.concatenate(sample_list)
#                     output_sample[key] = np.array(sample, dtype=np.int64)
#             except IndexError:
#                 shuffle_idx_len = self.shuffle_idx[key].shape[0] - 1
#                 sample_idx_len = self.sample_idx[key].shape[0] - 1
#                 klen = min(shuffle_idx_len, sample_idx_len)
#                 new_idx = idx % klen
#                 print(
#                     f'WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})')
#                 return self[new_idx]
#         print("VAV DEBUG sample:", output_sample)
#         raise ValueError
#         return output_sample
#
#
# def _build_nontext_index_mappings(name, data_prefix, documents, sizes, num_samples, seq_length, seed):
#     import os, time
#     """Build doc-idx, sample-idx, and shuffle-idx.
#     doc-idx: is an array (ordered) of documents to be used in training.
#     sample-idx: is the start document index and document offset for each
#        training sample.
#     shuffle-idx: maps the sample index into a random index into sample-idx.
#     """
#     from megatron.data.gpt2_dataset import _num_tokens, _num_epochs, _build_doc_idx, _build_sample_idx
#     # Number of tokens in each epoch and number of required epochs.
#     tokens_per_epoch = _num_tokens(documents, sizes)
#     num_epochs = _num_epochs(tokens_per_epoch, 1, num_samples)
#     # rng state
#     np_rng = np.random.RandomState(seed=seed)
#
#     # Filename of the index mappings.
#     _filename = data_prefix
#     _filename += '_{}_indexmap'.format(name)
#     _filename += '_{}ns'.format(num_samples)
#     _filename += '_{}sl'.format(seq_length)
#     _filename += '_{}s'.format(seed)
#     doc_idx_filename = _filename + '_doc_idx.npy'
#     sample_idx_filename = _filename + '_sample_idx.npy'
#     shuffle_idx_filename = _filename + '_shuffle_idx.npy'
#
#     import pdb
#     pdb.set_trace()
#
#     # Build the indexed mapping if not exist.
#     if torch.distributed.get_rank() == 0:
#         if (not os.path.isfile(doc_idx_filename)) or \
#                 (not os.path.isfile(sample_idx_filename)) or \
#                 (not os.path.isfile(shuffle_idx_filename)):
#             print_rank_0(' > WARNING: could not find index map files, building '
#                          'the indices on rank 0 ...')
#             # doc-idx.
#             start_time = time.time()
#             doc_idx = _build_doc_idx(documents, num_epochs, np_rng)
#             np.save(doc_idx_filename, doc_idx, allow_pickle=True)
#             print_rank_0(' > elasped time to build and save doc-idx mapping '
#                          '(seconds): {:4f}'.format(time.time() - start_time))
#             # sample-idx.
#             start_time = time.time()
#             # Use C++ implementation for speed.
#             from megatron.data import helpers
#             assert doc_idx.dtype == np.int32
#             assert sizes.dtype == np.int32
#             # sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
#             #                                       num_epochs, tokens_per_epoch)
#             # oldTODO: fix _build_sample_idx. Currently it is doing the chunking into seq_length pieces,
#             #   but if we have variable length vectors this won't work...
#             sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
#                                           num_epochs, tokens_per_epoch)
#             np.save(sample_idx_filename, sample_idx, allow_pickle=True)
#             print_rank_0(' > elapsed time to build and save sample-idx mapping '
#                          '(seconds): {:4f}'.format(time.time() - start_time))
#             # shuffle-idx.
#             start_time = time.time()
#             # -1 is due to data structure used to retieve the index:
#             #    sample i --> [sample_idx[i], sample_idx[i+1])
#             shuffle_idx = _build_shuffle_idx(sample_idx.shape[0] - 1, np_rng)
#             np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
#             print_rank_0(' > elapsed time to build and save shuffle-idx mapping'
#                          ' (seconds): {:4f}'.format(time.time() - start_time))
#
#     # This should be a barrier but nccl barrier assumes
#     # device_index=rank which is not the case for model
#     # parallel case
#     counts = torch.cuda.LongTensor([1])
#     torch.distributed.all_reduce(counts, group=mpu.get_io_parallel_group())
#     assert counts[0].item() == torch.distributed.get_world_size(
#         group=mpu.get_io_parallel_group())
#
#     # Load mappings.
#     start_time = time.time()
#     print_rank_0(' > loading doc-idx mapping from {}'.format(
#         doc_idx_filename))
#     doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
#     print_rank_0(' > loading sample-idx mapping from {}'.format(
#         sample_idx_filename))
#     sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
#     print_rank_0(' > loading shuffle-idx mapping from {}'.format(
#         shuffle_idx_filename))
#     shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
#     print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
#         time.time() - start_time))
#     print_rank_0('    total number of samples: {}'.format(
#         sample_idx.shape[0]))
#     print_rank_0('    total number of epochs: {}'.format(num_epochs))
#
#     return doc_idx, sample_idx, shuffle_idx
#
# tokenized_datasets = trd.load_from_disk("varmisuse_hfdata")
#
#
# def build_train_valid_test_datasets(data_path, data_key, other_data_keys,
#                                     data_impl, splits_string,
#                                     train_valid_test_num_samples,
#                                     seq_length, seed, skip_warmup):
#     """Build train, valid, and test datasets."""
#
#     from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
#     from megatron.data.data_utils import get_train_valid_test_split_
#
#     dataset_dict, split_dict, document_dict = {}, {}, {}
#     # Indexed dataset.
#     data_prefix = data_path + data_key + "_document"
#     print(f"VAV DEBUG in build_train_valid_test_datasets:", data_prefix)
#     indexed_dataset = make_indexed_dataset(data_prefix,
#                                            data_impl,
#                                            skip_warmup)
#     dataset_dict['text'] = indexed_dataset
#     total_num_of_documents = indexed_dataset.sizes.shape[0]
#     splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
#     assert all(x < y for x, y in zip(splits, splits[1:])), \
#         f"Splits should be strictly increasing, but are not: {splits}"
#     split_dict['text'] = splits
#
#     # Other dataset samples
#     for key in other_data_keys:
#         data_prefix = data_path + key + "_document"
#         dataset_dict[key] = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
#         ndoc = dataset_dict[key].sizes.shape[0]
#         splits = get_train_valid_test_split_(splits_string, ndoc)
#         split_dict[key] = splits
#         assert all(x < y for x, y in zip(splits, splits[1:])), \
#             f"Splits should be strictly increasing, but are not: {splits}"
#
#     # Print stats about the splits.
#     print_rank_0(' > dataset split:')
#
#     def print_split_stats(name, index):
#         print_rank_0('    {}:'.format(name))
#         print_rank_0('     document indices in [{}, {}) total of {} '
#                      'documents'.format(splits[index], splits[index + 1],
#                                         splits[index + 1] - splits[index]))
#
#     print_split_stats('train', 0)
#     print_split_stats('validation', 1)
#     print_split_stats('test', 2)
#
#     def build_dataset(index, name):
#             # documents = np.arange(start=splits[index], stop=splits[index + 1],
#             #                       step=1, dtype=np.int32)
#         document_dict = {k: np.arange(start=s[index], stop=s[index + 1], step=1, dtype=np.int32)
#                          for k, s in split_dict.items()}
#         # data_path, all_datasets, all_documents,
#         dataset = VarMisuseDataset(name, data_path, dataset_dict, document_dict,
#                                    # ['error_location', 'repair_targets', 'repair_candidates', 'has_bug'],
#                                    # documents, indexed_dataset,
#                                    train_valid_test_num_samples[index],
#                                    seq_length, seed)
#         return dataset
#
#     train_dataset = build_dataset(0, 'train')
#     valid_dataset = build_dataset(1, 'valid')
#     test_dataset = build_dataset(2, 'test')
#
#     return train_dataset, valid_dataset, test_dataset