import glob
import os, sys

import datasets as trd
from datasets import Sequence, Value
import numpy
import torch


def build_lore_dataset(dummy_neox_arg, dummy_iter_arg, data_path=None, tokenizer=None):
    if os.path.exists(os.path.join(data_path, "test/data.arrow")):
        tokenized_datasets = trd.load_from_disk(data_path)
    else:
        # Build the dataset
#        from megatron.tokenizer import build_tokenizer
#        tokenizer = build_tokenizer(args)

        # JSON fields are:
        #   source: directory name to find the codelet
        #   code: text of the source code
        
        feature_types = trd.Features({
            "source": Value("string"),
            "code": Value("string"),
        })

        dataset_dir = data_path
        split_names = ["train", "val", "test"]
        split_dict = {}
        for s in split_names:
            split_dict.update({s: f'{dataset_dir}/{s}.jsonl'})
        d = trd.load_dataset('json', data_files=split_dict, features=feature_types)

        def tokenize(example):
            example["input_ids"] = tokenizer.tokenize(example["code"])
            example["length"] = len(example["input_ids"])
            return example

        tokenized_dataset = d.map(tokenize, batched=False)

        tokenized_dataset.set_format(type="torch",
                                     columns=['input_ids'],
                                     output_all_columns=True)

        tokenized_dataset.save_to_disk(data_path)

        return tokenized_dataset["train"], tokenized_dataset["val"], tokenized_dataset["test"]


def lore_batch_fn(neox_args, tokenizer, keys, data, datatype=torch.int64):
    """Polycoder/Megatron-LM support function for get_batch / get_batch pipe (to avoid code repetition). See tasks/data_utils.py."""
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

    nbatch, seq_length = tokens.shape
    # Create masks for loss function. Mask values of 1 are kept. Default 0.

    # Placeholder for MPI labels once we figure out how to score them
    # labels = (lm_labels, torch.empty((0, 1)))  #data_b["mpi_labels"])
    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, 0, True
    )

    return tokens, (labels, loss_mask), attention_mask, position_ids

