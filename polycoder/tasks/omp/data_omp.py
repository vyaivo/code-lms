import glob
import os, sys

from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.data_utils import broadcast_data_list

import datasets as trd
from datasets import Sequence, Value
import numpy
import torch


def build_omp_dataset(neox_args, unused_nsample_list=None, rebuild=False):
    if (not rebuild) and os.path.exists(os.path.join(neox_args.finetune_data_path, "test/data.arrow")):
        tokenized_datasets = trd.load_from_disk(neox_args.finetune_data_path)
    else:
        # Build the dataset
        from megatron.tokenizer import build_tokenizer
        tokenizer = build_tokenizer(neox_args)
        if neox_args.tokenizer_type.lower() == 'GPT2BPETokenizer'.lower():
            eos_token = tokenizer.eod_id
        elif neox_args.tokenizer_type.lower() == 'Tokompiler'.lower():
            eos_token = tokenizer.eod
        else:
            raise NotImplementedError(f"We do not support the tokenizer type {neox_args.tokenizer_type}")

        feature_types = trd.Features({
            "code": Value("string"),
            "pragma": Value("string"),
            "hash": Value("string"),
        })

        dataset_dir = neox_args.finetune_data_path
        dpath = glob.glob(f'{dataset_dir}/*.jsonl')
        d = trd.load_dataset('json', data_files=dpath, features=feature_types,
                             split=['train[0%:80%]', 'train[80%:90%]', 'train[90%:100%]'])
        d = trd.DatasetDict({'train': d[0], 'validation': d[1], 'test': d[2]})

        def tokenize_and_parse(example, eos_token=eos_token):
            tmp = f'{example["code"]}\n{example["pragma"]}'
            example["input_ids"] = tokenizer.tokenize(tmp) + [eos_token]  #[tokenizer.eod_id]
            example["length"] = len(example["input_ids"])
            return example

        # JSON fields are:
        #   hash: an alphanumeric identifier
        #   code: text of the source code
        #   pragma: the pragma to predict given the input code

        tokenized_dataset = d.map(tokenize_and_parse, batched=False)

        tokenized_dataset.set_format(type="torch",
                                     columns=['input_ids'],
                                     output_all_columns=True)

        tokenized_dataset.save_to_disk(neox_args.finetune_data_path)

        return tokenized_dataset["train"], tokenized_dataset["validation"], tokenized_dataset["test"]


def omp_batch_fn(neox_args, tokenizer, keys, data, datatype=torch.int64):
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

    nbatch, seq_length = tokens.shape
    # Create masks for loss function. Mask values of 1 are kept. Default 0.

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, 0, True
    )

    return tokens, (labels, loss_mask), attention_mask, position_ids

