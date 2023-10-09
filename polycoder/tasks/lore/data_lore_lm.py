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
        # tokenized_dataset.save_to_disk(data_path)

    return tokenized_dataset["train"], tokenized_dataset["val"], tokenized_dataset["test"]

