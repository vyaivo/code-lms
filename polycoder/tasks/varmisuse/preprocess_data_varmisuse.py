# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import glob
from functools import partial, reduce
from itertools import chain

import numpy as np
import os
import sys

import datasets as trd
from datasets import Sequence, Value

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.pardir))
)
print(sys.path)

from megatron.tokenizer import build_tokenizer

import json


def tokenize(example, tokenizer):
    tmp = [tokenizer.tokenize(t) for t in example["edited_tokens"]]
    example["input_ids"] = list(chain(*tmp))
    ind_map = np.array([len(t) for t in tmp])
    example["token_indmap"] = np.cumsum(ind_map)
    # example["input_ids"] = tokenizer.tokenize(example["source_code"])
    example["length"] = len(example["input_ids"])
    assert example["token_indmap"][-1] == example["length"]
    return example


def find_slice(seq, subseq):
    n = len(seq)
    m = len(subseq)
    for i in range(n - m + 1):
        if np.all(seq[i:i + m] == subseq):
            yield i


def align_token_targets(example, tokenizer):
    if len(example["repair_targets"]) > 0:
        new_inds = [list(range(example["token_indmap"][r-1], example["token_indmap"][r])) for r in example["repair_targets"]]
        assert tokenizer.detokenize([example["input_ids"][x] for x in new_inds[0]]).strip() == example["source_tokens"][example["repair_targets"][0]], \
            "Index mapping has gone awry!"
        example["repair_targets"] = list(reduce(lambda x, y: x + y, new_inds))

    new_inds = [list(range(example["token_indmap"][r-1], example["token_indmap"][r])) for r in example["repair_candidates"]]
    assert tokenizer.detokenize([example["input_ids"][x] for x in new_inds[0]]).strip() == example["source_tokens"][example["repair_candidates"][0]], \
        "Index mapping has gone awry!"
    example["repair_candidates"] = list(reduce(lambda x, y: x + y, new_inds))

    return example


def remap_err_location(example, tokenizer):
    if example["error_location"] > 0:
        r = example["error_location"]
        new_inds = list(range(example["token_indmap"][r - 1], example["token_indmap"][r]))
        assert tokenizer.detokenize([example["input_ids"][x] for x in new_inds]).strip() == example["source_tokens"][
            example["error_location"]], "Index mapping has gone awry!"
        example["error_inds"] = new_inds
    else:
        example["error_inds"] = [0]
    return example


def build_varmisuse_datasets(args):
    # Build the dataset
    tokenizer = build_tokenizer(args)

    dataset_dir = os.path.dirname(args.input_data_path)
    splits = ['train', 'dev', 'eval']
    split_rename = ['train', 'validation', 'test']
    split_dict = {}
    for s, sr in zip(splits, split_rename):
        sfiles = glob.glob(f'{dataset_dir}/{s}/*.jsonl')
        assert isinstance(sfiles, list), f"Did not get a list of {s} files for {dataset_dir}/{s}/"
        assert len(sfiles) > 0, f"Did not get a list of {s} files for {dataset_dir}/{s}/"
        split_dict[sr] = sfiles
    d = trd.load_dataset('json', data_files=split_dict)

    tok_fn = partial(tokenize, tokenizer=tokenizer)
    tokenized_datasets = d.map(tok_fn, batched=False, remove_columns=['provenances'])

    align_fn = partial(align_token_targets, tokenizer=tokenizer)
    new_dataset = tokenized_datasets.map(align_fn, batched=False)

    remap_fn = partial(remap_err_location, tokenizer=tokenizer)
    new_dataset = new_dataset.map(remap_fn, batched=False, remove_columns=['error_location'])

    # feature_types = trd.Features({
    #     "input_ids": Sequence(Value("int32")),
    #     "error_inds": Sequence(Value("int32")),
    #     "repair_targets": Sequence(Value("int32")),
    #     "repair_candidates": Sequence(Value("int32")),
    #     "length": Value("int32"),
    #     "has_bug": Value("bool")
    # })
    for split, ds in new_dataset.items():
        ds.set_format(type="torch",
                      columns=['input_ids', 'error_inds', 'repair_targets',
                               'repair_candidates', 'has_bug', 'length'],
                      output_all_columns=True)
    new_dataset.save_to_disk(args.output_data_path)
    print(f"Saved datasets to {args.output_data_path}")

    # return new_dataset["train"], new_dataset["validation"], new_dataset["test"]


# class Encoder(object):
#     def __init__(self, args):
#         self.args = args
#
#     def initializer(self):
#         # Use Encoder class as a container for global data
#         Encoder.tokenizer = build_tokenizer(self.args)
#
#     def encode(self, json_raw_str):
#         jdict_list = json.loads(json_raw_str)
#         ids = {}
#         proc_size = 0
#         # Right now, assumes that it's just a plain text file without any other format!
#         for key in self.args.jsonl_keys:
#             doc_ids = []
#             for json_sample in jdict_list:
#                 text = json_sample[key]
#                 text_ids = Encoder.tokenizer.tokenize(text)
#                 if len(text_ids) > 0:
#                     doc_ids.append(text_ids)
#                 proc_size += len(text)
#             if self.args.append_eod:
#                 doc_ids[-1].append(Encoder.tokenizer.eod)
#             ids[key] = doc_ids
#         for key in self.args.jsonl_keys_no_transform:
#             doc_ids = [json_sample[key] for json_sample in jdict_list]
#             ids[key] = doc_ids
#         return ids, proc_size
#
#
def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input-data-path",
        type=str,
        required=True,
        help="Path to input jsonl files",
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "Tokompiler",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-data-path",
        type=str,
        required=True,
        help="Path to output file",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args
#
#
# def yield_from_files(dir, semaphore):
#     """
#     Modified for reading code corpus.
#
#     Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
#     other compressed formats. Also filters out empty documents.
#
#     :param fnames: list of filenames
#     """
#     fnames = []
#     for root, _, files in os.walk(dir):
#         for file in files:
#             fnames.append(os.path.join(root, file))
#     random.shuffle(fnames)
#
#     def read(fname):
#         with open(fname) as inp:
#             doc = inp.read()
#         return doc
#
#     def yielder(fname, semaphore):
#         f = read(fname)
#         if f:
#             semaphore.acquire()
#             yield f
#
#     for fname in fnames:
#         yield from yielder(fname, semaphore)
#
#
# def main():
#     args = get_args()
#     encoder = Encoder(args)
#     tokenizer = build_tokenizer(args)
#     print(f"Vocab size: {tokenizer.vocab_size}")
#     print(f"Output prefix: {args.output_prefix}")
#
#     # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
#     # hence building up memory
#     semaphore = Semaphore(10000 + args.workers)
#
#     # use multiprocessing to iterate over input documents
#     # modified to read code documents
#     fin = yield_from_files(args.input, semaphore)
#
#     if args.workers > 1:
#         pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
#         encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
#     else:
#         encoder.initializer()
#         encoded_docs = (encoder.encode(doc) for doc in fin)
#
#     # make a dataset builder for each key in args.jsonl_keys
#     # each key will output to a different file beginning with args.output_prefix
#     output_bin_files = {}
#     output_idx_files = {}
#     builders = {}
#     for key in args.jsonl_keys:   #modified for code corpus
#         output_bin_files[key] = "{}_{}_{}.bin".format(
#             args.output_prefix, key, "document"
#         )
#         output_idx_files[key] = "{}_{}_{}.idx".format(
#             args.output_prefix, key, "document"
#         )
#         builders[key] = indexed_dataset.make_builder(
#             output_bin_files[key],
#             impl=args.dataset_impl,
#             vocab_size=tokenizer.vocab_size,
#         )
#     for key in args.jsonl_keys_no_transform:   #modified for code corpus
#         output_bin_files[key] = "{}_{}_{}.bin".format(
#             args.output_prefix, key, "document"
#         )
#         output_idx_files[key] = "{}_{}_{}.idx".format(
#             args.output_prefix, key, "document"
#         )
#         builders[key] = indexed_dataset.make_builder(
#             output_bin_files[key],
#             impl=args.dataset_impl,
#             vocab_size=tokenizer.vocab_size,
#         )
#
#     # actually do tokenization
#     proc_start = time.time()
#     total_bytes_processed = 0
#     pbar = tqdm.tqdm()
#     for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
#         total_bytes_processed += bytes_processed
#
#         # release semaphore so `yield_from_files` can add another file to the buffer
#         semaphore.release()
#
#         # add each tokenized document / sentence
#         for key, sentences in doc.items():
#             if key in args.jsonl_keys:
#                 for sentence in sentences:
#                     builders[key].add_item(torch.IntTensor(sentence))
#             elif key in args.jsonl_keys_no_transform:
#                 if isinstance(sentences[0], list):
#                     for sentence in sentences:
#                         builders[key].add_item(torch.IntTensor(sentence))
#                 else:
#                     builders[key].add_item(torch.IntTensor(sentences))
#             # separate with eos token
#             builders[key].end_document()
#
#         # log progress
#         if i % args.log_interval == 0:
#             current = time.time()
#             elapsed = current - proc_start
#             mbs = total_bytes_processed / elapsed / 1024 / 1024
#             pbar.set_description(
#                 f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed} docs/s, {mbs} MB/s)."
#             )
#             if i != 0:
#                 pbar.update(args.log_interval)
#
#     # save output file
#     for key in args.jsonl_keys:
#         builders[key].finalize(output_idx_files[key])
#     for key in args.jsonl_keys_no_transform:
#         builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    args = get_args()
    build_varmisuse_datasets(args)
