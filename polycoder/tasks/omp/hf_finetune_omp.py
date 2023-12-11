# v0.1 of code by Tal Kadosh, BGU/Technion
# Taken from talkad fork of vyaivo/code-lms


import os, sys
import torch
import logging
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, Trainer, TrainingArguments
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader, Dataset, BatchSampler, Sampler
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.cuda.amp import autocast

# My own script/function imports
import hf_data_omp as data_omp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import SeqLengthSampler

logger = logging.getLogger()


class DistributedSampler(Sampler):
    """ 
    VAV NB: this function taken from torchnlp
    Iterable wrapper that distributes data across multiple workers.

    Args:
        iterable (iterable)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.

    Example:
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
        [0, 2, 4, 6, 8]
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
        [1, 3, 5, 7, 9]
    """

    def __init__(self, iterable, num_replicas=None, rank=None):
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.rank = rank

        if num_replicas is None or rank is None:  # pragma: no cover
            if not torch.distributed.is_initialized():
                raise RuntimeError('Requires `torch.distributed` to be initialized.')

            self.num_replicas = (
                torch.distributed.get_world_size() if num_replicas is None else num_replicas)
            self.rank = torch.distributed.get_rank() if rank is None else rank

        if self.rank >= self.num_replicas:
            raise IndexError('`rank` must be smaller than the `num_replicas`.')

    def __iter__(self):
        return iter(
            [e for i, e in enumerate(self.iterable) if (i - self.rank) % self.num_replicas == 0])

    def __len__(self):
        return len(self.iterable)


class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.data[idx]['input_ids']),
                'labels': torch.tensor(self.data[idx]['labels'])}


def tokenize(args, tokenizer, sample):
    max_size = 2048
    if not args.is_replaced:
        encodings = tokenizer(sample['full'], max_length=max_size, add_special_tokens=True, truncation=True, padding=True)
#        if len(encodings['input_ids']) < max_size:
#            encodings['input_ids'].append(tokenizer.eos_token_id)
        encodings['length'] = len(encodings['input_ids'])
    else:
        encodings = {}
        encodings['input_ids'] = tokenizer(sample['full'], max_length=max_size, add_special_tokens=True, truncation=True, padding=True)
        encodings['labels'] = encodings['input_ids'][:]
    return encodings


def finetune(args):
    logger.info(f'start finetune {args.model_name}')

    if args.accel:
        from accelerate import Accelerator
        if args.device == 'cpu':
            accel = Accelerator()
            logger.info(accel.__dict__)
        else:
            raise NotImplementedError("Untested for combination of HF Accelerator + devices other than CPU")

    # TOKENIZER
    if args.is_replaced:
        from tokenizer import TokompilerTokenizer
        tokom_extended_tokens = ['parallel', 'private', 'reduction']
        tokenizer = TokompilerTokenizer(vocab_path=args.vocab_file)
        tokenizer.add_tokens(tokom_extended_tokens)
        tokenizer.enable_padding(length=2048)
    else:
        tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-2.7B", 
                                  truncation=True, model_input_names=['input_ids'])

        # tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, padding=True,
        #                         truncation=True, model_input_names=['input_ids'])
        tokenizer.pad_token = tokenizer.eos_token

    # DATA
    datasets = data_omp.build_omp_dataset(args)

    newd = []
    for i in range(len(datasets)):
        d = datasets[i]
        outd = d.map(lambda examples: tokenize(args, tokenizer, examples), remove_columns=['pragma', 'code', 'hash', 'full'])     
        newd.append(outd)
    traind, testd = newd
    
    if args.is_replaced:
        train_data = []
        for ids, labels in tqdm(zip(traind['input_ids'], traind['labels'])):
            train_data.append({'input_ids': ids, 'labels': labels})

        test_data = []
        for ids, labels in tqdm(zip(testd['input_ids'], testd['labels'])):
            test_data.append({'input_ids': ids, 'labels': labels})

        train_loader = DataLoader(dataset=CustomDataset(train_data), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=CustomDataset(test_data), batch_size=args.batch_size)
    else:
        world_size = accel.num_processes
        logger.info(f"Reading world_size as {world_size}...")
        sampler_batch_sz = args.batch_size
#        distr_batch_sz = sampler_batch_sz / world_size if world_size > 1 else 1

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        length_sampler = SeqLengthSampler(traind, bucket_boundaries=[2048, 1526, 1024, 0],
                                      batch_size=sampler_batch_sz)
        batch_sampler = DistributedBatchSampler(length_sampler, num_replicas=world_size, rank=accel.process_index)
        train_loader = DataLoader(dataset=traind, sampler=None, batch_sampler=batch_sampler, collate_fn=collator)
#        test_loader = DataLoader(dataset=testd, batch_size=args.batch_size, collate_fn=collator)

    # MODEL
    if args.load_model_ckpt:
        model = GPTNeoXForCausalLM.from_pretrained(args.load_model_ckpt)
    else:
        model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-2.7B")
    if args.accel:
        model = model.to(accel.device)
        model = accel.prepare(model)
        logger.info(model)
        logger.info("Using HuggingFace Accelerator")
    else:
        model.to(args.device)
    model.train()

    # update model embeddings
    if args.is_replaced:
        embedding_layer = model.get_input_embeddings()
        num_embeddings = embedding_layer.weight.shape[0]
        new_num_embeddings = num_embeddings+len(tokom_extended_tokens)
        model.resize_token_embeddings(new_num_embeddings)
        logger.info(f'Embedding layer has changed: {num_embeddings} -> {new_num_embeddings}')

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
                      weight_decay=args.weight_decay)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=args.warmup_steps,
                                                   num_training_steps=(len(train_loader) * args.num_epochs))
    # import pdb; pdb.set_trace()
    if args.accel:
        train_loader, optimizer, lr_scheduler = accel.prepare([train_loader, optimizer, lr_scheduler])
        if args.save_steps:
            accel.register_for_checkpointing(lr_scheduler)
        if args.load_accel_state:
            logger.info("Loading previous Accelerate state from {args.load_accel_state}...")
            accel.load_state(args.load_accel_state)

    # TRAIN
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader, miniters=1, desc=f"Epoch {epoch}")
        loss_total = 0.0 

        for step, batch in enumerate(pbar):  #train_loader):
            tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask', 'attention_mask']}

            if args.accel:
                batch.to(accel.device)
                outputs = model(**tensor_batch)
                loss = outputs.loss 
                accel.backward(loss)
            else:            
                with autocast():
                    outputs = model(**tensor_batch)
                    loss = outputs.loss 
                    loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_total += loss.detach().clone().item()

            if (step > 0) and (step % 10 == 0):
                logger.info(f'loss: {loss_total / (step+1)}')
                pbar.set_postfix({"avg_train_loss": loss_total / (step+1)})
            if args.accel and args.save_steps:
                if (step % args.save_steps == 0):
                    try:
                        logger.info('Saving HF accelerate training state...')
                        accel.save_state()
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()

        # VALIDATION       
        # val_loss = 0.0
        # for step_val, batch_val in enumerate(test_loader):
        #     tensor_batch = {k: v.to(args.device) for k, v in batch_val.items() if k in ['input_ids', 'labels', 'mask', 'attention_mask']}

        #     outputs = model(**tensor_batch)
        #     loss = outputs.loss 
        #     val_loss += loss.detach().clone().item()
        # logger.info(f'val loss:  {val_loss / (step_val+1)}')

        logger.info('Saving model checkpoint...')
        if args.accel:
            try:
                model.save_pretrained(os.path.join(args.save_dir, f'original_poly_bpe/epoch{epoch:02d}'), is_main_process=accel.is_main_process, save_function=accel.save)
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
        else:
            model.save_pretrained(os.path.join(args.save_dir, 'original_poly_bpe'), from_pt=True) 

#    print('Saving model checkpoint...')
#    model.save_pretrained(os.path.join(args.save_dir, 'original_poly_bpe'), from_pt=True) 

