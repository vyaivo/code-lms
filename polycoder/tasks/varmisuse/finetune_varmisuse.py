"""VarMisuse task -- finetuning/evaluation."""

import deepspeed

from functools import partial

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from megatron.neox_arguments import NeoXArgs
from megatron import mpu, print_rank_0
from megatron.training import get_optimizer, get_learning_rate_scheduler
from megatron.utils import get_total_params

# from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.model.classification import Classification
# from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune
# from megatron.arguments import core_transformer_config_from_args

import torch
import datasets as trd


# TODO: config file for this -- any new arguments needed?
# TODO: Run Docker test E2E


def varmisuse_multi_class(neox_args, num_classes):

    # """Build train and validation dataset."""
    # data_path = 'lm_data/hf/'  # '/export/data/vyvo/huggingface/'
    # tokenized_datasets = trd.load_from_disk(f"{data_path}varmisuse_hfdata")
    #
    # train_loader = tokenized_datasets['train']
    # valid_loader = tokenized_datasets['validation']

    def model_optim_lr_setup(neox_args, num_classes,
                             inference=False, get_key_value=None):
        """Build the model."""
        print_rank_0('building classification model for variable misuse task')
        # TODO: fix the model for predicting pointers to parts of the sequence
        model = Classification(neox_args, num_classes=num_classes)

        # print_rank_0(model)  # VAV DEBUG
        ## VAV DEBUG ##
        for name, p in model.named_parameters():
            if p.requires_grad:
                print_rank_0(name, p.shape)
        ## VAV DEBUG ##

        optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
        lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)

        if neox_args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")
            if neox_args.no_load_optim:
                assert optimizer is None
                _model_params = None
                _lr_scheduler = None
            else:
                _model_params = param_groups if optimizer is None else None
                _lr_scheduler = lr_scheduler

            model, optimizer, _, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                args=neox_args,
                lr_scheduler=_lr_scheduler,
                dist_init_required=False,
                model_parameters=_model_params,
                config_params=neox_args.deepspeed_config,
                mpu=mpu if not neox_args.is_pipe_parallel else None,
            )
            model.total_params = get_total_params(model.module)
            print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')
            #        raise ValueError("Stopping here so we can check!")  ## VAV DEBUG
            if neox_args.is_pipe_parallel:
                model.set_has_attention_mask(True)
                model.set_batch_fn(model.module._megatron_batch_fn)
        else:
            raise ValueError("Must be using deepspeed to run neox")

        if neox_args.load is not None:
            neox_args.iteration = load_checkpoint(
                neox_args=neox_args,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                inference=False,
                iteration=None,
            )
            print_rank_0(
                f"Loading checkpoint and starting from iteration {neox_args.iteration}"
            )
        else:
            neox_args.iteration = 0
        return model, optimizer, lr_scheduler

    def metrics_func_provider():
        """Provide metrics callback function."""
        pass
        # TODO: add in accuracy metrics
        # def single_dataset_provider(datapath):
        #     args = get_args()
        #     tokenizer = get_tokenizer()
        #
        #     name = name_from_datapath_func(datapath)
        #     return Dataset(name, [datapath], tokenizer, args.seq_length)
        # return accuracy_func_provider(single_dataset_provider)

    from data_varmisuse import build_train_valid_test_data_iterators

    """Finetune/evaluate."""
    finetune(neox_args, partial(model_optim_lr_setup, num_classes=num_classes),
             build_train_valid_test_data_iterators)
    # finetune(neox_args, train_valid_datasets_provider, model_provider,
    #          end_of_epoch_callback_provider=metrics_func_provider,
    #          task_collate_fn=collate_fn_pad)


if __name__ == "__main__":

    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined

    varmisuse_multi_class(neox_args, num_classes=2)
