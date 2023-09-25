"""VarMisuse task -- finetuning/evaluation."""

import deepspeed

from functools import partial
from pathlib import Path
from pprint import pformat

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from megatron.neox_arguments import NeoXArgs
from megatron import mpu, print_rank_0
from megatron.training import get_optimizer, get_learning_rate_scheduler
from megatron.utils import get_total_params

from megatron.checkpointing import load_checkpoint
# from megatron import get_tokenizer
# from megatron.checkpointing import check_checkpoint_args, check_forward_pass
from megatron.model.classification import Classification
# from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune
# from megatron.arguments import core_transformer_config_from_args

import torch


# from megatron.utils import average_losses_across_data_parallel_group
from sklearn.metrics import accuracy_score
import datasets as trd


# TODO: Run Docker test E2E


# TODO: figure out why we get an OOM error?!?! seems like it's during validation?


def varmisuse_multi_class(neox_args, num_classes):

    # """Build train and validation dataset."""
    from data_varmisuse import build_train_valid_test_data_iterators

    def model_optim_lr_setup(neox_args, num_classes,
                             inference=False, get_key_value=None):
        """Build the model."""
        print_rank_0('building classification model for variable misuse task')
        # TODO: fix the model for predicting pointers to parts of the sequence
        model = Classification(neox_args, num_classes=num_classes,
                               post_process=False, load_lm_checkpoint=neox_args.load)

        optimizer, param_groups = get_optimizer(model=model, neox_args=neox_args)
        lr_scheduler = get_learning_rate_scheduler(optimizer=optimizer, neox_args=neox_args)

        # print_rank_0(model)  # VAV DEBUG
        ## VAV DEBUG ##
        for name, p in model.named_parameters():
            if p.requires_grad:
                print_rank_0(name, p.shape)
        ## VAV DEBUG ##

        model.total_params = get_total_params(model)
        print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')

        # if neox_args.deepspeed:
        #     print_rank_0("DeepSpeed is enabled.")
        #     if neox_args.no_load_optim:
        #         assert optimizer is None
        #         _model_params = None
        #         _lr_scheduler = None
        #     else:
        #         _model_params = param_groups if optimizer is None else None
        #         _lr_scheduler = lr_scheduler
        #
        #     model, optimizer, _, lr_scheduler = deepspeed.initialize(
        #         model=model,
        #         optimizer=optimizer,
        #         args=neox_args,
        #         lr_scheduler=_lr_scheduler,
        #         dist_init_required=False,
        #         model_parameters=_model_params,
        #         config_params=neox_args.deepspeed_config,
        #         mpu=mpu if not neox_args.is_pipe_parallel else None,
        #     )
        #     model.total_params = get_total_params(model.module)
        #     print_rank_0(f' > total params: {"{:,}".format(model.total_params)}')
        #     #        raise ValueError("Stopping here so we can check!")  ## VAV DEBUG
        #     if neox_args.is_pipe_parallel:
        #         import pdb; pdb.set_trace()
        #         model.set_has_attention_mask(True)
        #         model.set_batch_fn(model.module._megatron_batch_fn)
        # else:
        #     raise ValueError("Must be using deepspeed to run neox")

        # if neox_args.load is not None:
        #     neox_args.iteration = load_checkpoint(neox_args, model,
        #                                           optimizer=optimizer,
        #                                           lr_scheduler=lr_scheduler,
        #                                           inference=False, iteration=None)
        #     print_rank_0(
        #         f"Loading checkpoint and starting from iteration {neox_args.iteration}"
        #     )
        # else:
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

    """Finetune/evaluate."""
    finetune(neox_args, partial(model_optim_lr_setup, num_classes=num_classes),
             build_train_valid_test_data_iterators,
             compute_varmisuse_loss)


def compute_varmisuse_loss(label_data, logits, metric=False):
    # Unpack label data
    # (error_locations, repair_loc), (lengths, has_bug, repair_cands) = label_data
    # print("VAV DEBUG dp & mp rank:", mpu.get_data_parallel_rank(), mpu.get_model_parallel_rank())
    # print("VAV DEBUG devices:", error_locations.device, target_mask.device, mask.device, has_bug.device, logits.device)

    nbatch, seq_length, _ = logits.shape

    label_data = [l.to(logits.device) for l in label_data]
    error_locations, target_mask, mask, has_bug = label_data
    n_buggy = torch.sum(has_bug).item()
    eps = 1e-16

    # Localization loss is simply calculated with sparse CE
    loc_pred = logits[:, :, 0]  # * mask  # (nbatch, seq_length)
    loc_loss = torch.nn.functional.cross_entropy(loc_pred, error_locations)
    if metric:
        loc_max = loc_pred.detach().clone().softmax(dim=-1).argmax(dim=-1)
        loc_acc_nobug = accuracy_score(y_pred=loc_max[~has_bug],
                                       y_true=error_locations[~has_bug])
        loc_acc_bug = accuracy_score(y_pred=loc_max[has_bug],
                                     y_true=error_locations[has_bug])
    else:
        loc_acc_nobug, loc_acc_bug, mean_repair_acc, joint_acc = None, None, None, None

    # Repair loss is only computed at buggy samples (with error locations),
    # using (negative) cross-entropy
    repair_pred = logits[:, :, 1] * mask  # (nbatch, seq_length)
    # repair_locations becomes a (nbatch, seq_length) Tensor of mostly 0s
    #  it has values of 1 if it is a repair location. each row can have >1 label
    ### trying to match hellendoorn metric
    pointer_logits = repair_pred + (1.0 - mask) * eps
    pointer_probs = torch.nn.functional.softmax(pointer_logits, dim=-1)
    target_probs = (target_mask * pointer_probs).sum(-1)
    repair_loss = -torch.log(target_probs[has_bug] + eps).sum() / (eps + n_buggy)
    # Since this is a multi-label problem, we need a threshold for marking a location.
    # For simplicity and to follow the previous work, we use a probability thresh of 0.5.
    if metric:
        ### trying to match hellendoorn metric
        target_loc_acc = (target_probs[has_bug].detach().clone() > 0.5).type(torch.FloatTensor)
        # Also compute the joint accuracy, but on buggy samples only
        lacc_by_sample = (loc_max[has_bug] == error_locations[has_bug]).type(torch.FloatTensor)
        joint_acc = (lacc_by_sample * target_loc_acc).mean()
    loss = loc_loss + repair_loss

    # Reduce loss for logging.
    # averaged_loss = average_losses_across_data_parallel_group([loss])

    # return loss, {'lm loss': averaged_loss[0]}
    return loss, (loc_acc_nobug, loc_acc_bug, mean_repair_acc, joint_acc)


# def compute_varmisuse_loss_create_mask(label_data, logits, metric=False):
#     # Unpack label data
#     (error_locations, repair_loc), (lengths, has_bug, repair_cands) = label_data
#
#     nbatch, seq_length, _ = logits.shape
#     n_buggy = torch.sum(has_bug).item()
#     eps = 1e-16
#     # Create sequence length mask. Mask values of 1 are kept. Default 0.
#     mask = torch.zeros((nbatch, seq_length), dtype=torch.long, device=logits.device)
#     for b in range(nbatch):
#         mask[b, repair_cands[b]] = 1.
#
#     # Localization loss is simply calculated with sparse CE
#     loc_pred = logits[:, :, 0]  # * mask  # (nbatch, seq_length)
#     loc_loss = torch.nn.functional.cross_entropy(loc_pred, error_locations)
#     if metric:
#         loc_max = loc_pred.detach().clone().softmax(dim=-1).argmax(dim=-1)
#         loc_acc_nobug = accuracy_score(y_pred=loc_max[~has_bug],
#                                        y_true=error_locations[~has_bug])
#         loc_acc_bug = accuracy_score(y_pred=loc_max[has_bug],
#                                      y_true=error_locations[has_bug])
#     else:
#         loc_acc_nobug, loc_acc_bug, mean_repair_acc, joint_acc = None, None, None, None
#
#     # Repair loss is only computed at buggy samples (with error locations),
#     # using (negative) cross-entropy
#     repair_pred = logits[:, :, 1] * mask  # (nbatch, seq_length)
#     # repair_locations becomes a (nbatch, seq_length) Tensor of mostly 0s
#     #  it has values of 1 if it is a repair location. each row can have >1 label
#     target_mask = torch.vstack(
#         [torch.nn.functional.one_hot(rlab, num_classes=seq_length).sum(dim=0).to(rlab.device) for ii, rlab in enumerate(repair_loc)])
#     ### trying to match hellendoorn metric
#     pointer_logits = repair_pred + (1.0 - mask) * eps
#     pointer_probs = torch.nn.functional.softmax(pointer_logits, dim=-1)
#     target_probs = (target_mask * pointer_probs).sum(-1)
#     repair_loss = -torch.log(target_probs[has_bug] + eps).sum() / (eps + n_buggy)
#     # Since this is a multi-label problem, we need a threshold for marking a location.
#     # For simplicity and to follow the previous work, we use a probability thresh of 0.5.
#     if metric:
#         ### trying to match hellendoorn metric
#         target_loc_acc = (target_probs[has_bug].detach().clone() > 0.5).type(torch.FloatTensor)
#         # Also compute the joint accuracy, but on buggy samples only
#         lacc_by_sample = (loc_max[has_bug] == error_locations[has_bug]).type(torch.FloatTensor)
#         joint_acc = (lacc_by_sample * target_loc_acc).mean()
#     loss = loc_loss + repair_loss
#
#     # Reduce loss for logging.
#     averaged_loss = average_losses_across_data_parallel_group([loss])
#
#     # return loss, {'lm loss': averaged_loss[0]}
#     return loss, (loc_acc_nobug, loc_acc_bug, mean_repair_acc, joint_acc)


if __name__ == "__main__":

    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined

    varmisuse_multi_class(neox_args, num_classes=2)
