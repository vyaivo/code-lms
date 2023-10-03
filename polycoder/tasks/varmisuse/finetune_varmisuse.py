"""VarMisuse task -- finetuning/evaluation."""

import deepspeed
from functools import partial
import math
import torch
# from pathlib import Path
# from pprint import pformat

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from megatron.neox_arguments import NeoXArgs
from megatron import mpu, print_rank_0
from megatron.training import get_optimizer, get_learning_rate_scheduler, evaluate
from megatron.utils import get_total_params, reduce_losses

from megatron.checkpointing import load_checkpoint
from megatron.model.classification import Classification
from tasks.finetune_utils import finetune
# from tasks.eval_utils import accuracy_func_provider
# from megatron.arguments import core_transformer_config_from_args

from data_varmisuse import build_train_valid_test_data_iterators, get_batch_pipe

from sklearn.metrics import accuracy_score


# TODO: Run Docker test E2E


def varmisuse_multi_class(neox_args, num_classes):

    def model_optim_lr_setup(neox_args, num_classes,
                             inference=False, get_key_value=None):
        """Build the model."""
        print_rank_0('building classification model for variable misuse task')
        model = Classification(neox_args, num_classes=num_classes, post_process=False,
                               parallel_output=True, topology=mpu.get_topology(), get_key_value=True,
                               loss_fn=compute_varmisuse_loss
                               ) #, load_lm_checkpoint=neox_args.load)
        if not neox_args.is_pipe_parallel:
            # Export PipeParallel model to nn.Sequential model to avoid the overhead of deepspeed's pipe parallel training
            model = model.to_sequential()

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
                model.set_batch_fn(partial(get_batch_pipe, neox_args=neox_args))
        else:
            raise ValueError("Must be using deepspeed to run neox")

        if neox_args.load is not None:
            neox_args.iteration = load_checkpoint(neox_args, model,
                                                  optimizer=optimizer,
                                                  lr_scheduler=lr_scheduler,
                                                  inference=False, iteration=None)
            print_rank_0(
                f"Loading checkpoint and starting from iteration {neox_args.iteration}"
            )
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
             compute_varmisuse_loss, evaluate)  # evaluate_varmisuse)


def compute_varmisuse_loss(logits, label_data, metric=False):
    # Unpack label data
    # (error_locations, repair_loc), (lengths, has_bug, repair_cands) = label_data
    # print("VAV DEBUG dp & mp rank:", mpu.get_data_parallel_rank(), mpu.get_model_parallel_rank())
    # print("VAV DEBUG devices:", error_locations.device, target_mask.device, mask.device, has_bug.device, logits.device)

    # import pdb; pdb.set_trace()
    nbatch, seq_length, _ = logits.shape

    # VAV: deepspeed code automatically moves the label data onto the correct device
    # label_data = [l.to(logits.device) for l in label_data]
    error_locations, target_mask, mask, has_bug = label_data
    # error_locations = error_locations.squeeze()
    n_buggy = torch.sum(has_bug).item()
    eps = 1e-16

    # Localization loss is simply calculated with sparse CE
    loc_pred = logits[:, :, 0]  # * mask  # (nbatch, seq_length)
    # loc_loss = torch.nn.functional.cross_entropy(loc_pred, error_locations)
    loc_loss = torch.nn.functional.binary_cross_entropy_with_logits(loc_pred, error_locations.type(loc_pred.dtype))
    if metric:
        # TODO: fix this for multi-token targets
        loc_max = loc_pred.detach().clone().softmax(dim=-1).argmax(dim=-1)
        loc_acc_nobug = accuracy_score(y_pred=loc_max[~has_bug],
                                       y_true=error_locations[~has_bug])
        loc_acc_bug = accuracy_score(y_pred=loc_max[has_bug],
                                     y_true=error_locations[has_bug])

    # Repair loss is only computed at buggy samples (with error locations),
    # using (negative) cross-entropy
    repair_pred = logits[:, :, 1] * mask  # (nbatch, seq_length)
    # repair_locations becomes a (nbatch, seq_length) Tensor of mostly 0s
    #  it has values of 1 if it is a repair location. each row can have >1 label
    ### trying to match hellendoorn metric
    pointer_logits = repair_pred + (1.0 - mask) * eps
    pointer_probs = torch.nn.functional.softmax(pointer_logits, dim=-1)
    if n_buggy > 0:
        target_probs = (target_mask * pointer_probs).sum(-1)
        repair_loss = -torch.log(target_probs[has_bug] + eps).sum() / (eps + n_buggy)
        loss = loc_loss + repair_loss
    else:
        # ignore repair loss element since there are no buggy samples in the batch
        loss = loc_loss
    if metric:
        # Since this is a multi-label problem, we need a threshold for marking a location.
        # For simplicity and to follow the previous work, we use a probability thresh of 0.5.
        ### trying to match hellendoorn metric
        target_loc_acc = (target_probs[has_bug].detach().clone() > 0.5).type(torch.FloatTensor)
        mean_repair_acc = target_loc_acc.mean()
        # Also compute the joint accuracy, but on buggy samples only
        lacc_by_sample = (loc_max[has_bug] == error_locations[has_bug]).type(torch.FloatTensor)
        joint_acc = (lacc_by_sample * target_loc_acc).mean()


    # Reduce loss for logging.
    # averaged_loss = average_losses_across_data_parallel_group([loss])

    # return loss, {'lm loss': averaged_loss[0]}
    if metric:
        return loss, (loc_acc_nobug, loc_acc_bug, mean_repair_acc, joint_acc)
    else:
        return loss


def evaluate_varmisuse(neox_args, forward_step_fn, data_iterator, model, verbose=False, timers=None):
    # VAV: I wanted this to be called during training, but it's too difficult to get the other
    # metrics without altering deepspeed functions. So let's just do it after loading the model at
    # certain checkpoints...
    """Evaluation.
    neox_args: NeoX Arguments
    forward_step_fn: function with args `neox_args, timers,
                    data_iterator & model that will run a forward pass on the model
    data_iterator: Iterator that iterates over batches of data. Should return data in the form:
                    {'text': np.array([tokens], dtype=np.int64)}
                    where the size of the array is the model's context size + 1
                    (`get_batch` transforms it into inputs / labels)
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    losses = []
    metric_lists = [[], [], [], []]
    metric_labels = ['loc_acc_nobug', 'loc_acc_bug', 'repair_acc', 'joint_acc']

    with torch.no_grad():
        iteration = 0
        while iteration < neox_args.eval_iters:
            iteration += 1
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0(
                    "Evaluating iter {}/{}".format(iteration, neox_args.eval_iters)
                )

            # although we're not accumulating gradients here, we count one iter as train_batch_size_per_gpu * g.a.s
            # to be consistent with deepspeed's pipe parallel engine
            # since pipe parallel already takes gas into account - default to 1 here if pipe parallel is true
            for _ in range(
                1
                if neox_args.is_pipe_parallel
                else neox_args.gradient_accumulation_steps
            ):
                # Forward evaluation
                loss, metric_info = forward_step_fn(
                    model=model,
                    data_iterator=data_iterator,
                    neox_args=neox_args,
                    timers=timers,
                )
                for i, metric in enumerate(metric_info):
                    metric_lists[i].append(metric)
                losses.append(loss)

            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

    # reduces losses across processes for logging & run eval harness tasks
    eval_results = {"lm_loss": reduce_losses(losses).mean().item()}
    eval_results["lm_loss_ppl"] = math.exp(eval_results["lm_loss"])
    for i, k in enumerate(metric_labels):
        eval_results[k] = reduce_losses(metric_lists[i]).mean().item()

    # Move model back to the train mode.
    model.train()
    return eval_results


# def compute_varmisuse_acc(logits, label_data):
#     nbatch, seq_length, _ = logits.shape
#
#     # Unpack label data
#     error_locations, target_mask, mask, has_bug = label_data
#     # n_buggy = torch.sum(has_bug).item()
#     eps = 1e-16
#
#     # Localization loss is simply calculated with sparse CE
#     loc_pred = logits[:, :, 0]
#     # loc_loss = torch.nn.functional.cross_entropy(loc_pred, error_locations)
#     loc_max = loc_pred.detach().clone().softmax(dim=-1).argmax(dim=-1)
#     loc_acc_nobug = accuracy_score(y_pred=loc_max[~has_bug],
#                                    y_true=error_locations[~has_bug])
#     loc_acc_bug = accuracy_score(y_pred=loc_max[has_bug],
#                                  y_true=error_locations[has_bug])
#
#     # Repair loss is only computed at buggy samples (with error locations),
#     # using (negative) cross-entropy
#     repair_pred = logits[:, :, 1] * mask  # (nbatch, seq_length)
#     # repair_locations becomes a (nbatch, seq_length) Tensor of mostly 0s
#     #  it has values of 1 if it is a repair location. each row can have >1 label
#     ### trying to match hellendoorn metric
#     pointer_logits = repair_pred + (1.0 - mask) * eps
#     pointer_probs = torch.nn.functional.softmax(pointer_logits, dim=-1)
#     target_probs = (target_mask * pointer_probs).sum(-1)
#     # Since this is a multi-label problem, we need a threshold for marking a location.
#     # For simplicity and to follow the previous work, we use a probability thresh of 0.5.
#     ### trying to match hellendoorn metric
#     target_loc_acc = (target_probs[has_bug].detach().clone() > 0.5).type(torch.FloatTensor)
#     mean_repair_acc = target_loc_acc.mean()
#     # Also compute the joint accuracy, but on buggy samples only
#     lacc_by_sample = (loc_max[has_bug] == error_locations[has_bug]).type(torch.FloatTensor)
#     joint_acc = (lacc_by_sample * target_loc_acc).mean()
#
#     return loc_acc_nobug, loc_acc_bug, mean_repair_acc, joint_acc


if __name__ == "__main__":

    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined

    varmisuse_multi_class(neox_args, num_classes=2)
