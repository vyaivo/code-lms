# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Finetune utilities."""

# from functools import partial
import sys
import torch

from datetime import datetime

from megatron import print_rank_0
from megatron import mpu
from megatron.checkpointing import save_checkpoint
from megatron.training import evaluate_and_print_results
# from megatron.training import get_batch
from megatron.training import training_log
from megatron.utils import (
    OverflowMonitor,
    get_noise_scale_logger
)
from megatron.utils import average_losses_across_data_parallel_group

import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from megatron.training import setup_model_and_optimizer
# from megatron import get_num_microbatches
# from megatron import get_timers
# from megatron.core.enums import ModelType
# from megatron.checkpointing import load_checkpoint
# from megatron.utils import calc_params_l2_norm
# from megatron.utils import check_adlr_autoresume_termination
# from megatron.utils import Timers


def finetune_step(neox_args, timers, data_iterator, model, optimizer, lr_scheduler):
    # TODO: use lr scheduler?
    # TODO: use custom batching and loss functions
    from megatron.training import backward_step
    from megatron.utils import reduce_losses
    """Single training step."""

    # Pipeline parallelism schedules forward/backward/step
    if neox_args.is_pipe_parallel:
        raise NotImplementedError
    else:
        losses = []
        for _ in range(neox_args.gradient_accumulation_steps):
            # Forward model for one step.
            timers("forward").start()
            loss = finetune_forward_step(
                neox_args=neox_args,
                timers=timers,
                data_iterator=data_iterator,
                model=model,
            )
            timers("forward").stop()
            losses.append(loss)
            # Calculate gradients, reduce across processes, and clip.
            timers("backward").start()
            backward_step(
                neox_args=neox_args,
                timers=timers,
                optimizer=optimizer,
                model=model,
                loss=loss,
            )
            timers("backward").stop()
            # Update parameters.
            timers("optimizer").start()
            if neox_args.deepspeed:
                model.step()
            else:
                raise ValueError("Must be using deepspeed to run neox")
            timers("optimizer").stop()
        reduced_loss = {
            "lm_loss": reduce_losses(losses).mean()
        }  # reduces losses across machines for logging

    if neox_args.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    return reduced_loss, skipped_iter


from varmisuse.data_varmisuse import get_batch as get_batch_fn
from varmisuse.data_varmisuse import compute_varmisuse_loss as compute_loss_fn


def finetune_forward_step(data_iterator, model, neox_args, timers,
                          return_logits=False):
    """Forward step."""
    if neox_args.is_pipe_parallel:
        raise NotImplementedError

    # Get the batch.
    if timers is not None:
        timers("batch generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch_fn(
        neox_args=neox_args, data_iterator=data_iterator
    )
    if timers is not None:
        timers("batch generator").stop()

    # print_rank_0("VAV DEBUG:", model)
    lm_output = model.module.language_model((tokens, position_ids, attention_mask))
    if model.module.post_process:
        output = model.module.post_process_step(lm_output)
    else:
        output = lm_output

    # ((data_b["error_location"], data_b["repair_targets"]), (data_b["has_bug"], data_b["repair_candidates"]))
    # print_rank_0(output)  # 2 x 2
    # loss = cross_entropy_loss_func(labels, output)
    loss, _ = compute_loss_fn(labels, output, metric=False)

    if return_logits:
        return loss, output
    return loss


def finetune_loop(
    neox_args,
    timers,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
):
    """Train the model function. Simply modified from megatron.training to use a custom loss
    function and custom batching function"""
    # TODO: use custom batching and loss functions

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = neox_args.iteration

    timers("interval time").start()
    report_memory_flag = True

    # get noise scale logger (if neox_args.log_gradient_noise_scale is True)
    noise_scale_logger = get_noise_scale_logger(neox_args)

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    overflow_monitor = OverflowMonitor(optimizer)
    while iteration < neox_args.train_iters:
        loss_dict, skipped_iter = finetune_step(
            neox_args=neox_args,
            timers=timers,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        iteration += 1

        overflow_monitor.check(skipped_iter)  # check for repeated overflow
        if neox_args.log_gradient_noise_scale:  # log noise scale if applicable
            noise_scale_logger.update()

        # get learning rate (if present) - if doing soft prompt tuning + pipe parallel, you
        # may have no tunable parameters on a specific rank
        if optimizer.param_groups:
            lr = optimizer.param_groups[0].get("lr", 0)
        else:
            lr = 0

        # Logging.
        report_memory_flag = training_log(
            neox_args=neox_args,
            timers=timers,
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=lr,
            iteration=iteration,
            loss_scale=optimizer.cur_scale if neox_args.precision == "fp16" else None,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            model=model,
            optimizer=optimizer,
            noise_scale_logger=noise_scale_logger,
        )

        # Checkpointing
        if (
            neox_args.save
            and neox_args.save_interval
            and iteration % neox_args.save_interval == 0
        ):
            save_checkpoint(
                neox_args=neox_args,
                iteration=iteration,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        # Evaluation
        if (
            neox_args.eval_interval
            and iteration % neox_args.eval_interval == 0
            and neox_args.do_valid
        ):
            prefix = "iteration {}".format(iteration)
            evaluate_and_print_results(
                neox_args=neox_args,
                prefix=prefix,
                forward_step_func=finetune_step,
                data_iterator=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=False,
                timers=timers,
            )

        if neox_args.exit_interval and iteration % neox_args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rank = torch.distributed.get_rank()
            print_rank_0(
                "rank: {} | time: {} | exiting the program at iteration {}".format(
                    rank, time_str, iteration
                )
            )
            sys.exit()

    return iteration


def finetune(neox_args, model_setup_function, build_data_iters_function):
    """Main finetuning program.
    Based off of training.py/pretrain, but allowing the use of custom functions for:
        - data iteration
        - batching (TODO)
        - computing loss function (TODO)
    """
    from megatron.utils import (
        Timers,
        init_wandb,
    )
    from megatron.initialize import initialize_megatron

    # setup logging and timers
    init_wandb(neox_args=neox_args)
    timers = Timers(
        use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer
    )

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(neox_args=neox_args)

    # Model, optimizer, and learning rate.
    timers("model and optimizer").start()
    model, optimizer, lr_scheduler = model_setup_function(
        neox_args=neox_args, inference=False, get_key_value=True
    )
    timers("model and optimizer").stop()

    # Data stuff.
    timers("train/valid/test data iterators").start()
    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    ) = build_data_iters_function(neox_args=neox_args)
    timers("train/valid/test data iterators").stop()

    # Print setup timing.
    print_rank_0("done with setups ...")
    timers.log(["model and optimizer", "train/valid/test data iterators"])
    print_rank_0("training ...")

    iteration = 0
    if neox_args.do_train and neox_args.train_iters > 0:
        iteration = finetune_loop(
            neox_args=neox_args,
            timers=timers,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_data_iterator=train_data_iterator,
            valid_data_iterator=valid_data_iterator,
        )

    if neox_args.do_valid:
        prefix = "the end of training for val data"
        evaluate_and_print_results(
            neox_args=neox_args,
            prefix=prefix,
            forward_step_func=finetune_step,
            data_iterator=valid_data_iterator,
            model=model,
            iteration=iteration,
            verbose=False,
            timers=timers,
        )

    if neox_args.save and iteration != 0:
        save_checkpoint(
            neox_args=neox_args,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    if neox_args.do_test:
        # Run on test data.
        prefix = "the end of training for test data"
        evaluate_and_print_results(
            neox_args=neox_args,
            prefix=prefix,
            forward_step_func=finetune_step,
            data_iterator=test_data_iterator,
            model=model,
            iteration=0,  # iteration 0 in order to always use full test data
            verbose=True,
            timers=timers,
        )


# def old_finetune(neox_args, train_valid_datasets_provider, model_provider,
#              model_type=ModelType.encoder_or_decoder,
#              forward_step=_cross_entropy_forward_step,
#              end_of_epoch_callback_provider=None,
#              task_collate_fn=None):
#     """Main finetune function used across all tasks."""
#     # args = get_args()
#     timers = Timers(use_wandb=neox_args.use_wandb,
#                     tensorboard_writer=neox_args.tensorboard_writer)
#
#     # Train and validation data loaders.
#     timers('train/valid/test dataset/dataloader', log_level=0).start()
#     if neox_args.train_iters > 0:
#         train_dataset, valid_dataset = train_valid_datasets_provider()
#         train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
#             train_dataset, valid_dataset, task_collate_fn)
#     timers('train/valid/test dataset/dataloader').stop()
#
#     # Build calback function.
#     timers('callback function').start()  #, log_level=0).start()
#     end_of_epoch_callback = None
#     if end_of_epoch_callback_provider is not None:
#         end_of_epoch_callback = end_of_epoch_callback_provider()
#     timers('callback function').stop()
#
#     # Build model, optimizer and learning rate scheduler.
#     timers('model and optimizer').start()  #, log_level=0).start()
#     model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type)
#     timers('model and optimizer').stop()
#
#     # If pretrained checkpoint is provided and we have not trained for
#     # any iteration (i.e., iteration is zero), then load the pretrained
#     # checkpoint.
#     # timers('pretrained checkpoint').start(barrier=True)
#     if neox_args.iteration == 0 and neox_args.pretrained_checkpoint is not None:
#         original_load = neox_args.load
#         neox_args.load = neox_args.pretrained_checkpoint
#         original_rng = neox_args.no_load_rng
#         neox_args.no_load_rng = True
#         _ = load_checkpoint(model, None, None)
#         neox_args.load = original_load
#         neox_args.no_load_rng = original_rng
#         # This is critical when only model is loaded. We should make sure
#         # main parameters are also updated.
#         optimizer.reload_model_params()
#     # timers('pretrained checkpoint').stop()
#
#     # Print setup timing.
#     print_rank_0('done with setups ...')
#     timers.log(['train/valid/test dataset/dataloder', 'callback function',
#                 'model and optimizer', 'pretrained checkpoint']) #, barrier=True)
#     print_rank_0('training ...')
#
#     # Finetune the model.
#     if neox_args.train_iters > 0:
#         _train(model, optimizer, opt_param_scheduler, forward_step,
#                train_dataloader, valid_dataloader, end_of_epoch_callback)
#     # Or just evaluate.
#     else:
#         if end_of_epoch_callback is not None:
#             print_rank_0('evaluation only mode, setting epoch to -1')
#             end_of_epoch_callback(model, epoch=-1, output_predictions=True)
#     print_rank_0('done :-)')



# def build_data_loader(dataset, micro_batch_size, num_workers, drop_last,
#         task_collate_fn=None):
#     """Data loader. Note that batch-size is the local (per GPU) batch-size."""
#
#     # Sampler.
#     world_size = mpu.get_data_parallel_world_size()
#     rank = mpu.get_data_parallel_rank()
#     sampler = torch.utils.data.distributed.DistributedSampler(
#         dataset, num_replicas=world_size, rank=rank)
#
#     # Data loader. Note that batch size is the per GPU batch size.
#     data_loader = torch.utils.data.DataLoader(dataset,
#                                               batch_size=micro_batch_size,
#                                               sampler=sampler,
#                                               shuffle=False,
#                                               num_workers=num_workers,
#                                               drop_last=drop_last,
#                                               pin_memory=True,
#                                               collate_fn=task_collate_fn)
#
#     return data_loader
#
#
# def _build_infinite_size_dataloader(dataloader):
#     """Build a looped dataloader with infinite size."""
#
#     iterator = dataloader.__iter__()
#     while True:
#         try:
#             yield iterator.__next__()
#         except StopIteration:
#             iterator = dataloader.__iter__()


# def _build_train_valid_dataloaders(train_dataset, valid_dataset,
#     task_collate_fn=None):
#     """Traing and validation dataloaders."""
#     # args = get_args()
#
#     print_rank_0('building train and validation dataloaders ...')
#     # Training dataset.
#     train_dataloader = build_data_loader(train_dataset, args.micro_batch_size,
#                                          args.num_workers, not args.keep_last,
#                                          task_collate_fn)
#     # Set the training iterations.
#     args.train_iters_per_epoch = len(train_dataloader)
#     args.train_iters = args.epochs * args.train_iters_per_epoch
#     # Validation dataset. For this dataset, we do not need to set up
#     # shuffling so we can just use a simple infinite loop.
#     valid_dataloader_ = build_data_loader(valid_dataset, args.micro_batch_size,
#                                           args.num_workers, not args.keep_last,
#                                           task_collate_fn)
#     valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)
#
#     # Now that we've built the data loaders, set batch_size arguments
#     # to the actual batch size the model will see for this dataset.
#     # This is necessary so pipeline transfers know what size they are
#     # and the LR schedule, which is based on samples seen, gets set
#     # correctly.
#     args.orig_micro_batch_size = args.micro_batch_size
#     args.orig_global_batch_size = args.global_batch_size
#     if hasattr(train_dataset, 'sample_multiplier'):
#         # If our dataset as a sample_multiplier attribute that means
#         # each "sample" from the dataset actually has multiple samples
#         # that will collapse into the batch dimension (for example in
#         # the RACE dataset that has several options), we need to
#         # account for that when setting the micro batch size.
#         args.micro_batch_size *= train_dataset.sample_multiplier
#         args.global_batch_size *= train_dataset.sample_multiplier
#
#     return train_dataloader, valid_dataloader