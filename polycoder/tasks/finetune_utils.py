# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Finetune utilities."""

from functools import partial
import sys
import torch

from datetime import datetime

from megatron import print_rank_0
from megatron import mpu
from megatron.checkpointing import save_checkpoint
from megatron.training import evaluate_and_print_results
from megatron.training import training_log
from megatron.utils import (
    OverflowMonitor,
    get_noise_scale_logger
)

import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from varmisuse.data_varmisuse import get_batch as get_batch_fn


def finetune_forward_step(data_iterator, model, neox_args, timers, compute_loss_fn,
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

    lm_output = model.module.language_model((tokens, position_ids, attention_mask))
    output = model.module.forward_step(lm_output)

    # loss = cross_entropy_loss_func(labels, output)
    loss, _ = compute_loss_fn(labels, output, metric=False)

    if return_logits:
        return loss, output
    return loss


def finetune_step(neox_args, timers, data_iterator, model, optimizer, lr_scheduler, loss_function):
    # TODO: use custom batching
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
                compute_loss_fn=loss_function
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

            # VAV: We get the forward_microstep and forward timer logging errors after the optimizer step
            # if mpu.get_model_parallel_rank() == 1:
            #     import pdb; pdb.set_trace()

            # Update parameters.
            timers("optimizer").start()
            if neox_args.deepspeed:
                model.step()
            else:
                raise ValueError("Must be using deepspeed to run neox")
            timers("optimizer").stop()

            # if mpu.get_model_parallel_rank() == 1:
            #     import pdb; pdb.set_trace()

        overall_loss = reduce_losses(losses).mean()
        reduced_loss = {
            "lm_loss": overall_loss  #reduce_losses(losses).mean()
        }  # reduces losses across machines for logging

    if neox_args.precision == "fp16" and model.optimizer.overflow:
        skipped_iter = 1
    else:
        skipped_iter = 0

    return reduced_loss, skipped_iter


def finetune_loop(
    neox_args,
    timers,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
    loss_function
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

    # eval function
    eval_forward = partial(finetune_forward_step, compute_loss_fn=loss_function)

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
            loss_function=loss_function
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

        # # Logging.
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
                forward_step_func=eval_forward,
                data_iterator=valid_data_iterator,
                model=model,
                iteration=iteration,
                verbose=False,
                timers=timers,
            )
            # print_rank_0(f"VAV DEBUG losses: {overall_loss}")

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


def finetune(neox_args, model_setup_function, build_data_iters_function, loss_function):
    """Main finetuning program.
    Based off of training.py/pretrain, but allowing the use of custom functions for:
        - data iteration
        - batching (TODO)
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
            loss_function=loss_function,
        )

    # if neox_args.do_valid:
    #     # TODO: use same function call as above
    #     prefix = "the end of training for val data"
    #     evaluate_and_print_results(
    #         neox_args=neox_args,
    #         prefix=prefix,
    #         forward_step_func=finetune_step,
    #         data_iterator=valid_data_iterator,
    #         model=model,
    #         iteration=iteration,
    #         verbose=False,
    #         timers=timers,
    #     )

    if neox_args.save and iteration != 0:
        save_checkpoint(
            neox_args=neox_args,
            iteration=iteration,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    # if neox_args.do_test:
    #     # Run on test data.
    #     prefix = "the end of training for test data"
    #     evaluate_and_print_results(
    #         neox_args=neox_args,
    #         prefix=prefix,
    #         forward_step_func=finetune_step,
    #         data_iterator=test_data_iterator,
    #         model=model,
    #         iteration=0,  # iteration 0 in order to always use full test data
    #         verbose=True,
    #         timers=timers,
    #     )

