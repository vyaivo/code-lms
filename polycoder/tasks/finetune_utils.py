"""Finetune utilities."""

from collections.abc import Iterator
import gc
# from torch.profiler import profile, record_function, ProfilerActivity

from itertools import cycle
from functools import partial
import math
import os, sys
import torch

from datetime import datetime

import deepspeed
from megatron import print_rank_0
from megatron import mpu
from megatron.checkpointing import save_checkpoint
from megatron.training import evaluate_and_print_results
from megatron.training import training_log, train_step_pipe
from megatron.utils import (
    OverflowMonitor,
    get_noise_scale_logger,
    reduce_losses
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def finetune_forward_step(data_iterator, model, neox_args, timers, compute_loss_fn,
                          return_logits=False, compute_metric=False, get_batch_fn=None):
    """Forward step."""
    if neox_args.is_pipe_parallel:
        return model.eval_batch(data_iterator, return_logits=return_logits)

    # Get the batch.
    if timers is not None:
        timers("batch generator").start()
    tokens, (labels, loss_mask), attention_mask, position_ids = get_batch_fn(
        neox_args=neox_args, data_iterator=data_iterator
    )

    if timers is not None:
        timers("batch generator").stop()

    output = model((tokens, position_ids, attention_mask))

    # VAV: changed order of loss function inputs to be compatible with deepspeed pipelining defaults
    if compute_metric:
        loss, metric_info = compute_loss_fn(output, labels, metric=True)
        if return_logits:
            return loss, output, metric_info
        else:
            return loss, metric_info
    else:
        loss = compute_loss_fn(output, labels)
        if return_logits:
            return loss, output
        else:
            return loss


def evaluate_and_print_results(
    neox_args,
    eval_function,
    prefix,
    forward_step_func,
    data_iterator,
    model,
    verbose=False,
    timers=None,
):
    """Helper function to evaluate and dump results on screen."""
    eval_output_dict = eval_function(
        neox_args=neox_args,
        forward_step_fn=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        verbose=verbose,
        timers=timers,
    )
    string = f" validation results at {prefix} | "
    for k, v in eval_output_dict.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                k3 = "_".join([k, k2])
                string += f"{k3} value: {v2:.6E} | "
                # Removed wandb log -- see megatron.training for this
        else:
            string += f"{k} value: {v:.6E} | "
            # Removed wandb log -- see megatron.training for this

    length = len(string) + 1
    print_rank_0("-" * length)
    print_rank_0(string)
    print_rank_0("-" * length)


def finetune_step(neox_args, timers, data_iterator, model, optimizer, lr_scheduler, loss_function, custom_batch_fn):
    from megatron.training import backward_step
    """Single training step."""

    # Pipeline parallelism schedules forward/backward/step
    if neox_args.is_pipe_parallel:
        reduced_loss = train_step_pipe(
            neox_args=neox_args, timers=timers, model=model, data_iterator=data_iterator
        )
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
                compute_loss_fn=loss_function,
                get_batch_fn=custom_batch_fn,
            )
            timers("forward").stop()

            losses.append(loss)
            # Calculate gradients, reduce across processes, and clip.
            timers("backward").start()
            # try:
            backward_step(
                neox_args=neox_args,
                timers=timers,
                optimizer=optimizer,
                model=model,
                loss=loss,
            )
            # except Exception as e:
            #     print(e)
            #     import pdb; pdb.set_trace()
            timers("backward").stop()

            # Update parameters.
            timers("optimizer").start()
            if neox_args.deepspeed:
                model.step()
            else:
                raise ValueError("Must be using deepspeed to run neox")
            timers("optimizer").stop()

        # reduces losses across machines for logging
        overall_loss = reduce_losses(losses).mean()
        reduced_loss = {
            "lm_loss": overall_loss
        }

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
    loss_function,
    eval_function,
    eval_batch_fn,
    train_loader,
    val_loader
):
    """Finetune the model function. Modified from megatron.training to use a custom loss
    function, custom batching function.

    loss_function: a function which accepts (logits, labels) and returns the loss
    eval_function: currently unused, but should output metrics as well when implemented.
    custom_batch_fn: the custom batch function needed for the evaluation. note this assumes that
                     the training custom batch function is already set by a separate call,
                     perhaps when creating the data iterators -- e.g.:
                     model._megatron_batch_fn = partial(get_batch_pipe, neox_args=neox_args)
    """

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
    eval_forward = partial(finetune_forward_step, compute_loss_fn=loss_function,
                           return_logits=False, compute_metric=False, get_batch_fn=eval_batch_fn)

    # VAV DEBUG profiling
    # def trace_handler(p):
    #     output = p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=10)
    #     print(output)
    #     p.export_chrome_trace(f"debug_trace_{p.step_num}.json")

    # with profile(schedule=torch.profiler.schedule(wait=897, warmup=2, active=2, repeat=1),
    #              activities=[ProfilerActivity.CPU],
    #              with_stack=True,
    #              profile_memory=True,
    #              on_trace_ready=trace_handler) as prof:   # VAV DEBUG profiling

    # to monitor if we've skipped many iterations in a row and trigger an early exit
    refresh_count = 0
    overflow_monitor = OverflowMonitor(optimizer)
    while iteration < neox_args.train_iters:
        loss_dict, skipped_iter = finetune_step(
            neox_args=neox_args,
            timers=timers,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_function=loss_function,
            custom_batch_fn=eval_batch_fn,
        )
        iteration += 1
        refresh_count += 1
        if mpu.get_model_parallel_rank() == 0:
            if refresh_count >= (len(train_data_iterator) // neox_args.train_batch_size) - 1:
                print_rank_0("Starting new epoch. Refreshing training data iterator...")
                train_data_iterator = iter(train_loader)
                refresh_count = 0

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

        gc.collect()
        torch.cuda.empty_cache()

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
            # val_iter = ((iteration % neox_args.eval_interval) + 1) * neox_args.eval_iters
            # print(f'VAV DEBUG: val iter {val_iter} for iter {len(valid_data_iterator)}')
            # if val_iter >= len(valid_data_iterator):
            #     print_rank_0("Refreshing validation data iterator...")
            #     valid_data_iterator = iter(val_loader)
            prefix = "iteration {}".format(iteration)
            # # with torch.profiler.record_function("EVALUATE LOOP"):
            # try:
            evaluate_and_print_results(
                neox_args=neox_args,
                eval_function=eval_function,
                prefix=prefix,
                forward_step_func=eval_forward,
                data_iterator=valid_data_iterator,
                model=model,
                verbose=False,
                timers=timers
            )
            # print(prof.key_averages().table)
            # print(prof)
            # import pdb; pdb.set_trace()

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
        # prof.step()  # VAV DEBUG profiling

    return iteration


def finetune(neox_args, model_setup_function, build_data_function,
             loss_function, eval_function, custom_batch_fn):
    """Main finetuning program.
    Based off of training.py/pretrain, but allowing the use of custom functions for:
        - data iteration
        - batching
        - loss function
        - model, optimizer, lr scheduler
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
    data_output = build_data_function(neox_args=neox_args)
    train_dataloader, valid_dataloader = None, None
    if isinstance(data_output[0], Iterator):
        train_data_iterator, valid_data_iterator, test_data_iterator = data_output
        if mpu.get_model_parallel_rank() == 0:
            assert neox_args.train_iters <= len(train_data_iterator), \
                f"You want {neox_args.train_iters} train iterations, which exceeds the length of your iterator " \
                f"{len(train_data_iterator)}. Please change the data setup function to output dataloaders instead."
    elif isinstance(data_output[0], torch.utils.data.DataLoader):
        data_iters = [iter(d) if i == 0 else cycle(iter(d)) for i, d in enumerate(data_output)]
        train_data_iterator, valid_data_iterator, test_data_iterator = data_iters
        if True:  #len(train_data_iterator) > neox_args.train_iters:
            train_dataloader, valid_dataloader = data_output[0], data_output[1]
#        print_rank_0(f"VAV DEBUG: {len(train_data_iterator)} data iteration length")
    else:
        if mpu.get_model_parallel_rank() == 0:
            print(data_output[0])
            print(type(data_output[0]))
            raise ValueError("The custom data setup function did not produce the expected output")
        else:
            train_data_iterator, valid_data_iterator, test_data_iterator = (None, None, None)
    timers("train/valid/test data iterators").stop()

    #raise ValueError("Stopping here to figure out number of samples")
    # eval function
    eval_forward = partial(finetune_forward_step, compute_loss_fn=loss_function,
                           return_logits=False, compute_metric=False, get_batch_fn=custom_batch_fn)

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
            eval_function=eval_function,
            eval_batch_fn=custom_batch_fn,
            train_loader=train_dataloader,
            val_loader=valid_dataloader
        )


    if neox_args.do_valid:
        prefix = "the end of training for val data"
        evaluate_and_print_results(
            neox_args=neox_args,
            eval_function=eval_function,
            prefix=prefix,
            forward_step_func=eval_forward,
            data_iterator=valid_data_iterator,
            model=model,
            verbose=False,
            timers=timers
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
            eval_function=eval_function,
            prefix=prefix,
            forward_step_func=eval_forward,
            data_iterator=test_data_iterator,
            model=model,
            verbose=True,
            timers=timers,
        )
