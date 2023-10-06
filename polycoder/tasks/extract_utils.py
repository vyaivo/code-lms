"""Utilities to extract LM representations."""
import deepspeed
import numpy as np
import torch

from math import ceil

from megatron import print_rank_0


def extract_forward_step(neox_args, data_iterator, model, timers, get_batch_fn, batch_output_key="source"):
    """Eval forward step that only outputs logits and token sources, no loss."""
    # Get the batch.
    if timers is not None:
        timers("batch generator").start()
    tokens, _, attention_mask, position_ids = get_batch_fn(
        neox_args=neox_args, data_iterator=data_iterator
    )
    if timers is not None:
        timers("batch generator").stop()

    output = model((tokens, position_ids, attention_mask))
    out_labels = tokens[batch_output_key] if batch_output_key else None

    return output, out_labels


def extract_loop(
    neox_args,
    timers,
    model,
    data_iterator,
    eval_batch_fn,
    output_dir,
    batch_data_key=None,
    verbose=True
):
    """
    Simply run the data through the model and extract the logit representations.

    eval_batch_fn: custom batching function to use in an evaluation loop
    output_dir: where to store the output representations
    batch_data_key: the key in the dataset batch that needs to be stored along with the representations
    """
    assert not neox_args.is_pipe_parallel, "Please turn pipe parallelism off to extract logits"
    batch_size = neox_args.batch_size
    num_samples = len(data_iterator)
    flush_write_period = 100
    total_iters = int(ceil(num_samples / batch_size))

    model.eval()
    # Evaluation loop
    iteration = 0
    with torch.no_grad():
        while True:
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0(
                    "Evaluating iter {}/{}".format(iteration, neox_args.eval_iters)
                )
            try:
                prefix = "iteration {}".format(iteration)
                logits, tokens = extract_forward_step(neox_args, data_iterator, model, timers,
                                                      eval_batch_fn, batch_data_key)
            except StopIteration:  # out of data
                break
            if iteration == 0:
                import pdb; pdb.set_trace()
                tensor_shape = [num_samples] + list(logits.shape[1:])
                output = np.memmap(f"{output_dir}/extracted_tensors.npy", dtype=np.float32,
                                   mode='w', shape=tuple(tensor_shape))
                if batch_data_key:
                    path_array = np.array([None]*num_samples, dtype=object)
            i, j = iteration * batch_size, (iteration + 1) * batch_size
            output[i:j, :] = logits.to_numpy()
            if batch_data_key:
                path_array[i:j] = tokens
            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()
            iteration += 1
            if (iteration % flush_write_period) == 0:
                print(f"Flushing data to {output_dir} at {iteration}/{total_iters} iterations...")
                output.flush()
    output.flush()
    if batch_data_key:
        np.savez(f"{output_dir}/source_paths.npz", path_array)
    print(f"Finished writing data to {output_dir}!")

