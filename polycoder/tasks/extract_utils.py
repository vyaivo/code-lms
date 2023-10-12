"""Utilities to extract LM representations."""
import deepspeed
import numpy as np
import torch

from math import ceil

from megatron import print_rank_0


def extract_forward_step(neox_args, data_iterator, model, timers,
                         get_batch_fn, batch_output_key="source"):
    """Eval forward step that only outputs logits and token sources, no loss."""
    if neox_args.is_pipe_parallel:
        return model.inference_extract(data_iterator)  #, return_logits=True)
    else:
        raise ValueError("Model is not pipe parallel!")

    # Error: when we converted loaded pipeline model to sequential, we get a Half/Float error
    # seems to be due to the differences in devices? LayerNorm is on CPU while attention is on CUDA, and
    # torch 1.8.1 doesn't support bfloat16 yet...

    # Generate a batch, assuming a pipeline.
    # if timers is not None:
    #     timers("batch generator").start()
    # data = next(data_iterator)
    # (tokens, position_ids, attention_mask), (labels, loss_mask) = \
    #     get_batch_fn(data, neox_args=neox_args)
    # if timers is not None:
    #     timers("batch generator").stop()

    # # Get the batch.
    # if timers is not None:
    #     timers("batch generator").start()
    # tokens, (labels, _), attention_mask, position_ids = get_batch_fn(
    #     neox_args=neox_args, data_iterator=data_iterator
    # )
    # if timers is not None:
    #     timers("batch generator").stop()
    #
    # import pdb; pdb.set_trace()
    # output = model((tokens, position_ids, attention_mask))
    # out_labels = tokens[batch_output_key] if batch_output_key else None

    # return output, labels


def extract_loop(
    neox_args,
    timers,
    model,
    data_iterator,
    data_iter2,
    eval_batch_fn,
    output_dir,
    batch_data_key=None,
    max_seq_length=2048,
    verbose=True,
    samples_per_split_bin=None
):
    """
    Simply run the data through the model and extract the logit representations.

    eval_batch_fn: custom batching function to use in an evaluation loop
    output_dir: where to store the output representations
    batch_data_key: the key in the dataset batch that needs to be stored along with the representations
    """
    # assert not neox_args.is_pipe_parallel, "Please turn pipe parallelism off to extract logits"
    num_samples = len(data_iterator)
    flush_write_period = 100
    total_iters = int(ceil(num_samples / neox_args.batch_size))

    model.eval()
    # Evaluation loop
    iteration = 0
    i, j = 0, 0
    with torch.no_grad():
        while True:
            if verbose and iteration % neox_args.log_interval == 0:
                print_rank_0(
                    "Evaluating iter {}/{}".format(iteration, total_iters)
                )
            try:
                prefix = "iteration {}".format(iteration)
                logits = extract_forward_step(neox_args, data_iterator, model, timers,
                                                      eval_batch_fn, batch_data_key)
                logits = logits.cpu().numpy()
                data = next(data_iter2)
                data_tokens = data['input_ids'].squeeze(1)
                if data_tokens.ndim > 2:
                    data_tokens = data_tokens.squeeze()
                    assert data_tokens.ndim == 2
                assert logits.shape[0] == data_tokens.shape[0], f"Batch sizes don't match!"
                assert logits.shape[1] == data_tokens.shape[1], f"Sequence lengths don't match!"
                text_labels = data['source']
                if isinstance(text_labels[0], list) and len(text_labels) == 1:
                    text_labels = text_labels[0]
                assert logits.shape[0] == len(text_labels), "Batch size of logits doesn't match label data"
            except StopIteration:  # out of data
                print("We're out of data!")
                break
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
            if iteration == 0:
                print_rank_0("Creating memory mapped tensors...")
                if samples_per_split_bin:
                    seq_bins = np.append(np.array(list(samples_per_split_bin.keys())), np.array([0]))
                    seq_bin_count = list(samples_per_split_bin.values())
                    output = [None] * len(seq_bin_count)
                    for b, (seq_max, n_bin_samples) in enumerate(zip(seq_bins[:-1], seq_bin_count)):
                        tensor_shape = [n_bin_samples, seq_max, logits.shape[-1]]
                        output[b] = np.memmap(f"{output_dir}/extracted_tensors_{seq_max}.npy", dtype=np.float32,
                                              mode='w+', shape=tuple(tensor_shape))
                        print_rank_0(f"Tensor with shape {tensor_shape}")
                else:
                    tensor_shape = [num_samples, max_seq_length, logits.shape[-1]]
                    output = np.memmap(f"{output_dir}/extracted_tensors.npy", dtype=np.float32,
                                       mode='w+', shape=tuple(tensor_shape))
                    print_rank_0(f"Tensor with shape {tensor_shape}")
                if batch_data_key:
                    if samples_per_split_bin:
                        path_dict = {}
                        for b, (seq_max, n_bin_samples) in enumerate(zip(seq_bins[:-1], seq_bin_count)):
                            path_dict[b] = np.array([None]*n_bin_samples, dtype=object)
                    else:
                        path_array = np.array([None]*num_samples, dtype=object)
            this_batch_size, this_seq_length = logits.shape[0:2]
            if samples_per_split_bin:
                bin_id = np.digitize(np.array(this_seq_length), seq_bins) - 1
                i = j
                j = i + this_batch_size
                output[bin_id][i:j, :this_seq_length, :] = logits
                if batch_data_key:
                    path_dict[bin_id][i:j] = text_labels
                if j == output[bin_id].shape[0]:
                    print_rank_0("Resetting indices for next memmap tensor...")
                    i, j = 0, 0
                # print_rank_0(i, j)
            else:
                i = j
                j = i + this_batch_size
                output[i:j, :this_seq_length, :] = logits
                if batch_data_key:
                    path_array[i:j] = text_labels
            # When contiguous memory optimizations are enabled, the buffers
            # allocated by the optimizations are deallocated during backward pass
            # in the absence of backward pass the buffers should be reset after each
            # forward pass
            if neox_args.deepspeed and neox_args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()
            iteration += 1
            if (iteration % flush_write_period) == 0:
                print(f"Flushing data to {output_dir} at {iteration}/{total_iters} iterations...")
                if samples_per_split_bin:
                    for f_output in output:
                        f_output.flush()
                    if batch_data_key:
                        np.savez(f"{output_dir}/source_paths.npz", path_dict)
                else:
                    output.flush()
                    if batch_data_key:
                        np.savez(f"{output_dir}/source_paths.npz", path_array)
    # Save one last time!
    if samples_per_split_bin:
        for f_output in output:
            f_output.flush()
        if batch_data_key:
            np.savez(f"{output_dir}/source_paths.npz", path_dict)
    else:
        output.flush()
        if batch_data_key:
            np.savez(f"{output_dir}/source_paths.npz", path_array)
    print(f"Finished writing data to {output_dir}!")

