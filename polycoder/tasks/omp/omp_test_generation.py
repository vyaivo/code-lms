import datasets as trd
import os, sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from megatron.text_generation_utils import save_generated_samples_from_list
from megatron.utils import setup_for_inference_or_eval


run_generation = False
analyze = True

# Setup / load the model
model, neox_args = setup_for_inference_or_eval(inference=True, get_key_value=True, overwrite_values=None)

# Load the test dataset
dataset_path = neox_args.finetune_data_path
split = "test"
print(f"Loading dataset from {dataset_path}{split}...")
d = trd.load_from_disk(os.path.join(dataset_path, split))
samples = d['code']
labels = d['pragma']
print(f"Processing {len(samples)} samples in this dataset.")
max_len = 2048 - neox_args.maximum_tokens

if run_generation:
    # Now run the generation!
    print(f"Now running the prompts through model for generation...")
    save_generated_samples_from_list(neox_args, model, samples, max_input_length=max_len)

if analyze:
    print(f"Writing out the generated outputs vs labels...")
    gen_data_path = f'{os.getcwd()}/generated_output.pkl'
    if not os.path.exists(gen_data_path):
        gen_data_path = f'{dataset_path}/generated_output.pkl'
        print(f"Loading file from {dataset_path}/generated_output.pkl")
    with open(gen_data_path, 'rb') as f:
        data_list = pickle.load(f)
        gen_data = iter(data_list)
    ngen, nsamples = len(data_list), len(labels)
    print(f"Generated {ngen} out of {nsamples} samples")
    tokenizer = neox_args.tokenizer
    # Write output to text file for now
    output_path = f"omp_generation_outputs.txt"
    data_path = os.path.join(dataset_path, output_path)
    ilabels = iter(labels)
    isamples = iter(samples)
    with open(os.path.join(dataset_path, output_path), 'wt') as outf:
        i, j, skipped = 0, 0, 0
        while True:
            try:
                label = next(ilabels)
                sample = next(isamples)
                i += 1
                toks = tokenizer.tokenize(sample)
                if len(toks) >= (max_len):
                    skipped += 1
                    continue
                else:
                    # tokenize the label for printing
                    label = tokenizer.detokenize(tokenizer.tokenize(label))
                    gen = next(gen_data)
                    j += 1
                    outf.write(f"{label}\n{gen['text']}\n\n\n")
                    if j >= ngen:
                        print(f"We have iterated through all {j} generations")
                        print(f"We skipped {skipped} out of {nsamples - ngen} to skip")
            except StopIteration:
                if skipped != (nsamples - ngen):
                    print(f"Skipped wrong number of samples -- should be {nsamples - ngen} not {skipped}!")
                    print(i, j)
                    import pdb; pdb.set_trace()
                break
            except Exception as e:
                print(e)
                print(i, j)
                import pdb; pdb.set_trace()
    print(f"Written to {data_path}!")

