import datasets as trd
import os, sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from megatron.text_generation_utils import save_generated_samples_from_list, generate_samples_from_prompt
from megatron.utils import setup_for_inference_or_eval, is_mp_rank_0


run_generation = True
analyze = True

# Setup / load the model
model, neox_args = setup_for_inference_or_eval(inference=True, get_key_value=True, overwrite_values=None)

# Load the test dataset
dataset_path = neox_args.finetune_data_path  #"/export/data/vyvo/mpiricalplus/dataset/dataset_saved/"
split = "test"
print(f"Loading dataset from {dataset_path}{split}...")
d = trd.load_from_disk(os.path.join(dataset_path, split))
samples = d['code']
labels = d['mpi_labels']

if run_generation:
    # Now run the generation!
    print(f"Now running the prompts through model for generation...")
    save_generated_samples_from_list(neox_args, model, samples)

if analyze:
    print(f"Writing out the generated outputs vs labels...")
    with open(f'{os.getcwd()}/generated_output.pkl', 'rb') as f:
        gen_data = pickle.load(f)
    # Write output to text file for now
    output_path = "mpi_generation_outputs.txt"
    data_path = os.path.join(dataset_path, output_path)
    with open(os.path.join(dataset_path, output_path), 'wt') as outf:
        for gen, label in zip(gen_data, labels):
            outf.write(f"{label}\n{gen['text']}\n\n\n")
    print(f"Written to {data_path}!")

