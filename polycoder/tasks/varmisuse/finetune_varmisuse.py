"""VarMisuse task -- finetuning/evaluation."""

#from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.model.classification import Classification
from tasks.eval_utils import accuracy_func_provider
from tasks.finetune_utils import finetune
from megatron.arguments import core_transformer_config_from_args

import torch


def varmisuse_multi_class(neox_args, num_classes):

    def train_valid_datasets_provider(data_path='/export/data/vyvo/huggingface/'):
        """Build train and validation dataset."""
        tokenized_datasets = trd.load_from_disk(f"{data_path}varmisuse_hfdata") 
 
        return tokenized_datasets['train'], tokenized_datasets['validation']

    def model_provider(neox_args):
        """Build the model."""
        timers = Timers(
            use_wandb=neox_args.use_wandb, tensorboard_writer=neox_args.tensorboard_writer
        )

        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(neox_args=neox_args)

        # Model, optimizer, and learning rate.
        timers("model and optimizer").start()
        model, optimizer, lr_scheduler = setup_model_and_optimizer(
            neox_args=neox_args, inference=False, get_key_value=True
        )
        timers("model and optimizer").stop()

        print_rank_0('building classification model for {} ...'.format(
            args.task))
        model = Classification(config=config, num_classes=num_classes, num_tokentypes=2,
                               pre_process=pre_process, post_process=post_process)

        return model

    def setup_model_and_optimizer(
        neox_args, inference=True, get_key_value=True, iteration=None
    ):
        """Setup model and optimizer."""
        model = get_model(
            neox_args=neox_args, inference=inference, get_key_value=get_key_value
        )
        print_rank_0(model)
        for name, p in model.named_parameters():
            if p.requires_grad:
                print_rank_0(name, p.shape)
    
        if neox_args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")
            _model_params = None
            _lr_scheduler = None
    
            model, _, _, _ = deepspeed.initialize(
                model=model,
                optimizer=None,
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
                inference=inference,
                iteration=iteration,
            )
            print_rank_0(
                f"Loading checkpoint and starting from iteration {neox_args.iteration}"
            )
        else:
            raise ValueError("Must specify a checkpoint to load to finetune a downstream task")
        return model

    def metrics_func_provider():
        """Provide metrics callback function."""
        def single_dataset_provider(datapath):
            args = get_args()
            tokenizer = get_tokenizer()

            name = name_from_datapath_func(datapath)
            return Dataset(name, [datapath], tokenizer, args.seq_length)
        return accuracy_func_provider(single_dataset_provider)

    def collate_fn_pad(batch, max_len=2048):
        from torch.utils.data._utils.collate import default_collate
        elem = batch[0]
        elem_type = type(elem)
        try:
            out_dict = {}
            for key in elem:
                item_list = [d[key] for d in batch]
                if key in ['input_ids', 'attention_mask']:
                    # Custom behavior: pad this baby!
                    lengths = [sample.size(dim=0) for sample in item_list]
                    padded_item = torch.nn.utils.rnn.pad_sequence(item_list, batch_first=True)
                    if padded_item.shape[1] > max_len:
                        padded_item = padded_item[:, :max_len]
                    out_dict.update({'lengths': lengths})
                    out_dict.update({key: padded_item})
                elif key in ['repair_targets', 'repair_candidates']:
                    out_dict.update({key: item_list})
                else:
                    # Default collate behavior for a dictionary, according to pytorch 2.0.0
                    out_dict.update({key: default_collate(item_list)})
            return elem_type(out_dict)
        except TypeError:
            raise ValueError(f"This mapping type {elem_type} may not support `__init__(iterable)`.")

    """Finetune/evaluate."""
    finetune(train_valid_datasets_provider, model_provider,
             end_of_epoch_callback_provider=metrics_func_provider,
             task_collate_fn=collate_fn_pad)


if __name__=="__main__":

    varmisuse_multi_class(neox_args, num_classes)
