import torch
from torchvision import transforms
import transformers
from transformers import Trainer, TrainerCallback, GenerationConfig
from transformers.integrations import WandbCallback
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


import os
import argparse
import yaml
import wandb
from tqdm import tqdm
from peft import get_peft_model, LoraConfig,TaskType

from dataset.emodiversity import Emodiversity
from dataset.collator import DataCollatorForSupervisedDataset
from dataset.tokenizer import get_tokenizer
from model.vit_llama import ViTLlamaForCausalLM
from config.config import TrainingArguments
from trainer import LLMCallback
import numpy as np
import random


os.environ['DS_SKIP_CUDA_CHECK']="1"

# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    para = []
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            para.append(_)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    print(para)


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        "A CallBack to log samples a wandb.Table during training"
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    
    def samples_table(self, examples):
        "Create a wandb.Table to store the generations"
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        return records_table
        
    def on_evaluate(self, args, state, control,  **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})


def main(args):

    # accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    accelerator = Accelerator()

    args.dataloader_num_workers = accelerator.num_processes

    if 'lora' in args.llm_path:
        model = ViTLlamaForCausalLM.from_pretrained(args.llm_path.replace('checkpoint','epoch'),
                                        cache_dir=args.cache_dir)
        tokenizer = get_tokenizer(args.llm_path)
        tokenizer.model_max_length = args.model_max_length


    else:
        tokenizer = get_tokenizer(args.llm_path)
        tokenizer.model_max_length = args.model_max_length

        model = ViTLlamaForCausalLM.from_pretrained(args.llm_path,
                                                    cache_dir=args.cache_dir)
    transform = transforms.Compose([
        transforms.Resize((args.resize_image_shape[0], args.resize_image_shape[1])),
        transforms.ToTensor(),
    ])

    model.init_vision(args.vit_path, tokenizer, use_im_start_end=args.use_im_start_end, use_tube=args.use_tube,
                      use_im_patches=args.use_im_patches, use_bbox_tokens=args.use_bbox_tokens,use_atp=args.use_atp,use_tok_se = args.use_tok_se, use_lan=args.use_lan,
                      use_delta_transformer=args.use_delta_transformer, use_img=args.use_img, use_emo_tokens=args.use_emo_tokens)

    train_dataset = Emodiversity(json_file=args.json_file_train,
                                 tokenizer=tokenizer, 
                                 transform=transform, 
                                 conv_mode=args.conv_mode,
                                 fast_epoch=args.fast_epoch,
                                 use_im_start_end=args.use_im_start_end,
                                 use_im_patches=args.use_im_patches,
                                 num_convos=args.num_convos,
                                 use_img = args.use_img,
                                 use_cap=args.use_cap,
                                 reason = args.reason, 
                                 use_tok_se=args.use_tok_se,
                                 use_tube = args.use_tube,
                                 use_emo_tokens=args.use_emo_tokens,
                                 image_paths=args.image_paths,
                                 reason_path = args.reason_path)
    val_dataset = Emodiversity(json_file=args.json_file_val,
                               tokenizer=tokenizer, 
                               transform=transform, 
                               conv_mode=args.conv_mode,
                               fast_epoch=args.fast_epoch,
                               use_im_start_end=args.use_im_start_end,
                               use_im_patches=args.use_im_patches,
                               num_convos=args.num_convos,
                               use_img = args.use_img,
                               use_cap=args.use_cap,
                               use_tok_se=args.use_tok_se,
                               use_tube = args.use_tube,
                               use_emo_tokens=args.use_emo_tokens,
                               image_paths=args.image_paths,
                               reason_path = args.reason_path,
                               test=False)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    if args.freeze_llm:
        model.get_model().layers.requires_grad_(False)
        model.get_model().norm.requires_grad_(False)
    # assert args.freeze_llm, 'Only support freezing llm for now'

    if args.use_img:
        if args.freeze_backbone:
            model.get_model().vit.requires_grad_(False)
        # assert args.freeze_backbone, 'Only support freezing backbone for now'

        if not args.freeze_llm_proj:
            for p in model.get_model().llm_proj.parameters():
                p.requires_grad = True
        else:
            for p in model.get_model().llm_proj.parameters():
                p.requires_grad = False
    for p in model.get_input_embeddings().parameters():
        p.requires_grad = True
    if args.tune_lm_head:
        for p in model.get_output_embeddings().parameters():
            p.requires_grad = True
    else:
        for p in model.get_output_embeddings().parameters():
            p.requires_grad = False

    if args.apply_lora:
        target_modules=['model.layers.'+str(i)+'.'+ k for i in range(40) for k in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj","mlp.down_proj","mlp.up_proj"]]
        if args.use_img:
            modules_to_save = ['embed_tokens','llm_proj']  #'patch_net_tube',
        else:
            modules_to_save = ['embed_tokens']
        # modules_to_save = []
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target_modules, modules_to_save = modules_to_save,
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)

    # else:

    if args.apply_lora:
        callback_class =  LLMCallback
    else:
        callback_class =  TrainerCallback

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[callback_class],
        # callbacks=[TrainerCallback],
    )

    # Start training
    trainer.train()

    # Save model, tokenizer and state
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_state()

    accelerator.wait_for_everyone()

if __name__ == '__main__':
    init_seeds(0)
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config', type=str, default='/cis/home/yguo/honda/emollava/config/trans_delta/emo_vis_trans.yaml')
    config = cli_parser.parse_args().config

    args_parser = transformers.HfArgumentParser(TrainingArguments)
    args = args_parser.parse_yaml_file(config, allow_extra_keys=True)[0]
    os.environ['WANDB_PROJECT'] = args.project_name

    args.output_dir = os.path.join('work_dirs', args.run_name, args.output_dir.split('/')[-1])

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config to file
    with open(os.path.join(args.output_dir, 'training_args.yaml'), 'w') as f:
        yaml.dump(args.to_dict(), f, default_flow_style=False)

    main(args)

