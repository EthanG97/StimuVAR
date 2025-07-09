import torch
from torchvision import transforms
import transformers
from transformers import StoppingCriteria
from accelerate import Accelerator
import os
import cv2
import json
import random
import argparse
import re
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from dataset.conversation import conv_templates, SeparatorStyle

from model.vit_llama import ViTLlamaForCausalLM
from dataset.tokenizer import get_tokenizer
from dataset.emodiversity import Emodiversity
from dataset.collator import DataCollatorForSupervisedDataset
from config.config import InferenceArguments
from peft import PeftConfig, PeftModel
import pandas as pd

os.environ['DS_SKIP_CUDA_CHECK']="1"

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False
    

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def inference(args):
    init_seeds(0)

    device = torch.device('cuda')
    model_loc = args.model_path
    if 'lora' in model_loc:
        model = ViTLlamaForCausalLM.from_pretrained(model_loc.replace('checkpoint','epoch'),
                                    cache_dir=args.cache_dir)
        tokenizer = get_tokenizer(model_loc)
        print("load end")
    else:
        model = ViTLlamaForCausalLM.from_pretrained(model_loc,
                                                cache_dir=args.cache_dir)

        tokenizer = get_tokenizer(model_loc)
    tokenizer.model_max_length = args.model_max_length
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((args.resize_image_shape[0], args.resize_image_shape[1])),
        transforms.ToTensor(),
    ])

    

    eval_dataset = Emodiversity(json_file=args.json_file_val,
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
                               test=True)
    
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer).to(device)
    model.eval()

    conv = conv_templates[args.conv_mode]
    sep = conv.sep2


    bs = args.test_batch_size
    if args.num_test_samples is None:
        args.num_test_samples = len(eval_dataset)
    tem = 0.1 
    output_path = 'results/{}/'.format(args.model_path.split('/')[-3])
    output_file = '{}_{}.jsonl'.format(args.model_path.split('/')[-1], tem)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(output_path+output_file):
        filepath = output_path+output_file
        content = pd.read_json(path_or_buf=filepath, lines=True)
        start = len(content)
    else:
        start = 0
    for i in tqdm(range(start, args.num_test_samples, bs)):
        assert args.num_convos == 1 or bs == 1, 'num_convos > 1 only supported for batch_size=1'


        for convo_idx in range(args.num_convos):
            if i+bs > args.num_test_samples:
                end = args.num_test_samples
            else:
                end = i+bs
            data_dicts = [eval_dataset[l] for l in range(i, end)]

            data = collator(data_dicts)

            keywords = [sep]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, data['input_ids'])

            with torch.inference_mode():
                output_ids= model.generate(data['input_ids'], attention_mask=data['attention_mask'], images=data['images'],use_img = data['use_img'], max_new_tokens=args.model_max_length, 
                                            stopping_criteria=[stopping_criteria], 
                                            temperature=tem, do_sample=True,
                                            pad_token_id=tokenizer.eos_token_id)
            input_token_len = data['input_ids'].shape[1]
            n_diff_input_output = (data['input_ids'] != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)
            for i,output in enumerate(outputs):
                save = {'videoid':str(data['videoid'][i]), 'response': output, 'gt': data['gt'][i]}
                with open(output_path+output_file, "a") as outfile:
                    json_entry = json.dumps(save, ensure_ascii=False)
                    outfile.write(json_entry + '\n')



if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('--config', type=str, default='/cis/home/yguo/honda/emollava/config/random_QC/reason/rqro_reason_tube.yaml')
    config = cli_parser.parse_args().config

    args_parser = transformers.HfArgumentParser(InferenceArguments)
    args = args_parser.parse_yaml_file(config)[0]

    metrics = inference(args)


                                              

