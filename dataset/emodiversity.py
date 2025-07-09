import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import json
import numpy as np
import re
import pandas as pd
from dataset.constants import DEFAULT_EMO_TOKEN
from dataset.tokenizer import preprocess
import random

class Emodiversity(Dataset):
    def __init__(self, json_file, tokenizer, image_paths, reason_path, transform=None, 
                 conv_mode='xywh_shape_color_bbox', obs_n= 6,fast_epoch=False,
                 test=False, use_im_start_end=True, use_im_patches=True, use_tube = False,
                 num_convos=None, use_img = False, use_cap = True, reason = False, use_emo_tokens= False, use_tok_se = False,
                 ):
        super().__init__()
        self.transform = transform
        self.data = pd.read_json(path_or_buf=json_file, lines=True)
        self.obs_n = obs_n


        if fast_epoch:
            self.data = self.data[:32]

        self.tokenizer = tokenizer

        self.conv_mode = conv_mode
        self.test = test
        self.use_im_start_end = use_im_start_end
        self.use_im_patches = use_im_patches

        self.convo_idx = 0
        self.prev_outputs = None
        self.num_convos = num_convos
        self.use_img = use_img
        self.use_cap = use_cap
        self.reason = reason
        self.use_emo_tokens = use_emo_tokens
        self.use_tok_se = use_tok_se
        self.use_tube = use_tube
        self.image_paths = image_paths
        self.select = 4
        self.train_videreason = pd.read_json(path_or_buf=reason_path, lines=True)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        metadata = self.data.iloc[idx]
        gt = metadata['gt']
        captions = metadata['captions']
        videoid = metadata['videoid']
        img_dir = os.path.join(self.image_paths, str(videoid).zfill(5))

        images = os.listdir(img_dir)
        images.sort()
        indices = np.linspace(0, len(images)-1, self.obs_n)
        indices = indices.astype(int)
        images = np.array(images)
        images_sel = images[indices]

        images = []
        for img in images_sel:
            img_path = os.path.join(img_dir, img)
            images.append(Image.open(img_path).convert("RGB"))
        num_frame_tokens = len(images)

        assert self.transform is not None, 'Transform must be provided for multiple images'
        image = torch.stack([self.transform(img) for img in images])
        if self.use_emo_tokens:
            answer = gt[0]
            if answer == 'Empathic Pain':
                answer = 'Empathic_Pain'
            elif answer == 'Aesthetic Appreciation':
                answer = 'Aesthetic_Appreciation'
            elif answer == 'Sexual Desire':
                answer = 'Sexual_Desire'
            elif answer == 'Awe (or Wonder)':
                answer = 'Awe'
            answer = DEFAULT_EMO_TOKEN.replace('emo', answer)
        else:
            answer = gt[0]
        if not self.reason:
            response = answer
        else:
            reason_vid = self.train_videreason[self.train_videreason['videoid'] == videoid]['video'].values[0]
            response = ' '.join([answer, reason_vid])
        if  self.use_img:
            questions = [
                "please determine which emotion label in the video represents.",
                "identify the displayed emotion in the video.",
                "determine the emotional state shown in the video.",
                "please ascertain the specific emotion portrayed in the video.",
                "assess and label the emotion evident in the video.",
                "what emotion is being expressed in the video?",
                "can you classify the emotion depicted in the video as one provided emotion?",
                "identify the emotional tone conveyed in the video.",
                "which of the provided emotions best describes the one shown in the video?",
                "evaluate the emotion shown in the video.",
                "how people might emotionally feel about the content?"
            ]

            emotions = [
                "Awkwardness", "Empathic Pain", "Fear", "Anger", "Sadness", "Relief", "Boredom", "Joy", 
                "Aesthetic Appreciation", "Adoration", "Admiration", "Amusement", "Satisfaction", "Disgust", 
                "Sexual Desire", "Confusion", "Romance", "Craving", "Horror", "Excitement", "Nostalgia", 
                "Awe (or Wonder)", "Interest", "Calmness", "Surprise", "Entrancement", "Anxiety"
            ]
            ground_truth_emotion = answer
            num_choices = random.randint(4, 26)
            remaining_emotions = [emotion for emotion in emotions if emotion != ground_truth_emotion]
            random_choices = random.sample(remaining_emotions, num_choices)
            all_options = random_choices + [ground_truth_emotion]
            random.shuffle(all_options)


            selected_question = random.choice(questions)
            
            convo = [{'from': 'user',
                        'value': "<image>\n. After watching the video, " + selected_question \
                        + " Only provide the one most likely emotion from the provided emotions. ### CHOICE:["+ ','.join(all_options) + "]" },
                    {'from': 'gpt', 'value': response}]
            Choice = f"choose from {len(all_options)} kinds of emotions, specifically, ### CHOICE:["+ ','.join(all_options) + "]" 
            
        else:
            convo = [{'from': 'user',
                        'value': captions },
                    {'from': 'gpt', 'value': response}]
        if self.test:
            convo = convo[:1]

        if self.use_im_patches:
            if self.use_tok_se:
                num_patch_tokens = 98
            elif self.use_tube:
                num_patch_tokens =  self.select * 32
            else:
                if image.dim() == 4:
                    num_patch_tokens = image.shape[2]//14 * image.shape[3]//14
                else:
                    num_patch_tokens = image.shape[1]//14 * image.shape[2]//14
        else:
            num_patch_tokens = 0
        data_dict = preprocess([convo], Choice, self.tokenizer, conv_mode=self.conv_mode, has_image=False,
                               use_im_start_end=self.use_im_start_end, num_patch_tokens=num_patch_tokens,
                               num_frame_tokens=num_frame_tokens, test = self.test)
        data_dict = dict(input_ids=data_dict['input_ids'][0], labels=data_dict['labels'][0])

        data_dict['image'] = image

        data_dict['videoid'] = videoid
        data_dict['gt'] = gt
        data_dict['use_img'] = self.use_img

        return data_dict