import pandas as pd
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import os
from transformers import CLIPVisionModel
import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from packaging import version
import warnings
import json
import sklearn
import re



class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)

def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in data:
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in data:
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features

def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    #as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser()
    parser.add_argument('--response_path', type=str,default="",  help="path of the output file")  
    parser.add_argument('--img_dir', type=str, default='', help="path of the image directory")

    args = parser.parse_args()
    img_dir = args.img_dir
    output_file = args.response_path.replace('.json', '_clipscore.json')
    response = pd.read_json(path_or_buf=args.response_path,lines=True)
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    for i in tqdm(range(len(response))):
        scores = []
        video_id = response.iloc[i]['videoid']
        responses = response.iloc[i]['response'][:-1]
        gt = response.iloc[i]['gt']
        label, reason = responses.split("""'''\n""")
        filtered, reason = reason.split('because', 1)
        for item in gt:
            reason = re.sub(re.escape(item), '', reason, flags=re.IGNORECASE)
        
        image_paths = os.listdir(os.path.join(img_dir,str(video_id)))
        for j in range(len(image_paths)):
            image_paths[j] = os.path.join(os.path.join(img_dir,str(video_id)), image_paths[j])
        image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)
        candidates = [reason]* len(image_feats)
        _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)
        for image_id, clipscore in zip(image_paths, per_instance_image_text):
            scores.append(str(float(clipscore)))
        save = {'videoid':str(video_id), 'response': scores}

        with open(output_file, "w") as outfile:
            json_entry = json.dumps(save, ensure_ascii=False)
            outfile.write(json_entry + '\n')


