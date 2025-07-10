import torch
from torchvision import transforms
from transformers import StoppingCriteria
from PIL import Image
import cv2
import transformers
from model.vit_llama import ViTLlamaForCausalLM
from dataset.tokenizer import get_tokenizer
from dataset.conversation import conv_templates
from config.config import InferenceArguments
from helpers.split_frame import opticalsplit_frame_to_frames
from dataset.tokenizer import preprocess_demo


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



def run_demo(video_path, prompt, model_path, config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    args_parser = transformers.HfArgumentParser(InferenceArguments)
    args = args_parser.parse_yaml_file(config_path)[0]

    # Load model and tokenizer
    model = ViTLlamaForCausalLM.from_pretrained(model_path.replace('checkpoint','epoch'),
        cache_dir=args.cache_dir).to(device)
    tokenizer = get_tokenizer(model_path)
    tokenizer.model_max_length = args.model_max_length
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((args.resize_image_shape[0], args.resize_image_shape[1])),
        transforms.ToTensor(),
    ])
    frames = opticalsplit_frame_to_frames(video_path, 6)
    image_tensors = []
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        tensor = transform(pil_image)
        image_tensors.append(tensor)

# Stack into a 4D tensor: (batch_size, C, H, W)
    image_tensor_batch = torch.stack(image_tensors).unsqueeze(0).to(device)
    
    conv = conv_templates[args.conv_mode]
    sep = conv.sep2
    # Build conversation
    conv =  [{'from': 'user',
                        'value': prompt}]
    num_patch_tokens =  4 * 32
    
    data_dict = preprocess_demo([conv], tokenizer, conv_mode=args.conv_mode, has_image=False,
                               use_im_start_end=args.use_im_start_end, num_patch_tokens=num_patch_tokens,
                               num_frame_tokens=6, test = True)
    input_ids = data_dict['input_ids'].to(device)
    # Generate
    stopping_criteria = KeywordsStoppingCriteria([sep], tokenizer, input_ids)
    breakpoint()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor_batch,
            use_img=True,
            max_new_tokens=args.model_max_length,
            stopping_criteria=[stopping_criteria],
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print("Response:", decoded_output.strip())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    args = parser.parse_args()
    model_path = '/your/model/path/checkpoint-150000'  # Replace with your model path
    config = 'config/Inference.yaml'  # Path to your Inference config file
    prompt = "<image>\n. After watching the video, how people might emotionally feel about the content. "\
                        + "Only provide the one most likely emotion from the provided emotions. ### CHOICE:[Awkwardness, Empathic Pain, Fear, Anger, Sadness, Relief, Boredom, Joy, Aesthetic Appreciation, \
         Adoration,  Admiration, Amusement, Satisfaction, Disgust, Sexual Desire, Confusion, Romance, Craving, Horror, Excitement, Nostalgia, Awe (or Wonder), \
         Interest, Calmness, Surprise, Entrancement, Anxiety]"
    run_demo(args.video, prompt, model_path, config)
