import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from typing import Dict, Sequence

from dataset.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_PATCH_TOKEN, \
    DEFAULT_PAD_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from dataset import conversation as conversation_lib

def get_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return tokenizer

def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [tokenizer.image_token_id] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess(
    sources,
    Choice,
    tokenizer: PreTrainedTokenizer,
    conv_mode: str = "llama",
    num_patch_tokens: int = 0,
    num_frame_tokens: int = 1,
    use_im_start_end: bool = False,
    test: bool = False,
    **kwargs,
) -> Dict:
    conv = conversation_lib.conv_templates[conv_mode].copy()
    roles = {"user": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if use_im_start_end and DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, 
                                                              DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, 
                                                            DEFAULT_IMAGE_TOKEN * num_frame_tokens + DEFAULT_IM_PATCH_TOKEN * num_patch_tokens)
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    conversation = conversations[0]
    location = conversation.find('when they watch the video and')
    conversations = [conversation[:location+len('when they watch the video and') ] + Choice + conversation[location+len('when they watch the video and') :]]

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=2048,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    # Mask targets
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(conv.sep)
            if len(parts) != 2:
                break
            parts[0] += conv.sep

            round_len = len(tokenizer(rou).input_ids)  
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2 

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        
        if not test:
            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def prompt(
    sources,
    conv_mode: str = "xywh_shape_color_bbox",
    num_patch_tokens: int = 0,
    num_frame_tokens: int = 1,
    use_im_start_end: bool = False,
) -> Dict:
    conv = conversation_lib.conv_templates[conv_mode].copy()
    roles = {"user": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if use_im_start_end and DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, 
                                                              DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, 
                                                            DEFAULT_IMAGE_TOKEN * num_frame_tokens + DEFAULT_IM_PATCH_TOKEN * num_patch_tokens)
            conv.append_message(role, sentence["value"])
        
    return conv.get_prompt()