import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image


class SeparatorStyle(Enum):
    """Different separator style."""
    LLAMA_3 = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.LLAMA_2
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages

        if self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if i == 0: message = wrap_sys(self.system) + message
                if i % 2 == 0:
                    message = wrap_inst(message)
                    ret += '<s>' + message
                else:
                    ret += " " + message + " " + '</s>'
                    if i != len(messages) - 1:
                        ret += '\n'
            ret = ret.lstrip('<s>')
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            wrap_header = lambda header, msg: f'<|start_header_id|>{header}<|end_header_id|>\n\n{msg}<|eot_id|>'
            wrap_next = lambda header: f'<|start_header_id|>{header}<|end_header_id|>\n\n' 
            ret = '<|begin_of_text|>'
            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if i == 0: ret += wrap_header('system', self.system)
                if i % 2 == 0:
                    ret += wrap_header('user', message)
                    if i == len(messages) - 1:
                        ret += wrap_next('assistant') + '\n'
                else:
                    ret += wrap_header('assistant', message)
                    if i == len(messages) - 1:
                        ret += '<|end_of_text|>'
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret


        # if len(messages) > 0 and type(messages[0][1]) is tuple:
        #     messages = self.messages.copy()
        #     init_role, init_msg = messages[0].copy()
        #     init_msg = init_msg[0].replace("<image>", "").strip()
        #     if 'mmtag' in self.version:
        #         messages[0] = (init_role, init_msg)
        #         messages.insert(0, (self.roles[0], "<Image><image></Image>"))
        #         messages.insert(1, (self.roles[1], "Received."))
        #     else:
        #         messages[0] = (init_role, "<image>\n" + init_msg)


        # if self.sep_style == SeparatorStyle.LLAMA_2:
        #     wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
        #     wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
        #     ret = ""

        #     for i, (role, message) in enumerate(messages):
        #         if i == 0:
        #             assert message, "first message should not be none"
        #             assert role == self.roles[0], "first message should come from user"
        #         if message:
        #             if type(message) is tuple:
        #                 message, _, _ = message
        #             if i == 0: message = wrap_sys(self.system) + message
        #             if i % 2 == 0:
        #                 message = wrap_inst(message)
        #                 ret += self.sep + message
        #             else:
        #                 ret += " " + message + " " + self.sep2
        #         else:
        #             ret += ""
        #     ret = ret.lstrip(self.sep)



    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }
    


conv_llama = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)


conv_llama_inf = Conversation(

    system="You are a large language assistant trained to understand caption of the video that the user provides "
            "and assist the user to estimate the emotion of viewers when they watch the video and choose from "
            "27 kinds of emotions, specifically, ### CHOICE:[Awkwardness, Empathic Pain, Fear, Anger, Sadness, Relief, Boredom, Joy, Aesthetic Appreciation, \
         Adoration,  Admiration, Amusement, Satisfaction, Disgust, Sexual Desire, Confusion, Romance, Craving, Horror, Excitement, Nostalgia, Awe (or Wonder), \
         Interest, Calmness, Surprise, Entrancement, Anxiety], I will provide a video description and you need to estimate the most likely one emotion of the viewers when they watch the video."
        "carefully and explain your answers in detail." + """please return the answer in this way: \
         '''
         The viewer feels xxxx", because xxxxx
         '''
            """,
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)


conv_llama_tok = Conversation(

    system="You are a large language assistant trained to understand caption of the video that the user provides "
            "and assist the user to estimate the emotion of viewers when they watch the video and choose from "
            "27 kinds of emotions, specifically, ### CHOICE:[ <Awkwardness>, <Empathic_Pain>, <Fear>, <Anger>, <Sadness>, <Relief>, <Boredom>, <Joy>, <Aesthetic_Appreciation>, \
         <Adoration>,  <Admiration>, <Amusement>, <Satisfaction>, <Disgust>, <Sexual_Desire>, <Confusion>, <Romance>, <Craving>, <Horror>, <Excitement>, <Nostalgia>, <Awe>, \
         <Interest>, <Calmness>, <Surprise>, <Entrancement>, <Anxiety> ], I will provide a video description and you need to estimate the most likely one emotion of the viewers when they watch the video."
        "carefully and explain your answers in detail." + """please return the answer in this way: \
         '''
         The viewer feels xxxx", because xxxxx
         '''
            """,
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llama_emo = Conversation(

    system="You are a large language assistant trained to understand the video that the user provides "
            "and assist the user to estimate the emotion of viewers when they watch the video and choose from "
            "provided emotions. I will provide a video and you need to estimate the most likely one emotion of the viewers when they watch the video.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="[/INST] ",
    sep2="</s>",
)
conv_llama_emo_nochoice_reason = Conversation(

    system="You are a large language assistant trained to understand the video that the user provides "
            "and assist the user to estimate the emotion of viewers when they watch the video and choose from "
            "provided emotions. I will provide a video and you need to estimate the most likely one emotion of the viewers when they watch the video."
        "carefully and explain your answers in detail." ,
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="[/INST] ",
    sep2="</s>",
)

default_conversation = conv_llama
conv_templates = {
    "llama":conv_llama,
    "emo_llama":conv_llama_emo,
    "emo_llama_nochoice_reason":conv_llama_emo_nochoice_reason,
}