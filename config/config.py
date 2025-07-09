from transformers import Seq2SeqTrainingArguments

import os
from dataclasses import dataclass, field
import time

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(default=os.path.join('work_dirs', time.strftime("%Y%m%d-%H%M%S")))
    per_device_train_batch_size: int = field(default=4)
    dataloader_num_workers: int = field(default=8)
    num_train_epochs: int = field(default=1)
    remove_unused_columns: bool = field(default=False)
    cache_dir: str = field(default='cache')
    model_max_length: int = field(default=2048)
    deepspeed: str = field(default=None)
    save_strategy: str = field(default='epoch')
    auto_find_batch_size: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=False)

    project_name: str = field(default='vlm')
    run_name: str = field(default='single_triangle')
    json_file_train: str = field(default='')
    json_file_val: str = field(default='')
    orig_image_shape: tuple[int, int] = field(default=(224, 224))
    resize_image_shape: tuple[int, int] = field(default=(224, 224))

    llm_path: str = field(default='')
    vit_path: str = field(default='')
    image_paths: str = field(default='')
    reason_path: str = field(default='')

    freeze_backbone: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_llm_proj: bool = field(default=False)

    fast_epoch: bool = field(default=False)
    conv_mode: str = field(default='')
    num_convos: int = field(default=None)
    use_im_start_end: bool = field(default=True)
    use_im_patches: bool = field(default=True)
    use_bbox_tokens: bool = field(default=False)
    use_delta_transformer: bool = field(default=False)
    use_img: bool = field(default=False)
    use_cap: bool = field(default=True)
    reason: bool = field(default=False)
    apply_lora: bool = field(default=False)
    use_emo_tokens: bool = field(default=False)
    tune_lm_head: bool = field(default=False)
    use_atp: bool = field(default=False)
    use_tok_se: bool = field(default=False)
    use_lan: bool = field(default=False)
    use_tube: bool = field(default=False)
    evaluation_strategy: str = field(default='no')
    lora_save_strategy: str = field(default='epoch')
    


@dataclass
class InferenceArguments(TrainingArguments):
    model_path: str = field(default='work_dirs/20240325-155244')
    test_batch_size: int = field(default=1)

    make_pdf: bool = field(default=False)
    output_pdf_name: str = field(default='output.pdf')
    num_test_samples: int = field(default=None)
    compute_metrics: bool = field(default=False)
    save_metrics: bool = field(default=False)

    is_scaled: bool = field(default=False)
    is_temporal: bool = field(default=False)
    is_path: bool = field(default=False)
    response_path: str = field(default='')
    use_img: bool = field(default=False)
    use_cap: bool = field(default=True)
    reason: bool = field(default=False)
    use_atp: bool = field(default=False)

