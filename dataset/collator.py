import torch
from transformers import PreTrainedTokenizer

from dataclasses import dataclass
from typing import Dict, Sequence

from dataset.constants import IGNORE_INDEX


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        if hasattr(self, 'device'):
            input_ids = input_ids.to(self.device)

        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        if hasattr(self, 'device'):
            labels = labels.to(self.device)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'].half() for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            if hasattr(self, 'device'):
                batch['images'] = batch['images'].to(self.device)
        if 'bbox' in instances[0]:
            bboxes = [instance['bbox'] for instance in instances]
            if all(x is not None and x.shape == bboxes[0].shape for x in bboxes):
                batch['bboxes'] = torch.stack(bboxes)
            else:
                batch['bboxes'] = bboxes
            # if hasattr(self, 'device'):
            #     batch['bboxes'] = batch['bboxes'].to(self.device)
        if 'conversations' in instances[0]:
            batch['conversations'] = [instance['conversations'] for instance in instances]
        batch['videoid'] = [instance['videoid'] for instance in instances]
        batch['gt'] = [instance['gt'] for instance in instances]
        batch['use_img'] = [instance['use_img'] for instance in instances]
        return batch
    
    def to(self, device):
        self.device = device
        return self