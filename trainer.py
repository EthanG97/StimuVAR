import copy
import json
import os
import math
import numpy as np
from transformers import (
    TrainerCallback,
    TrainingArguments,
)
from torch import nn
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer, is_datasets_available, PreTrainedModel
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch.distributed as dist
from transformers.trainer_utils import EvalPrediction
import torch
import re
from transformers.data.data_collator import DataCollator

class LLMCallback(TrainerCallback):
    "A callback that output infomation and do some operators"

    
    def output_log(self, args: TrainingArguments, state: TrainerState):
        def loss_log(data):
            try:
                loss_ = data["loss"]
                learning_rate_ = data["learning_rate"]
                step_ = data["step"]
                loss_log_str = f"step: {step_:<8} || learning_rate: {learning_rate_:<25} || loss: {loss_:<10}"
            except:
                loss_log_str = json.dumps(data)
            return loss_log_str

        output_file = os.path.join(args.output_dir, "trainer.log")
        log_history = map(loss_log, state.log_history)
        with open(output_file, "w") as f:
            for line in log_history:
                f.write(line + "\n")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # TODO: support deepspeed zero3 save extra weights not all llm weights
        if args.lora_save_strategy == 'steps' and state.global_step%args.save_steps == 0:
            self.output_log(args, state)
            peft_str = "peft"
            model_ = kwargs["model"]
            save_number = str(state.global_step)
            if (
                args.lora
                and peft_str in str(type(model_))
                and not is_deepspeed_zero3_enabled()
            ):
                # if model is peft-based, save the extra weights, zero3 not supprot
                epoch = "steps_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.save_pretrained(output_dir)
            if (
                args.lora
                and peft_str in str(type(model_))
                and args.tune_mm_mlp_adapter
                and not is_deepspeed_zero3_enabled()
            ):
                epoch = "steps_" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.base_model.model.save_pretrained(output_dir)
                
        return super().on_step_end(args, state, control, **kwargs)
    
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # TODO: support deepspeed zero3 save extra weights not all llm weights
        if args.lora_save_strategy == 'epoch' :
            self.output_log(args, state)
            peft_str = "peft"
            model_ = kwargs["model"]
            save_number = str(state.global_step)
            if (
                args.apply_lora
                and peft_str in str(type(model_))
                and not is_deepspeed_zero3_enabled()
            ):
                # if model is peft-based, save the extra weights, zero3 not supprot
                epoch = "epoch-" + save_number
                output_dir = os.path.join(args.output_dir, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.merge_and_unload()
                model_.base_model.model.save_pretrained(output_dir)
            # if (
            #     args.apply_lora
            #     and peft_str in str(type(model_))
            #     and not args.freeze_llm_proj
            #     and not is_deepspeed_zero3_enabled()
            # ):
            #     epoch = "epoch-" + save_number
            #     output_dir = os.path.join(args.output_dir, epoch)
            #     os.makedirs(output_dir, exist_ok=True)
            #     model_.base_model.model.save_pretrained(output_dir)
        return super().on_epoch_end(args, state, control, **kwargs)

    def merge_files(self, prediction_file_name):
        from pathlib import Path
        old_files = list(Path(prediction_file_name).parent.glob('*.worker*'))
        prediction_file_name = '.'.join(str(old_files[0]).split('.')[:-1])
        metrics = []
        with open(prediction_file_name, "w") as writer:
            for file_name in old_files:
                with open(file_name, "r") as reader:
                    for line in reader:
                        metrics.append(torch.tensor(self.compute_dev_metric(json.loads(line))))
                        writer.write(f"{line}")
            metrics = torch.mean(torch.stack(metrics, dim = 1 ),dim=1)
            bleu, rouge1, rouge2, rougeL, rougeLsum, bert_score_precision, bert_score_recall, bert_score_f1 = metrics.numpy().tolist()
        with open(prediction_file_name.replace('.jsonl','_metric.txt'), "w") as metric_writer:
            metric_str = json.dumps([bleu, rouge1, rouge2, rougeL, rougeLsum, bert_score_precision, bert_score_recall, bert_score_f1])
            metric_writer.write(metric_str)
        for file in old_files:
            os.system(f"rm {file}")
        return dict(bleu = bleu, 
                    rouge1 = rouge1, 
                    rouge2 = rouge2, 
                    rougeL = rougeL, 
                    rougeLsum = rougeLsum, 
                    bert_score_precision = bert_score_precision, 
                    bert_score_recall = bert_score_recall, 
                    bert_score_f1 = bert_score_recall)

    def compute_dev_metric(self,d):
        bert_score = d['bert_score']
        metric_score = d['metric_score']
        bleu = sum([metric['bleu'] for metric in metric_score])/len(metric_score) if len(metric_score)!= 0 else 0
        rouge1 = sum([metric['rouge1'] for metric in metric_score])/len(metric_score)if len(metric_score)!= 0 else 0
        rouge2 = sum([metric['rouge2'] for metric in metric_score])/len(metric_score)if len(metric_score)!= 0 else 0
        rougeL = sum([metric['rougeL'] for metric in metric_score])/len(metric_score)if len(metric_score)!= 0 else 0
        rougeLsum = sum([metric['rougeLsum'] for metric in metric_score])/len(metric_score)if len(metric_score)!= 0 else 0
        bert_score_precision = sum(bert_score['precision'])/len(bert_score['precision']) if len(bert_score['precision'])!= 0 else 0
        bert_score_recall = sum(bert_score['recall'])/len(bert_score['recall']) if len(bert_score['recall'])!= 0 else 0
        bert_score_f1 = sum(bert_score['precision'])/len(bert_score['precision']) if len(bert_score['precision'])!= 0 else 0
        return bleu, rouge1, rouge2, rougeL, rougeLsum, bert_score_precision, bert_score_recall, bert_score_f1 

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if dist.get_rank() == 0:
            # due to multiprocess, just do evaluation in rank 0
            metric = self.merge_files(args.prediction_file_name)
            kwargs['metrics'].update(metric)
            control.should_log = True
        else:
            control.should_log = False
        return super().on_evaluate(args, state, control, **kwargs)

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        if dist.get_rank() == 0:
                # due to multiprocess, just do evaluation in rank 0
            self.merge_files(args.prediction_file_name)

        return super().on_predict(args, state, control, metrics, **kwargs)
    

