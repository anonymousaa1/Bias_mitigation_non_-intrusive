#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import matplotlib.pyplot as plt

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import BertModel, BertConfig
import copy
import json
import tqdm as tqdm
names = locals()

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EncoderDecoderModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from fair_metrics import FairMetrics
from clustering import Clustering

import os
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# model = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_gigaword")
# tok = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_gigaword")
#
# article = """australian shares closed down #.# percent monday
# following a weak lead from the united states and
# lower commodity prices , dealers said ."""
#
# enc = tok(article, return_tensors="pt")
# hidden_states = model.encoder(**enc, return_dict=True)
#
# # perturb the last_hidden_state
# hidden_states.last_hidden_state = perturb(hidden_states.last_hidden_state)
#
# gen_ids = model.generate(input_ids=None, encoder_outputs=hidden_states, attention_mask=enc["attention_mask"])
# tok.batch_decode(gen_ids)


#
# bert_version = "bert-base-cased"
# bert_base_cased = BertModel.from_pretrained(bert_version)  # Instantiate model using the trained weights
# config = BertConfig.from_pretrained(bert_version)
# model = BertModel(config=config)  # Randomly initialize model, with the same size as the trained model
#
# # add these two lines
# model.embeddings = bert_base_cased.embeddings
# model.pooler = bert_base_cased.pooler
#
# layers_to_replace = [1, 2, 3, 8]
# for layer in layers_to_replace:
#     model.base_model.encoder.layer[layer] = bert_base_cased.base_model.encoder.layer[layer]

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 3 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        if sys.argv[2].endswith(".json"):
            f = open(sys.argv[2])
            parameters = f.read()
            param_data = json.loads(parameters)
            pert_size = param_data["pert_size"]
            if 'eval_dataset' in param_data:
            # if param_data.has_key('eval_dataset'):
                eval_dataset_file = "" + param_data["eval_dataset"]
            else:
                eval_dataset_file = None
            if 'do_value_file' in param_data:
                do_value_file = "" + param_data["do_value_file"]
            else:
                do_value_file = None
            do_layer = param_data["do_layer"]
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("-->arguments:")
    print(model_args)
    print(data_args)
    print(training_args)



    # data_args.max_predict_samples = 1000
    # training_args.per_device_eval_batch_size = 2

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("-->device", device)
    n_gpu = torch.cuda.device_count()
    print("-->n_gpu", n_gpu)
    training_args._n_gpu = n_gpu

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        # is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        is_regression = raw_datasets["train"].features["label"].dtype in ["float8", "float16"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # config.output_hidden_states = True
    # config.output_attentions = True

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Training ***")
        print("-->logger training")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        print("-->logger evaluate")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)


    if training_args.do_predict:
        logger.info("*** Predict ***")
        print("-->logger Predict")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])
        """
        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            print("-->type", type(predict_dataset))
            label = predict_dataset['label']
            print("-->predict_dataset", predict_dataset)
            # print("input_ids", predict_dataset['input_ids'])
            # print("token_type_ids", predict_dataset['token_type_ids'])
            # print("attention_mask", predict_dataset['attention_mask'])
            print(type(predict_dataset))
            predict_dataset = predict_dataset.remove_columns("label")
            print("num of predict_dataset", predict_dataset.num_rows)

            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            # print("-->predictions", predictions)
            # print("-->label", label)

            results = metric.compute(references=label, predictions=predictions)
            print("-->metric results:", results)
            accuracy_metric = load_metric('accuracy')
            print("-->accuracy:", accuracy_metric.compute(references=label, predictions=predictions))
            # print("-->original label", label)
            # print("-->original predictions", predictions)
            same = 0
            for i in range(0, len(label)):
                if label[i] == predictions[i]:
                    same += 1

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
        """

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    predict_datasets = [predict_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        predict_datasets.append(raw_datasets["test_mismatched"])

    predict_dataset = raw_datasets["test"]
    label = predict_dataset['label']
    predict_dataset = predict_dataset.remove_columns("label")
    print("-->predict_dataset", predict_dataset)
    print("number of data:", predict_dataset.num_rows)
    print("-->text", predict_dataset['text'][1], len(predict_dataset['text'][1]))

    raw_texts = predict_dataset['text']
    label = label

    # # eval_dataset_file = "eval_dataset/bias_madlibs_39k.csv"
    # # eval_dataset_file = "eval_dataset/bias_madlibs_10k.csv"

    # evaluate/cluster/causality analysis in 5k dataset
    """
    import csv
    with open(eval_dataset_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    # print("-->rows", rows)
    raw_texts = []
    label = []
    for i in range(1, len(rows)):
        raw_texts.append(rows[i][0])
        label.append(int(rows[i][1]))

    raw_texts = raw_texts
    label = label
    """

    # print("-->raw_texts", raw_texts[0])
    # print("-->lable", label)

    # import pandas as pd
    # import numpy as pn
    # df = pd.read_csv(eval_dataset_file, sep=',')
    # # 读取第一行
    # raw_texts = df.iloc[:0]
    # print("-->raw_texts", raw_texts)
    # # 读取第一列
    # label = df.iloc[:1]
    # print("-->label", label)


    do_output_hidden_states = True
    if do_output_hidden_states:
        from causality_analysis import Causality
        # model summary
        print("-->model", model)
        print("-->trainer", trainer)
        # print("-->get_input_embeddings", model.get_input_embeddings)
        # print("-->get_output_embeddings", model.get_output_embeddings)

        # predict_dataset = predict_dataset.remove_columns("label")
        # outputs = model(predict_dataset)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        config.output_hidden_states = True
        config.output_attentions = True

        """test sub_model prediction accuracy """
        """
        predictions = []
        for text in raw_texts:
            inputs = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True,
                               return_tensors="pt")
            outputs = model(**inputs)[0]
            logits = outputs.logits.tolist()
            predictions.append(logits[0].index(max(logits[0])))
            # class_output, label_output = sub_model.forward_orig(inputs)
            # predictions.append(label_output)
        print("-->predictions", predictions)
        accuracy_metric = load_metric('accuracy')
        print("-->accuracy:", accuracy_metric.compute(references=label, predictions=predictions))
        # For first 50 5k raw_texts, accuracy = 0.62
        # for first 50 original raw_texts, accuracy = 0.78
        """

        # do_layer = 12
        # do_neuron = [0]
        # do_neuron = list(range(0, 128))
        do_neuron = list(range(0, 2))
        # do_neuron = list(range(0, 2))

        # do_value_file = "hidden_states_5k/cluster_centers_" + str(1) + ".txt"  # str(do_layer)
        # do_values = []
        # with open(do_value_file) as f:
        #     for line in f.readlines():
        #         do_values.append(eval(line.strip('\n')))


        # # # Clustering pooler_output
        # all_pooler_output = []
        # i = 0
        # for text in raw_texts:
        #     print(i)
        #     inputs = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True,
        #                        return_tensors="pt")
        #     pooler_output = model(**inputs)[1].tolist()[0]
        #     all_pooler_output.append(pooler_output)
        #     i += 1
        # # file_name = "hidden_states/pooled_outputs.txt"
        # # with open(file_name, "w") as f:
        # #     f.writelines(str(all_pooler_output))
        # # f.close()
        #
        # # file_name = "hidden_states/pooled_outputs.txt"
        # # with open(file_name, "r") as f:
        # #     all_pooler_output = f.read()
        #
        # """
        # # # select neuron 0
        # # print(type(all_pooler_output))
        # # print("-->shape", np.array(all_pooler_output).shape)
        # # one_neuron_pooler_output = np.array(all_pooler_output)[:, 0].tolist()
        # # print("-->length", len(one_neuron_pooler_output))
        # # clustering = Clustering(do_layer=13, do_neuron=0)
        # # one_neuron_pooler_output = [[output] for output in one_neuron_pooler_output]
        # # clustering.clustering_elbow(one_neuron_pooler_output)
        # # clustering.clustering_gap_statistic(one_neuron_pooler_output)
        #
        # # # select neuron 200
        # # one_neuron_pooler_output = np.array(all_pooler_output)[:, 200].tolist()
        # # clustering = Clustering(do_layer=13, do_neuron=200)
        # # one_neuron_pooler_output = [[output] for output in one_neuron_pooler_output]
        # # clustering.clustering_elbow(one_neuron_pooler_output)
        # # clustering.clustering_gap_statistic(one_neuron_pooler_output)
        # #
        # # # select neuron 500
        # # one_neuron_pooler_output = np.array(all_pooler_output)[:, 500].tolist()
        # # clustering = Clustering(do_layer=13, do_neuron=500)
        # # one_neuron_pooler_output = [[output] for output in one_neuron_pooler_output]
        # # clustering.clustering_elbow(one_neuron_pooler_output)
        # # clustering.clustering_gap_statistic(one_neuron_pooler_output)
        # #
        # # # select neuron 700
        # # one_neuron_pooler_output = np.array(all_pooler_output)[:, 700].tolist()
        # # clustering = Clustering(do_layer=13, do_neuron=700)
        # # one_neuron_pooler_output = [[output] for output in one_neuron_pooler_output]
        # # clustering.clustering_elbow(one_neuron_pooler_output)
        # # clustering.clustering_gap_statistic(one_neuron_pooler_output)
        # """
        #
        # all_cluster_centers = []
        # print(list(range(0, 767)))
        # for i in range(0, 768):
        #     one_neuron_pooler_output = np.array(all_pooler_output)[:, i].tolist()
        #     clustering = Clustering(do_layer=13, do_neuron=i)
        #     one_neuron_pooler_output = [[output] for output in one_neuron_pooler_output]
        #     labels, centers, inertia = clustering.clustering(one_neuron_pooler_output)
        #     all_cluster_centers.append(centers)
        #
        # center_file = "hidden_states/cluster_centers(pooled_output).txt"
        # with open(center_file, "w") as f:
        #     f.writelines(str(all_cluster_centers))
        # f.close()

        with open(do_value_file) as f:
            values = eval(f.readline())
        do_values = []
        for one_set_value in values:
            one_set_do_value = [v[0] for v in one_set_value]
            do_values.append(one_set_do_value)

        all_do_neuron = list(range(701, 768))
        # all_do_neuron = list(range(0, 1))
        do_neuron = list(range(0, 1))

        # do_value_file = "hidden_states_5k/cluster_centers_" + str(1) + ".txt"  # str(do_layer)
        # do_values = []
        # with open(do_value_file) as f:
        #     for line in f.readlines():
        #         do_values.append(eval(line.strip('\n')))

        for do_neuron in all_do_neuron:
            do_neuron = [do_neuron]
            # Causality Analysis
            causality = Causality(model=model, do_layer=do_layer, do_neurons=do_neuron, do_values=do_values,
                                  pert_size=pert_size)
            typical_term = "gay"
            term_list = ["lesbian", "gay", "bisexual", "transgender", "trans", "queer", "lgbt", "lgbtq", "homosexual",
                         "straight", "heterosexual", "male", "female"]
            all_ie = causality.get_yfair_all_values(raw_texts, tokenizer, padding, max_seq_length, label, model,
                                                    typical_term, term_list)
            print("-->all ie for neurons {} in the pooler output {}".format(do_neuron, all_ie))

            save_file = './results/pooler_results/neuron_' + str(do_neuron[0]) + ".txt"
            with open(save_file, 'w') as f:
                f.write(str(all_ie))

        # outputs = model(**inputs)
        # print("-->outputs", outputs)
        # print("-->keys", outputs.keys(), type(outputs))
        # print("-->logits:", outputs.logits)
        # ## cls向量
        # # print("-->pooler_output", outputs.pooler_output)
        # ## hidden_states，包括13层，第一层即索引0是输入embedding向量，后面1-12索引是每层的输出向量
        # hidden_states = outputs.hidden_states
        # print("-->hidden_states", hidden_states)
        # print("type:", type(hidden_states), type(hidden_states[0]))
        # print("-->hidden_states", np.array(hidden_states).shape)
        # embedding_output = hidden_states[0]
        # print("-->embedding_output", embedding_output.size())
        #
        # encoder_output = model.bert.encoder(embedding_output)
        # print("-->encoder_output(base)", encoder_output)
        # print("num of model.bert.encoder.layer", len(model.bert.encoder.layer))
        #
        # """
        # Test sub-network
        # """
        # base_model = model
        #
        # inputs = tokenizer(raw_texts[0], padding=padding, max_length=max_seq_length, truncation=True,
        #                    return_tensors="pt")
        # print("-->inputs", inputs)
        # attention_mask = inputs.attention_mask
        # print("-->attention_mask", attention_mask)
        # base_outputs = base_model(**inputs)
        # base_hidden_states = base_outputs.hidden_states
        # print("-->shape", np.array(base_hidden_states).shape)
        # base_layer_9_output = base_hidden_states[9]
        # print("-->layer_9_output(base)", base_layer_9_output)
        # print("-->layer_last_output(base)", base_hidden_states[-1])
        #
        # # print("model.bert", base_model.bert)
        # print("num of model.bert.encoder.layer", len(base_model.bert.encoder.layer))
        # oldModuleList = base_model.bert.encoder.layer
        # newModuleList = nn.ModuleList()
        # num_layers_to_keep = [10, 11, 12]
        # for i in num_layers_to_keep:
        #     newModuleList.append(oldModuleList[i-1])
        # # for i in range(0, len(num_layers_to_keep)):
        # #     newModuleList.append(oldModuleList[i])
        # model = copy.deepcopy(base_model)
        # model.bert.encoder.layer = newModuleList
        # model.bert.embeddings = None
        #
        # print("-->success")
        #
        # new_model = model.bert.encoder
        # outputs = new_model(base_layer_9_output)
        # print("-->outputs", outputs)
        # encoder_outputs = outputs.last_hidden_state
        # print("-->encoder_outputs", encoder_outputs)
        #
        # pooler = model.bert.pooler
        # pooler_outputs = pooler(encoder_outputs)
        # print("-->pooler outputs", pooler_outputs)
        # dropout = model.dropout
        # dropout_outputs = dropout(pooler_outputs)
        # print("-->dropout outputs", dropout_outputs)
        # classifier = model.classifier
        # class_outputs = classifier(dropout_outputs)
        # print("-->classifier outputs", class_outputs)



# bert_version = "bert-base-cased"
# bert_base_cased = BertModel.from_pretrained(bert_version)  # Instantiate model using the trained weights
# config = BertConfig.from_pretrained(bert_version)
# config.num_hidden_layers = 4
# model = BertModel.from_pretrained(bert_version, config=config)  # auto skip unused layers
#
# for param_name in model.state_dict():
#     sub_param, full_param = model.state_dict()[param_name], bert_base_cased.state_dict()[param_name] # type: torch.Tensor, torch.Tensor
#     assert (sub_param.cpu().numpy() == full_param.cpu().numpy()).all(), param_name
#
# print("-->success")
# for n, p in model.named_parameters():
#     print(n)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

