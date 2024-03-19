import argparse
from pathlib import Path
from typing import List
from functools import partial
import evaluate
import numpy as np
from pprint import pprint
from transformers import (
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    T5Config,
    Seq2SeqTrainingArguments,
)
from transformers.utils import logging

from ..cl_controllers.selfreplaycontroller import SelfReplayController
from ..trainers.cl_trainer_callback import BaseCLTrainerCallback, EarlyStoppingCallback
import wandb
from ..utils.utils import extract_special_tokens
import random
import pandas as pd

logger = logging.get_logger("transformers")
logging.enable_explicit_format()


def create_directory_tree(args):
    Path(args.model_save_path).mkdir(parents=True, exist_ok=True)
    Path(args.dataset_save_path).mkdir(parents=True, exist_ok=True)
    Path(args.logging_dir).mkdir(parents=True, exist_ok=True)


def init_model_config(cill_tokenizer):
    config = T5Config.from_pretrained(
        "google/t5-v1_1-small", vocab_size=len(cill_tokenizer)
    )
    config.pad_token_id = cill_tokenizer.pad_token_id
    config.decoder_start_token_id = cill_tokenizer.pad_token_id
    config.eos_token_id = cill_tokenizer.eos_token_id
    config.tie_word_embeddings = True
    config.num_layers = 6
    config.num_decoder_layers = 6
    config.num_heads = 8
    model = T5ForConditionalGeneration(config)
    return model


def evaluate_on_test_set(lang_pairs, tokenizer, cl_controller, args):
    import json
    experiences = None

    if args.dataset_name == "iwslt2017":
        from ..utils.data_prep import prepare_experiences_iwslt17
        experiences = prepare_experiences_iwslt17(
            lang_pairs=lang_pairs, args=args, is_bidirectional=True
        )
    elif args.dataset_name == "unpc":
        from ..utils.data_prep import prepare_experiences_unpc
        experiences = prepare_experiences_unpc(
            lang_pairs=lang_pairs, args=args, is_bidirectional=True
        )
    else:
        raise ValueError(f"Dataset name {args.dataset_name} not supported")

    for i, exp in enumerate(experiences):
        eval_res = {}
        best_path = Path(cl_controller.trainer_args.output_dir) / f"exp_{i}-best_model"
        cl_controller.before_experience(i)
        device = cl_controller.trainer.model.device
        save_path = (
            Path(cl_controller.trainer_args.output_dir) / f"exp_{i}_test_res.json"
        )
        model = T5ForConditionalGeneration.from_pretrained(best_path, return_dict=False)
        model.to(device)
        cl_controller.trainer.model = model
        model.eval()
        for k, v in exp.test_datasets.items():
            res = cl_controller.trainer.evaluate(eval_dataset=v)
            eval_res[k] = res
        with open(save_path, "w") as outfile:
           json.dump(eval_res, outfile, indent=4)
        logger.info(f"Scores on test sets for Experience {i}:\n {str(eval_res)}")
    return eval_res


def evaluate_on_test_set_comet(lang_pairs, tokenizer, cl_controller, args):
    import json
    experiences = None

    if args.dataset_name == "iwslt2017":
        from ..utils.data_prep import prepare_experiences_iwslt17
        experiences = prepare_experiences_iwslt17(
            lang_pairs=lang_pairs, args=args, is_bidirectional=True
        )
    elif args.dataset_name == "unpc":
        from ..utils.data_prep import prepare_experiences_unpc
        experiences = prepare_experiences_unpc(
            lang_pairs=lang_pairs, args=args, is_bidirectional=True
        )
    else:
        raise ValueError(f"Dataset name {args.dataset_name} not supported")

    for i, exp in enumerate(experiences):
        eval_res = {}
        best_path = Path(cl_controller.trainer_args.output_dir) / f"exp_{i}-best_model"
        cl_controller.before_experience(i)
        device = cl_controller.trainer.model.device
        save_path = (
            Path(cl_controller.trainer_args.output_dir) / f"exp_{i}_test_res.json"
        )
        model = T5ForConditionalGeneration.from_pretrained(best_path, return_dict=False)
        model.to(device)
        cl_controller.trainer.model = model
        model.eval()
        for k, v in exp.test_datasets.items():
            # todo remove
            src_sents = tokenizer.batch_decode(
                v.select(range(len(v)))["input_ids"], skip_special_tokens=True
            )
            # get labels from test set
            labels = tokenizer.batch_decode(
                v.select(range(len(v)))["labels"], skip_special_tokens=True
            )
            # get predictions
            all_preds = []
            for batch in v:
                generated_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=128,
                    num_beams=5,
                    length_penalty=1.0,
                    use_cache=True,
                )
                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                all_preds.append(preds)


def evaluate_on_val_set(lang_pairs, tokenizer, cl_controller, args):
    import json
    experiences = None

    if args.dataset_name == "iwslt2017":
        from ..utils.data_prep import prepare_experiences_iwslt17
        experiences = prepare_experiences_iwslt17(
            lang_pairs=lang_pairs, args=args, is_bidirectional=True
        )
    elif args.dataset_name == "unpc":
        from ..utils.data_prep import prepare_experiences_unpc
        experiences = prepare_experiences_unpc(
            lang_pairs=lang_pairs, args=args, is_bidirectional=True
        )
    else:
        raise ValueError(f"Dataset name {args.dataset_name} not supported")
    for i, exp in enumerate(experiences):
        eval_res = {}
        best_path = Path(cl_controller.trainer_args.output_dir) / f"exp_{i}-best_model"
        cl_controller.before_experience(i)
        device = cl_controller.trainer.model.device
        save_path = (
            Path(cl_controller.trainer_args.output_dir) / f"exp_{i}_val_res.json"
        )
        model = T5ForConditionalGeneration.from_pretrained(best_path, return_dict=False)
        model.to(device)
        cl_controller.trainer.model = model
        model.eval()
        for k, v in exp.eval_dataset.items():
            res = cl_controller.trainer.evaluate(eval_dataset=v)
            eval_res[k] = res
        with open(save_path, "w") as outfile:
            json.dump(eval_res, outfile, indent=4)
        logger.info(f"Scores on test sets for Experience {i}:\n {str(eval_res)}")
    return eval_res


def replay_storage_to_disk(experiences: int, bench_controller):
    data_p = Path(bench_controller.output_dir) / "data"
    for exp in range(experiences):
        storage_ckp = data_p / f"storage_policy_exp{exp}"
        if storage_ckp.is_file():
            import pickle

            with open(storage_ckp, "rb") as fin:
                storage_p = pickle.load(fin)
                storage_p.save_to_disk(bench_controller, f"{storage_ckp}")


def main(args):
    logging.set_verbosity_info()

    create_directory_tree(args)
    lang_pairs = args.lang_pairs
    special_tokens = extract_special_tokens(lang_pairs)
    tokenizer_path = Path(args.dataset_save_path) / args.tokenizer_savename
    tokenizer = None
    experiences = None
    if Path.is_file(tokenizer_path):
        logger.info("Trying to load tokenizer from file: " + str(tokenizer_path))
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
        logger.info("Tokenizer loaded from file: " + str(tokenizer_path))
    else:
        logger.info("Tokenizer not found in file: " + str(tokenizer_path))
        logger.info("Training tokenizer")
        if args.dataset_name == "iwslt2017":
            from ..utils.utils import train_and_load_tokenizer_iwslt17
            tokenizer = train_and_load_tokenizer_iwslt17(special_tokens, lang_pairs, args)
        elif args.dataset_name == "unpc":
            from ..utils.utils import train_and_load_tokenizer_unpc
            tokenizer = train_and_load_tokenizer_unpc(special_tokens, lang_pairs, args)
        else:
            raise ValueError(f"Dataset name {args.dataset_name} not supported")

    tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "bos_token": "<s>",
        }
    )
    tokenizer.add_tokens(new_tokens=special_tokens, special_tokens=True)
    tokenizer.save_pretrained(str(tokenizer_path))
    logger.info("Tokenizer saved to file: " + str(tokenizer_path))

    model = init_model_config(tokenizer)

    if args.dataset_name == "iwslt2017":
        from ..utils.data_prep import prepare_iwslst17, prepare_experiences_iwslt17
        prepare_iwslst17(lang_pairs=lang_pairs, tokenizer=tokenizer, args=args)
        experiences = prepare_experiences_iwslt17(
            lang_pairs=lang_pairs, args=args, is_bidirectional=args.bidirectional
        )
    elif args.dataset_name == "unpc":
        from ..utils.data_prep import prepare_unpc, prepare_experiences_unpc
        prepare_unpc(lang_pairs=lang_pairs, tokenizer=tokenizer, args=args)
        experiences = prepare_experiences_unpc(
            lang_pairs=lang_pairs, args=args, is_bidirectional=args.bidirectional
        )
    else:
        raise ValueError(f"Dataset name {args.dataset_name} not supported")

    additional_callbacks = []
    if args.early_stopping != 0:
        early_stop = EarlyStoppingCallback(args.early_stopping)
        additional_callbacks.append(early_stop)
    else:
        pass

    rep_to = ["tensorboard", "wandb"] if args.wandb else ["tensorboard"]

    default_args = {
        "output_dir": str(args.model_save_path),
        "num_train_epochs": args.train_epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 5,
        "warmup_steps": 16000,
        "fp16": args.fp16,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "predict_with_generate": True,
        "generation_num_beams": 12,
        "generation_max_length": 128,
        "logging_dir": args.logging_dir,
        "log_level": "info",
        "logging_steps": args.logging_steps,
        "report_to": rep_to,
        "evaluation_strategy": "steps",
        "per_device_eval_batch_size": args.eval_batch_size,
        "eval_accumulation_steps": 1,
        "eval_steps": args.eval_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "bleu",
        "save_strategy": "steps",
        "save_steps": args.save_steps,  # default: 5k
        "save_total_limit": 3,
        "resume_from_checkpoint": True,
        "include_inputs_for_metrics": True,
    }

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(**default_args)

    cl_callback = BaseCLTrainerCallback()

    # Metrics
    bleu_metric = evaluate.load("sacrebleu")
    # comet_metric = evaluate.load("comet")
    ignore_pad_token_for_loss = True
    translations_dataframe = pd.DataFrame(columns=["source", "pred", "target"])

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def log_translation(sources, targets, preds):
        nonlocal translations_dataframe 
        random_sample = random.sample([i for i in range(len(sources))], k=2)
       
        _sources =  [sources[i] for i in random_sample]
        _preds =    [preds[i] for i in random_sample]
        _targets =  [targets[i] for i in random_sample]

        new_transl = pd.DataFrame({"source": _sources, "pred": _preds, "target": _targets})
        #print(new_transl.head())
        to_log = pd.concat([new_transl, translations_dataframe], ignore_index=True)
        wandb_tab = wandb.Table(dataframe=to_log)
        translations_dataframe = to_log 
        wandb.log({"translations": wandb_tab})


    def compute_metrics(eval_preds):
        preds, labels, inputs = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = bleu_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        result = {"bleu": result["score"]}
        # comet metric
        # comet_result = comet_metric.compute(
        #    predictions=decoded_preds, references=decoded_labels
        # )

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if args.wandb:
            log_translation(decoded_inputs, decoded_labels, decoded_preds)
        return result

    cl_controller = SelfReplayController(
        model=model,
        trainer_args=training_args,
        data_collator=data_collator,
        experiences=experiences,
        strategy_callback=cl_callback,
        mem_size=args.replay_memory,
        compute_metrics=compute_metrics,
        additional_callbacks=additional_callbacks,
        selfgen_size=args.num_selfgenerated_samples,
        topk=args.topk,
        tokenizer=tokenizer,
        fastdebug=args.fastdebug,
        start_from_exp=args.start_from_exp,
        round_trip=args.round_trip,
        filtering=args.filtering,
        generation_strategy=args.generation_strategy,
    )

    if args.testset_only:
        evaluate_on_test_set(
            lang_pairs=lang_pairs,
            tokenizer=tokenizer,
            cl_controller=cl_controller,
            args=args,
        )
        return
    if args.validset_only:
        evaluate_on_val_set(
            lang_pairs=lang_pairs,
            tokenizer=tokenizer,
            cl_controller=cl_controller,
            args=args,
        )
        return
    # ---------------------------
    # Start the training phase
    # ---------------------------
    print("=========== Command line arguments: =======================")
    pprint(vars(args))
    print("===========================================================")
    if args.wandb:
        wandb.init()

    cl_controller.continual_train()
    if not args.skip_testeval:
        evaluate_on_test_set(
            lang_pairs=lang_pairs,
            tokenizer=tokenizer,
            cl_controller=cl_controller,
            args=args,
        )
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parse the list of language pairs to train on
    parser.add_argument(
        "--lang_pairs",
        nargs="+",
        help="List of language pairs to train on. Languages has to be specified in ISO 639-1 codes e.g en-fr for English French",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--model_save_path",
        type=str,
        help="path to save models. if the folder does not exist it will be created",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="iwslt2017",
        choices=["iwslt2017", "unpc"],
        help="name of the dataset to use. Default to iwslt2017",
    )
    parser.add_argument(
        "-d",
        "--dataset_save_path",
        type=str,
        help="path to save datasets. if the folder does not exist it will be created",
    )
    parser.add_argument(
        "-l",
        "--logging_dir",
        type=str,
        default="train_logs",
        help="path of logging directory. We log to tensorboard by default",
    )
    parser.add_argument(
        "-e", "--train_epochs", type=int, default=150, help="number of training epochs"
    )
    parser.add_argument(
        "-m",
        "--replay_memory",
        type=int,
        help="number of sentences that will be in the repley buffer",
    )
    parser.add_argument(
        "-p",
        "--pairs_in_experience",
        type=int,
        default=2,
        help="number of language pairs in each experience",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=0,
        help="patience for early stopping. Default 0",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="learning rate to use. default to 5e-4",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        help="log every int steps. Default to 1000",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=5000,
        help="evaluate the model every <int> steps. Default to 5k",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="save the model every <int> steps. Default to 5k",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Per device batch size. Default to 10. Change this value",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=10,
        help="Per device batch size. Default to 10. Change this value",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Num of updates steps to accumulate the gradients for, before performing a"
        " backward/update pass.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="loss",
        help="Metric of the best model" " backward/update pass.",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="use fp16 precision"
    )
    parser.add_argument(
        "--num_selfgenerated_samples",
        type=int,
        default=-1,
        help="Num of samples to generate. By default is equal to the replay memory",
    )
    parser.add_argument(
        "--topk", type=int, default=1000, help="Value of k when samplin"
    )
    parser.add_argument(
        "--testset_only",
        action="store_true",
        default=False,
        help="Run evaluation of best models on test sets and quit",
    )
    parser.add_argument(
        "--validset_only",
        action="store_true",
        default=False,
        help="Run evaluation of best models on test sets and quit",
    )
    parser.add_argument(
        "--fastdebug",
        action="store_true",
        default=False,
        help="if set runs with a subset of train and eval dataset to be faster in debugging",
    )
    parser.add_argument(
        "--start_from_exp",
        type=int,
        default=0,
        help="Start from experience i (0-based indexing) skipping the previous ones",
    )
    parser.add_argument(
        "--round_trip",
        action="store_true",
        default=False,
        help="if set it performs also a backtranslation phase when generating syntetic data",
    )
    parser.add_argument(
        "--filtering",
        type=str,
        default="no_filtering",
        choices=[
            "no_filtering",
            "delete_phunspell",
            "delete_enchant",
            "length",
            "correct_enchant",
            "correct_phunspell",
        ],
        help="Type of filtering after the generation phase",
    )
    parser.add_argument(
        "--generation_strategy",
        type=int,
        default=0,
        help="0: mix of multinomial and beam search. 1: multinomial only 2: beam search only",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
    )
    parser.add_argument("--bidirectional", action="store_false", default=True)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument(
        "--skip_testeval",
        action="store_true",
        default=False,
        help="Skip test evaluation",
    )
    parser.add_argument(
        "--tokenizer_savename",
        type=str,
        default="tok_iwslt17.json",
        help="Name of the tokenizer file",
    )
    args = parser.parse_args()
    main(args)