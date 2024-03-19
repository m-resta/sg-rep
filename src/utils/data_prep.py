import os
from ..utils.cl_experience import CL_Experience
from pathlib import Path
from copy import copy

import datasets
from datasets import load_dataset
from transformers.utils import logging

logger = logging.get_logger("transformers")
logging.enable_explicit_format()

def _reverse_lang_pair(pair):
    """
    This function takes as input a string of language pair in the following form
    xx-yy and returns yy-xx the reverse direction
    :return:
    """
    src, tgt = pair.split("-")
    return f"{tgt}-{src}"


def prepare_unpc(lang_pairs, tokenizer, args):
    dataset_save_dir = Path(args.dataset_save_path)
    rev_lang_pairs = []
    for p in lang_pairs:
        rev_lang_pairs.append(_reverse_lang_pair(p))
    all_pairs = lang_pairs + rev_lang_pairs

    logger.info(f"Preparing UNPC dataset")
    for p in all_pairs:
        src, tgt = p.split("-")
        prefix = f"<2{tgt}>"
        train_path = dataset_save_dir / f"unpc_{p}_train"
        valid_path = dataset_save_dir / f"unpc_{p}_valid"
        test_path = dataset_save_dir / f"unpc_{p}_test"

        def preprocess_function(examples):
            inputs = [
                prefix + " " + example[src] for example in examples["translation"]
            ]
            targets = [example[tgt] for example in examples["translation"]]
            model_inputs = tokenizer(inputs, max_length=128, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True)
                if labels is None:
                    logger.warning("None labels identified", examples)
                    return
                model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        if os.path.isdir(train_path):
            logger.info(f"Loading UNPC dataset from {train_path}")
            train_data = datasets.load_from_disk(str(train_path))
        else:
            logger.info("UNPC not found on disk")
            logger.info("Preprocessing and saving UNPC dataset")
            try:
                train_data = load_dataset("un_pc", f"{p}")["train"]
                test_valid = train_data.train_test_split(test_size=4000, shuffle=True)
                valid_data = test_valid["test"].select(range(0, 2000))
                test_data = test_valid["test"].select(range(2000, 4000))
                train_data = test_valid["train"]
                train_data = train_data.map(
                    preprocess_function, batched=True, keep_in_memory=True, num_proc=64
                )
                valid_data = valid_data.map(
                    preprocess_function, batched=True, keep_in_memory=True, num_proc=64
                )
                test_data = test_data.map(
                    preprocess_function, batched=True, keep_in_memory=True, num_proc=64
                )
            except ValueError as e:
                logger.info(f"ValueError: {e}")
                logger.info(f'Using reverse pair "{_reverse_lang_pair(p)}"')
                rev_p = _reverse_lang_pair(p)
                train_data = load_dataset("un_pc", f"{rev_p}")["train"]
                test_valid = train_data.train_test_split(test_size=4000, shuffle=True)
                valid_data = test_valid["test"].select(range(0, 2000))
                test_data = test_valid["test"].select(range(2000, 4000))
                train_data = test_valid["train"]

                logger.info(f"Src: {src}, Tgt: {tgt}")
                logger.info(f"Prefix: {prefix}")

                def rev_preprocess_function(examples):
                    inputs = [
                        prefix + " " + example[src]
                        for example in examples["translation"]
                    ]
                    targets = [example[tgt] for example in examples["translation"]]
                    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(targets, max_length=128, truncation=True)
                        if labels is None:
                            logger.info("None labels identified", examples)
                            return
                        model_inputs["labels"] = labels["input_ids"]
                    return model_inputs

                def dummy_preprocess_func(examples):
                    inputs = [
                        prefix + " " + example[src]
                        for example in examples["translation"]
                    ]
                    logger.info(f"Inputs: {inputs}")
                    targets = [example[tgt] for example in examples["translation"]]
                    logger.info(f"Targets: {targets}")
                    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(targets, max_length=128, truncation=True)
                        if labels is None:
                            logger.info("None labels identified", examples)
                            return
                        model_inputs["labels"] = labels["input_ids"]
                    return model_inputs

                debug_data = train_data.select(range(0, 10))
                debug_data = debug_data.map(
                    dummy_preprocess_func, batched=True, keep_in_memory=True
                )

                train_data = train_data.map(
                    rev_preprocess_function,
                    batched=True,
                    keep_in_memory=True,
                    num_proc=48,
                )
                valid_data = valid_data.map(
                    rev_preprocess_function,
                    batched=True,
                    keep_in_memory=True,
                    num_proc=48,
                )
                test_data = test_data.map(
                    rev_preprocess_function,
                    batched=True,
                    keep_in_memory=True,
                    num_proc=48,
                )
            finally:
                if "translation" in train_data.column_names:
                    train_data = train_data.remove_columns(["translation"])
                if "translation" in valid_data.column_names:
                    valid_data = valid_data.remove_columns(["translation"])
                if "translation" in test_data.column_names:
                    test_data = test_data.remove_columns(["translation"])

                train_data.save_to_disk(train_path)
                valid_data.save_to_disk(valid_path)
                test_data.save_to_disk(test_path)


def prepare_experiences_unpc(lang_pairs, args, is_bidirectional=False):
    logger.info(f"Preparing experiences for {lang_pairs} with UNPC dataset")
    dataset_save_dir = Path(args.dataset_save_path)
    data_c = 0
    train_dict = {}
    eval_dict = {}
    test_dict = {}
    experiences = []
    # we gather all the test datasets
    for p in lang_pairs:
        test_dict[p] = datasets.load_from_disk(str(dataset_save_dir / f"unpc_{p}_test"))
        if is_bidirectional:
            tgt, src = p.split("-")
            rev_pair = f"{src}-{tgt}"
            test_dict[rev_pair] = datasets.load_from_disk(
                str(dataset_save_dir / f"unpc_{rev_pair}_test")
            )

    # now we construct the actual experiences
    logger.info(f"Preparing experiences")
    for p in lang_pairs:
        train_dict[p] = datasets.load_from_disk(str(dataset_save_dir / f"unpc_{p}_train"))
        eval_dict[p] = datasets.load_from_disk(str(dataset_save_dir / f"unpc_{p}_valid"))
        data_c += 1

        if is_bidirectional:
            tgt, src = p.split("-")
            rev_pair = f"{src}-{tgt}"
            train_dict[rev_pair] = (
                datasets.load_from_disk(str(dataset_save_dir / f"unpc_{rev_pair}_train"))
                .shuffle(seed=42)
                .select(range(0, len(train_dict[p]) // 2))
            )
            # we halves the size of the p train dataset
            train_dict[p] = (
                train_dict[p].shuffle(seed=42).select(range(0, len(train_dict[p]) // 2))
            )

            print(f"pair: {p}", train_dict[p].shuffle(seed=42).select(range(0, 1)))
            print(
                f"pair: {rev_pair}",
                train_dict[rev_pair].shuffle(seed=42).select(range(0, 1)),
            )

            eval_dict[rev_pair] = datasets.load_from_disk(
                str(dataset_save_dir / f"unpc_{rev_pair}_valid")
            )
            data_c += 1

        if data_c == args.pairs_in_experience:
            if args.fastdebug:
                train_dict = {k: train_dict[k].select(range(0, 100)) for k in train_dict}
                eval_dict = {k: eval_dict[k].select(range(0, 100)) for k in eval_dict}
                test_dict = {k: test_dict[k].select(range(0, 100)) for k in test_dict}

            exp = CL_Experience(
                train_dataset=train_dict,
                eval_dataset=eval_dict,
                test_datasets=test_dict,
                is_bidirectional=is_bidirectional,
                lang_pairs=list(train_dict.keys()),
            )

            experiences.append(exp)
            train_dict = {}
            eval_dict = {}
            data_c = 0
            exp = []

    logger.info(f"Total experiences created: {len(experiences)}")
    return experiences


def prepare_iwslst17(lang_pairs, tokenizer, args):
    logger.info(f"Preprocessing IWSLT17 for {lang_pairs}")
    dataset_save_dir = Path(args.dataset_save_path)
    rev_lang_pairs = []
    for p in lang_pairs:
        rev_lang_pairs.append(_reverse_lang_pair(p))
    all_pairs = lang_pairs + rev_lang_pairs

    for p in all_pairs:
        src, tgt = p.split("-")
        prefix = f"<2{tgt}>"
        train_path = dataset_save_dir / f"iwslt17_{p}_train"
        valid_path = dataset_save_dir / f"iwslt17_{p}_valid"
        test_path = dataset_save_dir / f"iwslt17_{p}_test"

        def preprocess_function(examples):
            inputs = [
                prefix + " " + example[src] for example in examples["translation"]
            ]
            targets = [example[tgt] for example in examples["translation"]]
            model_inputs = tokenizer(inputs, max_length=128, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True)
                if labels is None:
                    logger.info("None labels identified", examples)
                    return
                model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        if os.path.isdir(train_path):
            train_data = datasets.load_from_disk(train_path)
        else:
            train_data = load_dataset("iwslt2017", f"iwslt2017-{p}")["train"]
            train_data = train_data.map(preprocess_function, batched=True)
            if "translation" in train_data.column_names:
                train_data = train_data.remove_columns(["translation"])
            train_data.save_to_disk(train_path)
        # validation
        if os.path.isdir(valid_path):
            valid_data = datasets.load_from_disk(valid_path)
        else:
            valid_data = load_dataset("iwslt2017", f"iwslt2017-{p}")["validation"]
            valid_data = valid_data.map(preprocess_function, batched=True)
            if "translation" in valid_data.column_names:
                valid_data = valid_data.remove_columns(["translation"])
            valid_data.save_to_disk(valid_path)
        # Test
        if os.path.isdir(test_path):
            test_data = datasets.load_from_disk(valid_path)
        else:
            test_data = load_dataset("iwslt2017", f"iwslt2017-{p}")["test"]
            test_data = test_data.map(preprocess_function, batched=True)
            if "translation" in test_data.column_names:
                test_data = test_data.remove_columns(["translation"])
            test_data.save_to_disk(test_path)


def prepare_experiences_iwslt17(lang_pairs, args, is_bidirectional=False):
    logger.info(f"Preparing experiences for {lang_pairs} with IWSLT17 dataset") 
    dataset_save_dir = Path(args.dataset_save_path)
    data_c = 0
    train_dict = {}
    eval_dict = {}
    test_dict = {}
    experiences = []
    # we gather all the test datasets
    for p in lang_pairs:
        test_dict[p] = datasets.load_from_disk(dataset_save_dir / f"iwslt17_{p}_test")
        if is_bidirectional:
            tgt, src = p.split("-")
            rev_pair = f"{src}-{tgt}"
            test_dict[rev_pair] = datasets.load_from_disk(
                dataset_save_dir / f"iwslt17_{rev_pair}_test"
            )

    # now we construct the actual experiences
    for p in lang_pairs:
        train_dict[p] = datasets.load_from_disk(dataset_save_dir / f"iwslt17_{p}_train")
        eval_dict[p] = datasets.load_from_disk(dataset_save_dir / f"iwslt17_{p}_valid")
        data_c += 1

        if is_bidirectional:
            tgt, src = p.split("-")
            rev_pair = f"{src}-{tgt}"
            train_dict[rev_pair] = datasets.load_from_disk(
                dataset_save_dir / f"iwslt17_{rev_pair}_train"
            )
            eval_dict[rev_pair] = datasets.load_from_disk(
                dataset_save_dir / f"iwslt17_{rev_pair}_valid"
            )
            data_c += 1

        if data_c == args.pairs_in_experience:
            exp = CL_Experience(
                train_dataset=train_dict,
                eval_dataset=eval_dict,
                test_datasets=test_dict,
                is_bidirectional=is_bidirectional,
                lang_pairs=list(train_dict.keys()),
            )

            experiences.append(exp)
            train_dict = {}
            eval_dict = {}
            data_c = 0
            exp = []

    logger.info(f"Total experiences created: {len(experiences)}")
    return experiences
