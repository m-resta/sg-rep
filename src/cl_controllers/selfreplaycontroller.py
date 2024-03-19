import random
from transformers.utils import logging
from pathlib import Path
from typing import List
import string

from datasets import Dataset
import torch
from datasets import concatenate_datasets
from .cl_bench_controller import ContinualBenchmarkController
from .storage_policy import ExperienceBalancedBuffer
from ..trainers.clhf_trainer_seq2seq_v2 import Seq2SeqTrainerV2 as Seq2SeqTrainer
from ..trainers.logging_integration import TensorBoardCallback
from tqdm import tqdm
from ..dictionaries import dict_utils

# from reimplementation.dictionaries import dict_utils
import enchant
from enchant.checker import SpellChecker

SEED = 4242
logger = logging.get_logger("transformers")


class SelfReplayController(ContinualBenchmarkController):
    def __init__(
        self,
        model,
        trainer_args,
        data_collator,
        experiences,
        strategy_callback,
        compute_metrics,
        additional_callbacks,
        fastdebug=False,
        start_from_exp=0,
        state_savename="cl_bench_state.json",
        mem_size: int = 200,
        selfgen_size=None,
        round_trip=False,
        filtering="no-filtering",
        generation_strategy=0,
        top_k=32000,
        storage_policy=None,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__(
            model,
            trainer_args,
            data_collator,
            experiences,
            strategy_callback,
            compute_metrics,
            additional_callbacks,
            fastdebug,
            start_from_exp,
            state_savename,
        )
        self.mem_size = mem_size
        self.selfgen_size = selfgen_size
        self.topk = top_k
        self.round_trip = round_trip
        self.filtering = filtering
        self.gen_strategy = generation_strategy

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )
        self.tokenizer = tokenizer

    def before_experience(self, exp_number, **kwargs):
        if self.current_exp == 0:
            datasets = list(self.experiences[exp_number].train_dataset.values())
            if self.fastdebug:
                # Debug purpose
                self.current_train_data = concatenate_datasets(datasets).select(
                    range(15)
                )
                self.current_eval_data = concatenate_datasets(
                    list(self.experiences[exp_number].eval_dataset.values())
                ).select(range(10))
            else:
                # Run normally
                self.current_train_data = concatenate_datasets(datasets)
                self.current_eval_data = concatenate_datasets(
                    list(self.experiences[exp_number].eval_dataset.values())
                )

            # shuffling
            self.current_eval_data = self.current_eval_data.shuffle(seed=SEED)
            self.current_train_data = self.current_train_data.shuffle(seed=SEED)
        else:
            if self.mem_size > 0:
                if len(self.storage_policy.buffer) > 0:
                    datasets = list(
                        self.experiences[exp_number].train_dataset.values()
                    ) + [self.storage_policy.buffer]
                else:
                    logger.warning("STORAGE BUFFER IS EMPTY. Check your replay buffer")
                    datasets = list(self.experiences[exp_number].train_dataset.values())

            else:
                datasets = list(self.experiences[exp_number].train_dataset.values())

            eval_datasets = list(self.experiences[exp_number].eval_dataset.values())

            if self.fastdebug:
                # Debug purpose
                self.current_train_data = concatenate_datasets(datasets).select(
                    range(15)
                )
                self.current_eval_data = concatenate_datasets(eval_datasets).select(
                    range(10)
                )
            else:
                self.current_train_data = concatenate_datasets(datasets)
                self.current_eval_data = concatenate_datasets(eval_datasets)

            # shuffling
            self.current_eval_data = self.current_eval_data.shuffle(seed=SEED)
            self.current_train_data = self.current_train_data.shuffle(seed=SEED)
        logger.debug(f'Current train data length: {len(self.current_train_data)}')
        super().before_experience(exp_number, **kwargs)
        # we reinstantiate the correct trainer

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.trainer_args,
            train_dataset=self.current_train_data,
            eval_dataset=self.current_eval_data,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.strategy_callback.set_trainer(self.trainer)
        self.trainer.add_callback(self.strategy_callback)
        if self.logging_callback is not None:
            self.trainer.remove_callback(TensorBoardCallback)
            self.trainer.add_callback(self.logging_callback)
        # print('current train data len', len(self.current_train_data))

    def _reverse_lang_pair(pair):
        """
        This function takes as input a string of language pair in the following form
        xx-yy and returns yy-xx the reverse direction
        :return:
        """
        src, tgt = pair.split("-")
        return f"{tgt}-{src}"

    def filter_by_punct(self, sentences):
        from string import punctuation

        punc = punctuation + "—|–"
        res = []

        def len_punct(s, punc):
            return len([ch for ch in s if ch in punc])

        for s in sentences:
            len_punc = len_punct(s, punc)
            if len_punc > 0.5 * len(s):
                continue
            res.append(s)
        return res

    def after_experience(self, **kwargs):
        """This function is called after the training of the experience is finished.
        We can use this function to generate new data and store it in the replay buffer
        for future experiences
        """
        logger.info("Generating sentences")
        logger.info(self.experiences[self.current_exp].lang_pairs)
        if self.mem_size > 0:
            if self.current_exp == len(self.experiences) - 1:
                # we skip the generation of replay data
                pass
            else:
                lang_pairs = self.experiences[self.current_exp].lang_pairs
                logger.debug(f"Lang code{lang_pairs}")
                replay_datasets = []
                for pair in lang_pairs:
                    replay_data = self.generate_replay_data(
                        pair, round_trip=self.round_trip, filtering=self.filtering
                    )
                    replay_datasets.append(replay_data)

                concat_data = concatenate_datasets(replay_datasets)
                self.storage_policy.update_selfgenerated(
                    self, new_data=concat_data, **kwargs
                )
                current_buffer = self.storage_policy.buffer
                src, tgt = lang_pairs[0].split("-")
                current_buffer.save_to_disk(self.trainer_args.output_dir + f'/buffer_{self.current_exp}_{src}-{tgt}')

                self.storage_policy.save_to_disk(self, **kwargs)
        else:
            pass
        super().after_experience()

    def _get_ISO_lang_code(self, src_lang):
        """
        Return a ISO lang code corresponding to a 2 letter language codeE e.g. en --> en_US
        """
        iso_codes = {"en": "en_US", "ko": "ko_KR"}
        for l in ["de", "fr", "es", "nl", "it", "ro"]:
            iso_codes[l] = f"{l}_{str.upper(l)}"

        if src_lang not in ["en", "de", "fr", "es", "nl", "it", "ro", "ar", "ko", "ru"]:
            print(f"No spellchecker for {src_lang}")
            return None
        return iso_codes.get(src_lang)

    def filter_generated_sents(self, sentences, src_lang, filtering_type):
        res = []
        lang_code = self._get_ISO_lang_code(src_lang)
        if lang_code is None:
            return sentences
        #print(lang_code)

        if filtering_type == "delete_phunspell":
            try:
                import phunspell

                pspell = phunspell.Phunspell(lang_code)
                no_points = []
                for src_sent in tqdm(sentences, desc="Filtering with phunspell"):
                    translator = str.maketrans(
                        string.punctuation, " " * len(string.punctuation)
                    )
                    no_points = src_sent.translate(translator).split(" ")
                    mispelled = pspell.lookup_list(no_points)
                    if len(mispelled) > 0:
                        continue
                    res.append(src_sent)
            except ImportError as e:
                print(e)
            logger.info(
                f"Removed {len(sentences)- len(res)} sents out of {len(sentences)}"
            )
            return res
        elif filtering_type == "delete_enchant":
            try:
                import enchant
                from enchant.checker import SpellChecker

                chkr = SpellChecker(lang_code)
                d = enchant.request_dict(lang_code)
                for src_sent in tqdm(sentences, desc="Filtering with enchant"):
                    chkr.set_text(src_sent)
                    mispelled = [i.word for i in chkr]
                    if len(mispelled) >= 2:
                        logger.debug(f"Src_sent filtered:{src_sent}\t {mispelled}")
                        continue
                    res.append(src_sent)
            except ImportError as e:
                print(e)
            logger.debug(
                f"Removed {len(sentences)- len(res)} sents out of {len(sentences)}"
            )
            logger.debug(f"Kept sentences are: {res}")
            return res
        elif filtering_type == "length":
            import numpy as np

            sent_len = []
            for s in sentences:
                sent_len.append(len(s))

            avg = round(np.mean(sent_len), 3)
            std = round(np.std(sent_len), 3)
            lower_b = avg - 1.5 * std
            upper_b = avg + 2 * std

            for s in sentences:
                if len(s) <= lower_b or len(s) >= upper_b:
                    continue
                res.append(s)
            logger.info(
                f"Removed {len(sentences)- len(res)} sents out of {len(sentences)}"
            )
            return res
        elif filtering_type == "no_filtering":
            return sentences
        else:
            logger.info(
                f"{filtering_type} not yet implemented. returning unfiltered sentences"
            )
            return sentences

    def _get_word_from_tokenizer(self, lang_code, tokenizer):
        import enchant
        from enchant.checker import SpellChecker

        chkr = SpellChecker(lang_code)
        d = enchant.request_dict(lang_code)
        while True:
            rand_index = random.randint(4, 32002)
            rand_word = tokenizer.decode(rand_index)

            rand_word = rand_word.strip()

            if rand_word.strip() != "" and d.check(rand_word):
                return rand_word
            else:
                best_c = ""
                if rand_word == "":
                    continue
                for j in d.suggest(rand_word):
                    j = j.strip()
                    if j != "" and d.check(j):
                        if len(j) > len(best_c):
                            best_c = j
                if best_c != "" and d.suggest(best_c):
                    return best_c

    def get_rand_words(self, lang_code, text_dict, num=100):
        d = enchant.request_dict(lang_code)
        words = []
        while len(words) < num:
            w = dict_utils.get_random_word(text_dict)
            if d.check(w):
                words.append(w)
        return words


    def generate_replay_data(
        self, lang_pair, round_trip=False, filtering="no_filtering", **kwargs
    ):
        """
        Generate replay data by using the model itself for the lang-pair (src-tgt)
        If round trip is True the Translation (tgt) is used as input again to obtain
        the src lang of the pair
        filtering: - delete_phunspell: delete sentence if errors are detected by pyhunspell
                   - delete_enchant: delete if errors are detected by enchant spellchecker
                   - correct_phunspell: try to correct sentences using phunspell
                   - correct_enchant: try to correct using enchant
                   - length: keep senteces with lenght in chars  avg - 1.5std< l <  > avg + 1.5std
                   - no_filtering: do nothing
        """
        # set up device and number of input vector
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.selfgen_size != -1:
            num_vector = self.selfgen_size
        else:
            num_vector = self.mem_size // 4

        src, tgt = lang_pair.split("-")

        def prefix_tokens(prefix):
            return self.tokenizer(prefix, return_tensors="pt")["input_ids"]

        def generate_prefix_token_batch(num_vector, prefix_toks):
            unpadded = []
            tok_ids = prefix_toks.view(-1)
            # print(prefix_toks.shape)
            for i in range(num_vector):
                # input to datacollatorForSeq2Seq is a list of dict as :
                elem = {
                    "input_ids": tok_ids,
                    "attention_mask": torch.ones_like(tok_ids),
                    "labels": torch.zeros_like(tok_ids),  # fake
                }
                unpadded.append(elem)
            padded = self.data_collator(unpadded)
            return padded

        def create_hf_datasets(
            out_l1_detok: List, out_l2_detok: List, lang1: str, lang2: str
        ) -> Dataset:
            samples = []
            for i, pair in enumerate(zip(out_l1_detok, out_l2_detok)):
                elem = {}
                elem["id"] = i
                elem["translation"] = {lang1: pair[0], lang2: pair[1]}  # src-tgt
                samples.append(elem)
            replay_data = Dataset.from_list(samples)
            return replay_data

        # By using src prefix we will obtain sentences in src language
        prefix_toks = prefix_tokens(f"<2{src}>")
        padded = generate_prefix_token_batch(num_vector, prefix_toks)

        # move to GPU

        padded_ls = torch.chunk(padded["input_ids"], 96)
        iter_step = 100

        self.model.to(device)
        sentences = []
        filtered_sents = []
        logger.info("Starting sampling")
        while len(filtered_sents) < num_vector:
            for inp in tqdm(padded_ls[:16]):
                inp = inp.to(device)
                outs = self.model.generate(
                    inp,
                    do_sample=True,
                    top_k=self.topk,
                    max_new_tokens=self.trainer_args.generation_max_length,
                    temperature=0.93
                )
                inp = inp.to("cpu")
                outs_dec = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
                outs = outs.to("cpu")
                if len(sentences) != 0:
                    for e in outs_dec:
                        sentences.append(e)
                    # sentences = sentences + outs_dec
                else:
                    sentences = outs_dec
            logger.debug(f"Generated {len(sentences)} sentences by topk sampling")
            filtered = self.filter_generated_sents(
                sentences=sentences, src_lang=src, filtering_type=filtering
            )
            filtered_sents += filtered
            logger.info(
                f"Filtered sents len: {len(filtered_sents)}, newly added sentences: {len(filtered)}"
            )
            filtered_sents = list(dict.fromkeys(filtered_sents))  # remove dups
            logger.info(f"Filtered sents len: {len(filtered_sents)}")
            sentences = []

        # Prepend tgt token and translate
        prefix = f"<2{tgt}>"
        prefixed_sents = []
        for s in filtered_sents:
            s = prefix + " " + s
            prefixed_sents.append(s)

        translations = []
        logger.info("Starting translation")
        for idx in tqdm(range(0, len(prefixed_sents), iter_step)):
            e = prefixed_sents[idx : idx + iter_step]
            inp = self.tokenizer(
                e, max_length=128, truncation=True, padding=True, return_tensors="pt"
            )
            inp.pop("token_type_ids", None)
            inp.to(device)
            outs = self.model.generate(**inp, max_new_tokens=128)
            inp = inp.to("cpu")
            outs = outs.to("cpu")
            trans_dec = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            if len(translations) != 0:
                for e in trans_dec:
                    translations.append(e)
            else:
                translations = trans_dec

        replay_data = []

        # Another pass to backtranslate to src
        if round_trip:
            prefix = f"<2{src}>"
            prefixed_trans = []
            for s in translations:
                s = prefix + " " + s
                prefixed_trans.append(s)

            bck_translations = []
            logger.info("Starting backtranslation")
            for idx in tqdm(range(0, len(prefixed_trans), iter_step)):
                inp = self.tokenizer(
                    e,
                    max_length=128,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                inp.pop("token_type_ids", None)
                inp.to(device)
                outs = self.model.generate(**inp, max_new_tokens=128)
                inp = inp.to("cpu")
                outs.to("cpu")
                bcktrans_dec = self.tokenizer.batch_decode(
                    outs, skip_special_tokens=True
                )
                if len(bck_translations) != 0:
                    bck_translations = bck_translations + bcktrans_dec
                else:
                    bck_translations = bcktrans_dec

            # we create the dataset
            replay_data = create_hf_datasets(bck_translations, translations, src, tgt)

        else:
            replay_data = create_hf_datasets(filtered_sents, translations, src, tgt)

        tokenizer = self.tokenizer
        prefix = f"<2{tgt}>"
        source_lang = src
        target_lang = tgt

        replay_data.save_to_disk(self.trainer_args.output_dir + f'/replay_data_{src}-{tgt}')

        def _preprocess_function(examples):
            inputs = [
                prefix + example[source_lang] for example in examples["translation"]
            ]
            targets = [example[target_lang] for example in examples["translation"]]
            model_inputs = tokenizer(inputs, max_length=128, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=128, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        replay_data = replay_data.map(_preprocess_function, batched=True)
        if "translation" in replay_data.column_names:
            replay_data = replay_data.remove_columns(["translation"])
        replay_data.set_format(type="torch")

        logger.info(
            f"=== End of generation for {lang_pair}. Round trip is {round_trip} ==="
        )
        return replay_data

    def load_bench_state(self):
        super().load_bench_state()
        data_p = Path(self.output_dir) / "data"
        data_p.mkdir(parents=True, exist_ok=True)
        # path of the pickled file
        exp = self.current_exp 
        logger.info(f"Loading storage of experience {exp}")
        storage_ckp = data_p / f"storage_policy_exp{exp}"
        if storage_ckp.is_file():
            import pickle

            with open(storage_ckp, "rb") as fin:
                storage_p = pickle.load(fin)
                self.storage_policy = storage_p
            logger.info(f"Experience storage for exp {exp} loaded correctly")
        else:
            logger.warning(f"Cannot load storage for exp {exp}")

    def save_bench_state(self):
        super().save_bench_state()
        import pickle

        data_p = Path(self.output_dir) / "data"
        data_p.mkdir(parents=True, exist_ok=True)
        # path of the pickled file
        exp = self.current_exp
        storage_ckp = data_p / f"storage_policy_exp{exp}"
        with open(storage_ckp, "wb") as fout:
            pickle.dump(self.storage_policy, fout)
        self.storage_policy.save_to_disk(
            self, prefix=f"replay_storage_{self.current_exp}"
        )
