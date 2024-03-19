
from itertools import chain
from typing import Dict, Sequence, Union

import torch
from torch.utils.data import RandomSampler, DistributedSampler
from torch.utils.data.dataloader import DataLoader
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
_default_collate_mbatches_fn = DataCollatorForSeq2Seq
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
from transformers.utils import logging
from pathlib import Path
from transformers import (
    PreTrainedTokenizerFast,
)

logger = logging.get_logger("transformers")
logging.enable_explicit_format()

class __DistributedHelperPlaceholder:
    is_distributed = False
    world_size = 1
    rank = 0


_DistributedHelper = __DistributedHelperPlaceholder()
def trigger_plugins(strategy, event, **kwargs):
    """Call plugins on a specific callback

    :return:
    """
    for p in strategy.plugins:
        if hasattr(p, event):
            getattr(p, event)(strategy, **kwargs)

def collate_from_data_or_kwargs(data, kwargs):
    if "collate_fn" in kwargs:
        return
    elif hasattr(data, "collate_fn"):
        kwargs["collate_fn"] = data.collate_fn

class GroupBalancedInfiniteDataLoader:
    """Data loader that balances data from multiple datasets emitting an
    infinite stream."""

    def __init__(
        self,
        datasets: Sequence[Union[Dataset, torch.utils.data.IterableDataset]],
        collate_mbatches=_default_collate_mbatches_fn,
        distributed_sampling: bool = True,
        **kwargs
    ):
        """Data loader that balances data from multiple datasets emitting an
        infinite stream.

        Mini-batches emitted by this dataloader are created by collating
        together mini-batches from each group. It may be used to balance data
        among classes, experiences, tasks, and so on.

        :param datasets: an instance of `AvalancheDataset`.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param kwargs: data loader arguments used to instantiate the loader for
            each group separately. See pytorch :class:`DataLoader`.
        """
        self.datasets = datasets
        self.dataloaders = []
        self.collate_mbatches = collate_mbatches

        def dummy_collate(ex):
            return ex

        for data in self.datasets:
            if _DistributedHelper.is_distributed and distributed_sampling:
                seed = torch.randint(
                    0,
                    2 ** 32 - 1 - _DistributedHelper.world_size,
                    (1,),
                    dtype=torch.int64,
                )
                seed += _DistributedHelper.rank
                generator = torch.Generator()
                generator.manual_seed(int(seed))
            else:
                generator = None  # Default
            infinite_sampler = RandomSampler(
                data,
                replacement=True,
                num_samples=10 ** 10,
                generator=generator,
            )
            collate_from_data_or_kwargs(data, kwargs)
            dl = DataLoader(data, sampler=infinite_sampler, collate_fn=dummy_collate, **kwargs)
            self.dataloaders.append(dl)
        self.max_len = 10 ** 10

    def __iter__(self):
        iter_dataloaders = []
        for dl in self.dataloaders:
            iter_dataloaders.append(iter(dl))

        while True:
            mb_curr = []
            for tid, t_loader in enumerate(iter_dataloaders):
                batch = next(t_loader)
                mb_curr.append(batch)
            yield self.collate_mbatches(mb_curr[0])

    def __len__(self):
        return self.max_len
    
def train_tokenizer(data_iter, tokenizer_save_path, special_tokens):
    from tokenizers.implementations import SentencePieceBPETokenizer

    default_special = ["</s>", "<unk>", "<pad>"]
    tokenizer = SentencePieceBPETokenizer()
    template = TemplateProcessing(single="$0 </s>", special_tokens=[("</s>", 0)])
    tokenizer.post_processor = template
    tokenizer.train_from_iterator(
        data_iter,
        vocab_size=32000,
        min_frequency=5,
        show_progress=True,
        special_tokens=default_special + special_tokens,
    )
    print("Tokenizer trained")
    tokenizer.save(str(tokenizer_save_path))

def train_and_load_tokenizer_unpc(special_tokens, lang_pairs, args):
    data_save_dir = Path(args.dataset_save_path)
    tokenizer_save_path = data_save_dir / "tok_unpc.json"

    all_data = []
    logger.info("Loading datasets to train tokenizer")
    for p in lang_pairs:
        logger.info(f"Loading dataset {p} to train tokenizer")
        dataset = load_dataset("un_pc", f"{p}", keep_in_memory=True, num_proc=48)
        all_data.append(dataset["train"].shuffle().select(range(0, 500000)))

    def train_corpus_iter(all_data, batch_size=10000):
        for dset in all_data:
            logger.info(f"Dataset len: {len(dset)}")
            logger.info(f"Language pair: {dset[0]['translation'].keys()}")
            for i in range(0, len(dset), batch_size):
                if i % batch_size == 0:
                    logger.info(f"Batch {i} out of {len(dset)}")
                text = ""
                for translation in dset[i : i + batch_size]["translation"]:
                    for tr in translation.values():
                        text = text + tr + " "
                yield text

    train_tokenizer(
        train_corpus_iter(all_data),
        tokenizer_save_path=tokenizer_save_path,
        special_tokens=special_tokens,
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_save_path))
    tokenizer.add_tokens(new_tokens=special_tokens, special_tokens=True)
    tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "bos_token": "<s>",
        }
    )

    tokenizer.save_pretrained(tokenizer_save_path)
    return tokenizer



def train_and_load_tokenizer_iwslt17(special_tokens, lang_pairs, args):
    data_save_dir = Path(args.dataset_save_path)
    tokenizer_save_path = data_save_dir / args.tokenizer_savename

    all_data = []
    for p in lang_pairs:
        dataset = load_dataset("iwslt2017", f"iwslt2017-{p}")
        all_data.append(dataset["train"])
        all_data.append(dataset["validation"])

    def train_corpus_iter(all_data):
        for i, dataset in enumerate(all_data):
            logger.debug(f"Iterating over {i} dataset started")
            for el in iter(dataset):
                text = ""
                for translation in el.values():
                    if type(translation) == str:
                        continue  # it's the id
                    for tr in translation.values():
                        text = text + tr + " "
                yield text

    train_tokenizer(
        train_corpus_iter(all_data),
        tokenizer_save_path=tokenizer_save_path,
        special_tokens=special_tokens,
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_save_path))
    tokenizer.add_tokens(new_tokens=special_tokens, special_tokens=True)
    tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "bos_token": "<s>",
        }
    )

    tokenizer.save_pretrained(tokenizer_save_path)
    return tokenizer

def extract_special_tokens(lang_pairs):
    special_tokens = []
    for p in lang_pairs:
        src, tgt = p.split("-")
        special_tokens.append(f"<2{src}>")
        special_tokens.append(f"<2{tgt}>")
    special_tokens = list(set(special_tokens))
    return special_tokens