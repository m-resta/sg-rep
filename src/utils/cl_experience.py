from dataclasses import dataclass
from typing import List, Dict

import datasets


@dataclass
class CL_Experience:
    train_dataset: Dict[str, datasets.Dataset]
    eval_dataset: Dict[str, datasets.Dataset]
    test_datasets: Dict[str, datasets.Dataset]
    lang_pairs: List[str]
    is_bidirectional: bool = True

    def __init__(self, train_dataset, eval_dataset, test_datasets, lang_pairs, is_bidirectional):
        self.lang_pairs = lang_pairs
        self.is_bidirectional = is_bidirectional
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_datasets = test_datasets


    def _update_pair_list(self, lang_pairs):
        pairs = lang_pairs.copy()
        for p in lang_pairs:
            src, tgt = p.split('-')
            reversed = f'{tgt}-{src}'
            pairs.append(reversed)
        return pairs
