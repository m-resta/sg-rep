from pathlib import Path
import random
from typing import List

from datasets import concatenate_datasets, Dataset
from transformers.utils import logging
from ..cl_controllers.cl_bench_controller import ContinualBenchmarkController
from ..cl_controllers.storage_policy import ExperienceBalancedBuffer
from ..trainers.clhf_trainer_seq2seq_v2 import Seq2SeqTrainerV2 as Seq2SeqTrainer
from ..trainers.logging_integration import TensorBoardCallback
from ..utils.utils import GroupBalancedInfiniteDataLoader
from torch.utils.data import random_split

logger = logging.get_logger("transformers")

SEED = 4242


class AGEMController(ContinualBenchmarkController):
    def __init__(
        self,
        model,
        trainer_args,
        data_collator,
        experiences,
        strategy_callback,
        compute_metrics,
        additional_callbacks,
        patterns_per_exp,
        sample_size,
        fastdebug=False,
        start_from_exp=0,
        state_savename="cl_bench_state.json",
        storage_policy=None,
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
        self.mem_size = 100

        self.patterns_per_experience = int(patterns_per_exp)
        self.sample_size = int(sample_size)

        self.buffers: List[Dataset] = []  # one Dataset for each experience.
        self.buffer_dataloader = None
        self.buffer_dliter = None

        self.reference_gradients = None
        self.memory_x, self.memory_y = None, None

    def before_experience(self, exp_number, **kwargs):
        if self.current_exp == 0:
            datasets = list(self.experiences[exp_number].train_dataset.values())
            if self.fastdebug:
                # Debug purpose
                self.current_train_data = concatenate_datasets(datasets).select(
                    range(20000)
                )
                self.current_eval_data = concatenate_datasets(
                    list(self.experiences[exp_number].eval_dataset.values())
                ).select(range(1000))
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
            datasets = list(self.experiences[exp_number].train_dataset.values())

            eval_datasets = list(self.experiences[exp_number].eval_dataset.values())

            if self.fastdebug:
                # Debug purpose
                self.current_train_data = concatenate_datasets(datasets).select(
                    range(20000)
                )
                self.current_eval_data = concatenate_datasets(eval_datasets).select(
                    range(1000)
                )
            else:
                self.current_train_data = concatenate_datasets(datasets)
                self.current_eval_data = concatenate_datasets(eval_datasets)

            # shuffling
            self.current_eval_data = self.current_eval_data.shuffle(seed=SEED)
            self.current_train_data = self.current_train_data.shuffle(seed=SEED)
        super().before_experience(exp_number)
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
        self.strategy_callback.set_cl_controller(self)
        for c in self.additional_callbacks:
            self.trainer.add_callback(c)
        if self.logging_callback is not None:
            self.trainer.remove_callback(TensorBoardCallback)
            self.trainer.add_callback(self.logging_callback)
        logger.debug(f'Current train data length {len(self.current_train_data)}')

    def after_experience(self, **kwargs):
        self.update_memory(self.current_train_data, **kwargs)
        super().after_experience()

    def update_memory(self, dataset, num_workers=0, **kwargs):
        """
        Update replay memory with patterns from current experience.
        """
        if num_workers > 0:
            logger.warning(
                "Num workers > 0 is known to cause heavy" "slowdowns in AGEM."
            )
        removed_els = len(dataset) - self.patterns_per_experience
        if removed_els > 0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.select(indices[: self.patterns_per_experience])

        self.buffers.append(dataset)
        persistent_workers = num_workers > 0
        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size=self.sample_size // len(self.buffers),
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=persistent_workers,
            collate_mbatches=self.data_collator,
        )
        self.buffer_dliter = iter(self.buffer_dataloader)

    def load_bench_state(self):
        super().load_bench_state()
        data_p = Path(self.output_dir) / "data"
        data_p.mkdir(parents=True, exist_ok=True)
        exp = self.current_exp  
        logger.info(f"Loading AGEM buffers of experience {exp}")
        storage_ckp = data_p / f"buffers_exp{exp}"

        if storage_ckp.is_file():
            import pickle

            with open(storage_ckp, "rb") as fin:
                buffers = pickle.load(fin)
                self.buffers = buffers
        else:
            logger.warning("No buffer found")

        length = 1 if len(self.buffers) == 0 else len(self.buffers)

        if self.buffer_dataloader is None:
            self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
                self.buffers,
                batch_size=self.sample_size // length,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                collate_mbatches=self.data_collator,
            )
            self.buffer_dliter = iter(self.buffer_dataloader)

    def save_bench_state(self):
        super().save_bench_state()
        import pickle

        data_p = Path(self.output_dir) / "data"
        data_p.mkdir(parents=True, exist_ok=True)
        # path of the pickled file
        exp = self.current_exp
        storage_ckp = data_p / f"buffers_exp{exp}"
        with open(storage_ckp, "wb") as fout:
            pickle.dump(self.buffers, fout)
