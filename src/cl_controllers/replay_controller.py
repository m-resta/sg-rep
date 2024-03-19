from pathlib import Path

from datasets import concatenate_datasets
from transformers.utils import logging
from .cl_bench_controller import ContinualBenchmarkController
from .storage_policy import ExperienceBalancedBuffer
from ..trainers.clhf_trainer_seq2seq_v2 import Seq2SeqTrainerV2 as Seq2SeqTrainer
from ..trainers.logging_integration import TensorBoardCallback

logger = logging.get_logger("transformers")

SEED = 4242

class ReplayController(ContinualBenchmarkController):

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
            tokenizer=None,
            start_from_exp=0,
            state_savename='cl_bench_state.json',
            mem_size: int = 200,
            storage_policy=None,
            **kwargs
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
            state_savename)
        self.mem_size = mem_size
        self.tokenizer = tokenizer

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    def before_experience(self, exp_number, **kwargs):
        if self.current_exp == 0:
            datasets = list(self.experiences[exp_number].train_dataset.values())
            if self.fastdebug:
                # Debug purpose
                self.current_train_data = concatenate_datasets(datasets).select(range(15))
                self.current_eval_data = concatenate_datasets(list(self.experiences[exp_number].eval_dataset.values())).select(range(1))
            else:
                # Run normally
                self.current_train_data = concatenate_datasets(datasets)
                self.current_eval_data = concatenate_datasets(list(self.experiences[exp_number].eval_dataset.values()))

            # shuffling
            self.current_eval_data = self.current_eval_data.shuffle(seed=SEED)
            self.current_train_data = self.current_train_data.shuffle(seed=SEED)
        else:
            if self.mem_size > 0:
                datasets = list(self.experiences[exp_number].train_dataset.values()) + [self.storage_policy.buffer]
                logger.info(f'Concatenating buffer with current experience buffer size {len(self.storage_policy.buffer)}')
            else:
                datasets = list(self.experiences[exp_number].train_dataset.values())


            eval_datasets = list(self.experiences[exp_number].eval_dataset.values())

            if self.fastdebug:
                # Debug purpose
                self.current_train_data = concatenate_datasets(datasets).select(range(15))
                self.current_eval_data = concatenate_datasets(eval_datasets).select(range(1))
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
            compute_metrics=self.compute_metrics
        )

        self.strategy_callback.set_trainer(self.trainer)
        self.trainer.add_callback(self.strategy_callback)
        for c in self.additional_callbacks:
            self.trainer.add_callback(c)
        if self.logging_callback is not None:
            self.trainer.remove_callback(TensorBoardCallback)
            self.trainer.add_callback(self.logging_callback)

    def after_experience(self, **kwargs):
        if self.mem_size > 0:
            self.storage_policy.update(self, **kwargs)
            self.storage_policy.save_to_disk(self, **kwargs)
        super().after_experience()

    def load_bench_state(self):
        super().load_bench_state()
        data_p = Path(self.output_dir) / 'data'
        data_p.mkdir(parents=True, exist_ok=True)
        # path of the pickled file
        exp = self.current_exp #self.bench_state['current_exp']
        logger.info(f'Loading storage of experience {exp}')
        storage_ckp = data_p / f'storage_policy_exp{exp}'
        if storage_ckp.is_file():
            import pickle
            with open(storage_ckp, 'rb') as fin:
                storage_p = pickle.load(fin)
                self.storage_policy = storage_p
        else:
            print('No buffer found')

    def save_bench_state(self):
        super().save_bench_state()
        import pickle
        data_p = Path(self.output_dir) / 'data'
        data_p.mkdir(parents=True, exist_ok=True)
        # path of the pickled file
        exp = self.current_exp 
        storage_ckp = data_p / f'storage_policy_exp{exp}'
        with open(storage_ckp, 'wb') as fout:
            pickle.dump(self.storage_policy, fout)
