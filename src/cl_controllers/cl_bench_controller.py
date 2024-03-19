# Copyright 2022 Michele Resta
# Apache License
import datetime
import json
import shutil
from transformers.utils import logging
from pathlib import Path
import os
from typing import List


from transformers import Seq2SeqTrainingArguments, T5ForConditionalGeneration
from ..trainers.clhf_trainer_v2 import CLTrainerV2 as CLTrainer

logger = logging.get_logger("transformers")

class ContinualBenchmarkController:
    """
    Class that takes care of the outer loop of a continual learning
    stream of experiences
    A complete CL strategy is created by subclassing ContinualBenchmarkController
    and BasicCLTrainerCallback to customize the behaviour
    """

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
            state_savename='cl_bench_state.json',
            **kwargs
    ):
        self.experiences: List = experiences
        self.strategy_callback = strategy_callback
        self.trainer_args = trainer_args
        self.data_collator = data_collator
        self.model = model
        self.kwargs = kwargs
        self.trainer = None
        self.bench_state = {}
        self.experiences_result = []
        self.state_savename = state_savename
        self.output_dir = self.trainer_args.output_dir
        self.current_exp = 0
        self.current_train_data = self.experiences[0].train_dataset
        self.current_eval_data = self.experiences[0].eval_dataset
        self.compute_metrics = compute_metrics
        self.logging_callback = None
        self.logging_root = self.trainer_args.logging_dir
        self.additional_callbacks = additional_callbacks
        self.fastdebug=fastdebug
        self.start_from_exp = start_from_exp
        self.checkpoint_to_reload = None

    def create_current_exp(self, exp_num, **kwargs):
        pass


    def before_experience(self, exp_number, **kwargs):
        """
        Here we initialize the Trainer and we pass the arguments
        :param kwargs:
        :return:
        """

        self.trainer_args.logging_dir = self.logging_root + f'/exp_{exp_number}'

        if self.current_exp > 0:
            # we have to restart from epoch 0 so we add --ignore_data_skip to the Training arguments
            self.trainer_args.ignore_data_skip = True

        self.trainer = CLTrainer(
            model=self.model,
            args=self.trainer_args,
            train_dataset=self.current_train_data,
            eval_dataset=self.current_eval_data,
            data_collator=self.data_collator
        )

        self.strategy_callback.set_trainer(self.trainer)
        self.trainer.add_callback(self.strategy_callback)
        for c in self.additional_callbacks:
            self.trainer.add_callback(c)
        logger.debug('CL controller: before experience')
        self.checkpoint_to_reload = self.load_model_for_exp()
        


    def test_set_predictions(self, **kwargs):
        self.model.eval()
        eval_res = {}
        pred_res = {}
        for k, v in self.experiences[self.current_exp].test_datasets.items():
            # todo remove
            res = self.trainer.evaluate(eval_dataset=v)
            eval_res[k] = res
        logger.info(f'Scores on test sets for Experience {self.current_exp}:\n {str(eval_res)}')
        return eval_res


    def get_last_checkpoint_folder(self, checkpoints_path : Path) -> Path:
        folders = [str(c) for c in checkpoints_path.iterdir()]
        chk_folders = []
        last_chk = ''
        if Path.is_dir(checkpoints_path):
            for name in folders:
                if 'checkpoint' in name:
                    chk_folders.append(name)
        last_chk = sorted(chk_folders, reverse=True)[0]
        return Path(last_chk)

    def save_best_model(self, **kwargs):
        """ Saves the best model in a separate folder at the end of the end of exp"""
        save_path = Path(self.trainer_args.output_dir) / f'exp_{self.current_exp}-best_model'
        self.trainer._save(str(save_path))

    def after_experience(self, **kwargs):
        save_path = Path(self.trainer_args.output_dir) / f'exp_{self.current_exp}_test_res.json'
        save_path_s = Path(self.trainer_args.output_dir) / f'exp_{self.current_exp}_test_samples.json'
        logger.debug('CL controller: after experience')
        self.save_best_model()
        self.current_exp += 1

    def load_model_for_exp(self):
        """ 
        Load the bet model into self.model and return a path to a checkpoint or None
        """
        chk_path = Path(self.trainer_args.output_dir)
        folders = [str(c) for c in chk_path.iterdir()]
        reload = False
        chk_folders = []
        last_chk = ''
        if Path.is_dir(chk_path):
            for name in folders:
                if 'checkpoint' in name:
                    reload = True
                    chk_folders.append(name)

        if reload:
            to_reload = ''
            best_path = Path(self.trainer_args.output_dir) / f'exp_{self.current_exp -1}-best_model'
            if Path.is_dir(best_path):
                to_reload = ''
                last_chk = sorted(chk_folders, reverse=True)[0]
                date_best = os.path.getmtime(best_path)
                date_chkp = os.path.getmtime(last_chk)
                if date_best > date_chkp: # best checkpoint has been modified later. we load it
                    to_reload = best_path
                    for d in chk_folders:
                        shutil.rmtree(d)
                elif date_best < date_chkp:
                    to_reload = last_chk
                else:
                    to_reload = best_path

                self.model = T5ForConditionalGeneration.from_pretrained(str(best_path))
                logger.info(f'Model save dir exists, attempting to reload latest model from {best_path}')
                return to_reload
            else:
                last_chk = sorted(chk_folders, reverse=True)[0]
                to_reload = last_chk
                logger.info(f'Model save dir exists, attempting to reload latest model from last chkp {last_chk}')
                # TODO Dynamically load correct model class based on cmdline args or config.json
                #self.model = T5ForConditionalGeneration.from_pretrained(last_chk)
            return to_reload

        else:
            # We don't have checkpoints folders: If we have a best model from prec exp we load it.
            # Otherwise we start from scratch
            best_path = Path(self.trainer_args.output_dir) / f'exp_{self.current_exp -1}-best_model'
            if Path.is_dir(best_path):
                to_reload = best_path
                return best_path
            else:
                logger.info('Model save dir not found. Starting from scratch')
                return None



    def train_experience(self, **kwargs):
        to_reload = self.checkpoint_to_reload
        if to_reload is None:
            result = self.trainer.train()
        else:
            result = self.trainer.train(resume_from_checkpoint=to_reload)

        self.experiences_result.append(result)


    def save_bench_state(self):
        self.bench_state['current_exp'] = self.current_exp
        self.bench_state['trainer_args'] = self.trainer_args.to_dict()
        self.bench_state['exp_result'] = self.experiences_result
        self.bench_state['total_exps'] = len(self.experiences)
        now = datetime.datetime.now().isoformat()
        self.bench_state['save_timestamp'] = now

        out_dir = self.trainer_args.output_dir
        out_path = Path(out_dir)
        save_path = out_path / self.state_savename
        with open(save_path, 'w') as outfile:
            json.dump(self.bench_state, outfile, indent=4)
        logger.info(f'CL Benchmark state saved to {save_path}')


    def load_bench_state(self):
        out_path = Path(self.output_dir)
        save_path = out_path / self.state_savename
        if save_path.is_file():
            with open(save_path, 'r') as infile:
                self.bench_state = json.load(infile)
            self.current_exp = self.bench_state['current_exp']
            self.trainer_args = Seq2SeqTrainingArguments(**self.bench_state['trainer_args'])
            self.experiences_result = self.bench_state['exp_result']
            logger.info(f'CL Benchmark state restored from {save_path}')
        else:
            logger.info(f'Save state for CL benchmark not found in {out_path}.')
            logger.info('Starting Training from scratch')


    def partial_train(self, start_index):
        # check if best model exists and if storage exists
        best_path = Path(self.trainer_args.output_dir) / f'exp_{start_index - 1}-best_model'
        data_p = Path(self.trainer_args.output_dir) / 'data'
        storage_ckp = data_p / f'storage_policy_exp{start_index}'
        if not Path.is_dir(best_path):
            logger.error(f"Cannot start from index {start_index}: the model of the previous experience is not there. Searched {str(best_path)}")
            return -1
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(best_path)
            if storage_ckp.is_file():
                return start_index
            else:
                logger.info(f'Trying to run before_exp and after_experience for exp {start_index -1}')
                self.before_experience(start_index -1)
                self.after_experience()
                self.save_bench_state()
                return start_index



    def continual_train(self):
        self.load_bench_state()
        if self.start_from_exp != 0:
            start_from = self.partial_train(self.start_from_exp)
            if start_from == -1:
                return
            else:
                self.load_bench_state()
                self.current_exp = start_from
        if self.current_exp == -1:
            # we are starting
            self.current_exp += 1
        for i, experience in enumerate(self.experiences):
            if i < self.current_exp:
                continue

            logger.info(f' ========== Starting from Experience {i} ==========')
            self.create_current_exp(i)
            self.before_experience(i)
            self.train_experience()
            self.after_experience()
            self.save_bench_state()
        logger.info('Training finished!')
