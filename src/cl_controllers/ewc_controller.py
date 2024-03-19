from collections import defaultdict
import itertools
from pathlib import Path

from datasets import concatenate_datasets
from transformers.utils import logging
from ..cl_controllers.cl_bench_controller import ContinualBenchmarkController
from ..trainers.clhf_trainer_seq2seq_v2 import Seq2SeqTrainerV2
from ..trainers.logging_integration import TensorBoardCallback
import torch

logger = logging.get_logger("transformers")

SEED = 4242


class EWCController(ContinualBenchmarkController):
    def __init__(
        self,
        model,
        trainer_args,
        data_collator,
        experiences,
        strategy_callback,
        compute_metrics,
        additional_callbacks,
        ewc_lambda,
        criterion=None,
        fastdebug=False,
        start_from_exp=0,
        state_savename="cl_bench_state.json",
        storage_policy=None,
        mode="separate",
        decay_factor=None,
        keep_importance_data=None,
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

        assert (decay_factor is None) or (
            mode == "online"
        ), "You need to set `online` mode to use `decay_factor`."
        assert (decay_factor is not None) or (
            mode != "online"
        ), "You need to set `decay_factor` to use the `online` mode."
        assert (
            mode == "separate" or mode == "online"
        ), "Mode must be separate or online."

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor
        self.criterion = criterion if not None else CrossEntropyLoss()

        if self.mode == "separate":
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)

    def zerolike_params_dict(self, model):
        """
        Create a list of (name, parameter), where parameter is initalized to zero.
        The list has as many parameters as model, with the same size.

        :param model: a pytorch model
        """

        return [
            (k, torch.zeros_like(p).to(p.device)) for k, p in model.named_parameters()
        ]

    def copy_params_dict(self, model, copy_grad=False):
        """
        Create a list of (name, parameter), where parameter is copied from model.
        The list has as many parameters as model, with the same size.

        :param model: a pytorch model
        :param copy_grad: if True returns gradients instead of parameter values
        """

        if copy_grad:
            return [(k, p.grad.data.clone()) for k, p in model.named_parameters()]
        else:
            return [(k, p.data.clone()) for k, p in model.named_parameters()]

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
        super().before_experience(exp_number)
        # we reinstantiate the correct trainer

        self.trainer = Seq2SeqTrainerV2(
            model=self.model,
            args=self.trainer_args,
            train_dataset=self.current_train_data,
            eval_dataset=self.current_eval_data,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.strategy_callback.set_trainer(self.trainer)
        self.strategy_callback.set_cl_controller(self)
        self.trainer.add_callback(self.strategy_callback)
        for c in self.additional_callbacks:
            self.trainer.add_callback(c)
        if self.logging_callback is not None:
            self.trainer.remove_callback(TensorBoardCallback)
            self.trainer.add_callback(self.logging_callback)
        # print('current train data len', len(self.current_train_data))

    def after_experience(self, **kwargs):
        """
        Compute importances of parwameters after each experience.
        """
        exp_counter = self.current_exp
        current_device = self.model.device
        self.model.to("cpu")
        importances = self.compute_importances(
            self.model,
            self.criterion,
            self.trainer.optimizer,
            self.current_train_data,
            current_device,
            # self.trainer.model.device,
            self.trainer_args.per_device_train_batch_size,
        )
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = self.copy_params_dict(self.model)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

        self.model.to("cpu")

        super().after_experience()

    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size
    ):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()
        model.to(device)

        # list of list
        importances = self.zerolike_params_dict(model)
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        # importances.to(device)
        self.trainer._train_batch_size = 100
        dataloader = self.trainer.get_train_dataloader()
        for step, inputs in enumerate(dataloader):
            if step % 10000 == 0:
                logger.info(
                    f"Computing EWC importance... step {step}, device: {str(model.device)}"
                )

            inputs.to(device)
            outputs = model(**inputs)
            loss = outputs["loss"]
            loss.backward()
            for (k1, p), (k2, imp) in zip(model.named_parameters(), importances):
                assert k1 == k2
                if p.grad is not None:
                    imp = imp.to(device)
                    imp += p.grad.data.clone().pow(2)
                imp.to("cpu")
            inputs.to("cpu")
            del outputs

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))
        # importances = importances.to('cpu')

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == "separate" or t == 0:
            self.importances[t] = importances
        elif self.mode == "online":
            for (k1, old_imp), (k2, curr_imp) in itertools.zip_longest(
                self.importances[t - 1],
                importances,
                fillvalue=(None, None),
            ):
                # Add new module importances to the importances value (New head)
                if k1 is None:
                    self.importances[t].append((k2, curr_imp))
                    continue

                assert k1 == k2, "Error in importance computation."

                self.importances[t].append(
                    (k1, (self.decay_factor * old_imp + curr_imp))
                )

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong EWC mode.")

    def load_bench_state(self):
        super().load_bench_state()
        data_p = Path(self.output_dir) / "data"
        data_p.mkdir(parents=True, exist_ok=True)
        # path of the pickled file
        exp = self.current_exp  # self.bench_state['current_exp']
        logger.info(f"Loading EWC state of experience {exp}")
        imp_ckp = data_p / f"importances_exp{exp}"
        saved_params_ckp = data_p / f"saved_params_exp{exp}"
        if imp_ckp.is_file():
            import pickle

            with open(imp_ckp, "rb") as fin:
                importances = pickle.load(fin)
                self.importances = importances
        else:
            print("No importence files found")
        if saved_params_ckp.is_file():
            import pickle

            with open(saved_params_ckp, "rb") as fin:
                saved_params = pickle.load(fin)
                self.saved_params = saved_params
        else:
            print("No saved ewc params found files found")

    def save_bench_state(self):
        super().save_bench_state()
        import pickle

        data_p = Path(self.output_dir) / "data"
        data_p.mkdir(parents=True, exist_ok=True)
        # path of the pickled file
        exp = self.current_exp
        imp_ckp = data_p / f"importances_exp{exp}"
        saved_params_ckp = data_p / f"saved_params_exp{exp}"
        with open(imp_ckp, "wb") as fout:
            pickle.dump(self.importances, fout)
        with open(saved_params_ckp, "wb") as fout:
            pickle.dump(self.saved_params, fout)
