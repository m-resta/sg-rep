# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
Callbacks to use with the Trainer class and customize the training loop.
"""
from dataclasses import dataclass
from typing import Dict, Optional
from warnings import WarningMessage

import numpy as np
from tqdm.auto import tqdm
from transformers import TrainerState

from transformers.trainer_utils import IntervalStrategy, has_length
from transformers.training_args import TrainingArguments
from transformers.utils import logging

import torch


logger = logging.get_logger("transformers")

@dataclass
class TrainerControl:
    """
    A class that handles the [`Trainer`] control flow. This class is used by the [`TrainerCallback`] to activate some
    switches in the training loop.

    Args:
        should_training_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the training should be interrupted.

            If `True`, this variable will not be set back to `False`. The training will just stop.
        should_epoch_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the current epoch should be interrupted.

            If `True`, this variable will be set back to `False` at the beginning of the next epoch.
        should_save (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be saved at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_evaluate (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be evaluated at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_log (`bool`, *optional*, defaults to `False`):
            Whether or not the logs should be reported at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


class TrainerCallback:
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args ([`TrainingArguments`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            The model being trained.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for encoding the data.
        optimizer (`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        metrics (`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event `on_log`.

    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.

    The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformer.PrinterCallback`].

    Example:

    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```"""

    def before_backward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def after_backward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def before_forward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def after_forward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def before_optim_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def after_optim_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def before_eval_fwd(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def after_eval_fwd(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass


class CLCallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(self, callbacks, model, tokenizer, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None
        self.trainer = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    def set_trainer_instance(self, trainer):
        self.trainer = trainer

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def before_backward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("before_backward", args, state, control)

    def after_backward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("after_backward", args, state, control)

    def before_forward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("before_forward", args, state, control)

    def after_forward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("after_forward", args, state, control)

    def before_optim_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("before_optim_step", args, state, control)

    def after_optim_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("after_optim_step", args, state, control)

    def before_eval_fwd(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("before_eval_fwd", args, state, control)

    def after_eval_fwd(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("after_eval_fwd", args, state, control)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_substep_end", args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        return self.call_event("on_predict", args, state, control, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_prediction_step", args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % args.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
                args.evaluation_strategy == IntervalStrategy.STEPS
                and state.global_step % args.eval_steps == 0
                and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if (
                args.save_strategy == IntervalStrategy.STEPS
                and args.save_steps > 0
                and state.global_step % args.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Save
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True

        return control


class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(total=state.max_steps)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader), leave=self.training_bar is None)
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            self.training_bar.write(str(logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None


class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)


class EarlyStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
       early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
       early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`].
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
                args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
                args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

class BaseCLTrainerCallback(TrainerCallback):
    """
    Base class to implement callbacks and modify the training loop of a single experience
    """

    def __init__(self, trainer=None):
        self.trainer = trainer
        if trainer is None:
            print('No trainer specified. Rembember to pass a trainer object later',
                  'by using set_trainer() function')

    def set_trainer(self, trainer):
        self.trainer = trainer

    def set_cl_controller(self, controller):
        self.cl_controller = controller

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        logger.debug('CL-callback: on_init_end')
        return control

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        logger.debug(self.trainer)
        logger.debug('CL-callback: on_train_begin')
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        logger.debug('CL-callback: on_train_end')
        return control

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        logger.debug('CL-callback: on_epoch_begin')
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        logger.debug('CL-callback: on_epoch_end')
        return control

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        logger.debug('CL-callback: on_step_begin')
        return control

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        logger.debug('CL-callback: on_substep_end')
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        logger.debug('CL-callback: on_step_end')
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        logger.debug('CL-callback: on_evaluate')
        return control

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        logger.debug('CL-callback: on_predict')
        return control

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        logger.debug('CL-callback: on_save')
        return control

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        logger.debug('CL-callback: on_log')
        return control

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        logger.debug('CL-callback: on_prediction_step')
        return control

    def before_backward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug('CL-callback: before_backward')
        #logger.debug(self.trainer.__dict__)
        return control

    def after_backward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug('CL-callback: after_backward')
        return control

    def before_forward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug('CL-callback: before_forward')
        return control

    def after_forward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug('CL-callback: after_forward')
        return control

    def before_optim_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug('CL-callback: before_optim_step')
        return control

    def after_optim_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug('CL-callback: after_optim_step')
        return control

    def before_eval_fwd(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug('CL-callback: before_eval_fwd')
        return control

    def after_eval_fwd(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug('CL-callback: after_eval_fwd')
        return control


class EWCCallback(BaseCLTrainerCallback):
    """
        Implement EWC regularization. Code from Avalanche by Continual AI
    """

    def __init__(self, trainer=None, cl_controller=None):
        super().__init__(trainer)
        self.cl_controller = cl_controller

    def before_backward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.cl_controller.current_exp == 0:
            return control

        strategy = self.cl_controller
        import torch

        penalty = torch.tensor(0).float().to(strategy.model.device)

        if strategy.mode == "separate":
            for experience in range(strategy.current_exp):
                named_params = strategy.model.named_parameters()
                saved_params = strategy.saved_params[experience]
                importances = strategy.importances[experience]
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(),
                    strategy.saved_params[experience],
                    strategy.importances[experience],
                ):
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    n_units = saved_param.shape[0]
                    imp = imp.to(strategy.model.device)
                    saved_param = saved_param.to(strategy.model.device)
                    cur_param = cur_param[:n_units].to(strategy.model.device)
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
                    imp = imp.to('cpu')
                    saved_param = saved_param.to('cpu')
                    cur_param = cur_param[:n_units].to('cpu')
        elif strategy.mode == "online":
            prev_exp = strategy.current_exp - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                strategy.model.named_parameters(),
                strategy.saved_params[prev_exp],
                strategy.importances[prev_exp],
            ):
                # dynamic models may add new units
                # new units are ignored by the regularization
                n_units = saved_param.shape[0]
                cur_param = cur_param[:n_units].to(strategy.model.device)
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        self.trainer.step_loss += strategy.ewc_lambda * penalty

        return control


class AGEMCallback(BaseCLTrainerCallback):
    """
    Code adapteed from  Avalanche by ContinualAI
    """

    def __init__(self, trainer=None, cl_controller=None):
        super().__init__(trainer)
        self.cl_controller = cl_controller



    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        strategy = self.cl_controller
        # WORKING CODE
        if len(strategy.buffers) > 0 :
            strategy.model.train()
            self.trainer.optimizer.zero_grad()
            mb = self.sample_from_memory()
            used_cols = strategy.trainer._signature_columns
            to_remove = []
            for k,v in mb.items():
                if k not in used_cols:
                    to_remove.append(k)
            for k in to_remove:
                mb.pop(k)
            mb.to(strategy.model.device)
            outputs = strategy.model(**mb)
            loss = outputs['loss']
            loss.backward()
            # gradient can be None for some head on multi-headed models
            self.reference_gradients_list = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.model.device)
                for n, p in strategy.model.named_parameters()
            ]
            self.reference_gradients = torch.cat(self.reference_gradients_list)
            self.trainer.optimizer.zero_grad()
            mb.to('cpu')
        #if len(strategy.buffers) > 0 :
        #    strategy.model.train()
        #    self.trainer.optimizer.zero_grad()
        #    mb = self.sample_from_memory()
        #    used_cols = strategy.trainer._signature_columns
        #    to_remove = []
        #    for k,v in mb.items():
        #        if k not in used_cols:
        #            to_remove.append(k)
        #    for k in to_remove:
        #        mb.pop(k)
        #    mb.to(strategy.model.device)
        #    for i, batch in enumerate(mb):
        #        outputs = strategy.model(**batch)
        #        loss = outputs['loss']
        #        loss.backward()
        #    # gradient can be None for some head on multi-headed models
        #    self.reference_gradients_list = [
        #        p.grad.view(-1)
        #        if p.grad is not None
        #        else torch.zeros(p.numel(), device=strategy.model.device)
        #        for n, p in strategy.model.named_parameters()
        #    ]
        #    self.reference_gradients = torch.cat(self.reference_gradients_list)
        #    self.trainer.optimizer.zero_grad()
        #    mb.to('cpu')

        return control 


    def sample_from_memory(self):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """
        return next(self.cl_controller.buffer_dliter)

    def after_backward(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Project gradient based on reference gradients
        """
        strategy = self.cl_controller

        if len(strategy.buffers) > 0 and strategy.model.training:
            current_gradients_list = [
                p.grad.view(-1)
                if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients_list)

            assert (
                current_gradients.shape == self.reference_gradients.shape
            ), "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(
                    self.reference_gradients, self.reference_gradients
                )
                grad_proj = (
                    current_gradients - self.reference_gradients * alpha2
                )

                count = 0
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(
                            grad_proj[count : count + n_param].view_as(p)
                        )
                    count += n_param

        return control
