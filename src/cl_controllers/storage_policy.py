"""
Code adapted from continualAI Avalanche Library
"""
from abc import ABC, abstractmethod
from pathlib import Path

import datasets.arrow_dataset
import torch
from datasets import concatenate_datasets

from .cl_bench_controller import ContinualBenchmarkController


class MemoryBuffer(ABC):
    """ABC for rehearsal buffers to store exemplars.

    `self.buffer` is an AvalancheDataset of samples collected from the previous
    experiences. The buffer can be updated by calling `self.update(strategy)`.
    """

    def __init__(self, max_size: int):
        """Init.

        :param max_size: max number of input samples in the replay memory.
        """
        self.max_size = max_size
        """ Maximum size of the buffer. """
        self._buffer = []

    @property
    def buffer(self):
        """Buffer of samples."""
        return self._buffer

    @buffer.setter
    def buffer(self, new_buffer):
        self._buffer = new_buffer

    @abstractmethod
    def update(self, bench_controller, **kwargs):
        """Update `self.buffer` using the `strategy` state.

        :param controller of cl benchmark:
        :param kwargs:
        :return:
        """
        ...

    @abstractmethod
    def resize(self, bench_controller, new_size: int):
        """Update the maximum size of the buffer.

        :param controller of cl benchmark:
        :param new_size:
        :return:
        """
        ...

    @abstractmethod
    def save_to_disk(self, bench_controller, path):
        """

        :param bench_controller:
        :param path:
        :return:
        """

    def set_from_disk(self, bench_controller, buffer):
        """

        :param bench_controller:
        :return:
        """


class ReservoirSamplingBuffer(MemoryBuffer):
    """Buffer updated with reservoir sampling."""

    def __init__(self, max_size: int):
        """
        :param max_size:
        """
        # The algorithm follows
        # https://en.wikipedia.org/wiki/Reservoir_sampling
        # We sample a random uniform value in [0, 1] for each sample and
        # choose the `size` samples with higher values.
        # This is equivalent to a random selection of `size_samples`
        # from the entire stream.
        super().__init__(max_size)
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)

    def update(self, bench_controller: ContinualBenchmarkController, **kwargs):
        """Update buffer."""
        self.update_from_dataset(
            bench_controller.experiences[bench_controller.current_exp]["train_dataset"]
        )

    def update_from_dataset(self, new_data):
        """Update the buffer using the given dataset.

        :param new_data:
        :return:
        """
        new_weights = torch.rand(len(new_data))

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        if isinstance(new_data, datasets.arrow_dataset.Dataset):
            if len(self.buffer) == 0:
                cat_data = new_data
            else:
                cat_data = concatenate_datasets([new_data, self.buffer])
        else:
            raise NotImplementedError(
                "we only support hugginface datasets at the moment"
            )
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer = cat_data.select(buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]

    def resize(self, strategy, new_size):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        if isinstance(self.buffer, datasets.arrow_dataset.Dataset):
            self.buffer = self.buffer.select(torch.arange(self.max_size))
            self._buffer_weights = self._buffer_weights[: self.max_size]
        else:
            raise NotImplementedError(
                "we only support hugginface datasets at the moment"
            )

    def save_to_disk(self, bench_controller, path):
        pass


class BalancedExemplarsBuffer(MemoryBuffer):
    """A buffer that stores exemplars for rehearsal in separate groups.

    The grouping allows to balance the data (by task, experience,
    classes..). In combination with balanced data loaders, it can be used
    to sample balanced mini-batches during training.

    `self.buffer_groups` is a dictionary that stores each group as a
    separate buffer. The buffers are updated by calling
    `self.update(strategy)`.
    """

    def __init__(
        self, max_size: int, adaptive_size: bool = True, total_num_groups=None
    ):
        """
        :param max_size: max number of input samples in the replay memory.
        :param adaptive_size: True if max_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param total_num_groups: If adaptive size is False, the fixed number
                                of groups to divide capacity over.
        """
        super().__init__(max_size)
        self.adaptive_size = adaptive_size
        self.total_num_groups = total_num_groups
        if not self.adaptive_size:
            assert self.total_num_groups > 0, (
                "You need to specify `total_num_groups` if " "`adaptive_size=True`."
            )
        else:
            assert self.total_num_groups is None, (
                "`total_num_groups` is not compatible with " "`adaptive_size=False`."
            )
        self.buffer_groups = {}
        """ Dictionary of buffers. """

    @property
    def buffer_datasets(self):
        """Return group buffers as a list of `AvalancheDataset`s."""
        return [g.buffer for g in self.buffer_groups.values()]

    def get_group_lengths(self, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        if self.adaptive_size:
            lengths = [self.max_size // num_groups for _ in range(num_groups)]
            # distribute remaining size among experiences.
            rem = self.max_size - sum(lengths)
            for i in range(rem):
                lengths[i] += 1
        else:
            lengths = [
                self.max_size // self.total_num_groups for _ in range(num_groups)
            ]
        return lengths

    @property
    def buffer(self):
        # return concatenate_datataset([g.buffer for g in self.buffer_groups.values() if len(g.buffer) > 0])
        to_concat = [g.buffer for g in self.buffer_groups.values() if len(g.buffer) > 0]
        if len(to_concat) > 0:
            return concatenate_datasets(to_concat)
        else:
            return []

    @buffer.setter
    def buffer(self, new_buffer):
        assert NotImplementedError(
            "Cannot set `self.buffer` for this class. "
            "You should modify `self.buffer_groups instead."
        )

    @abstractmethod
    def update(self, bench_controller, **kwargs):
        """Update `self.buffer_groups` using the `strategy` state.

        :param strategy:
        :param kwargs:
        :return:
        """
        ...

    def resize(self, bench_controller, new_size):
        """Update the maximum size of the buffers."""
        self.max_size = new_size
        lens = self.get_group_lengths(len(self.buffer_groups))
        for ll, buffer in zip(lens, self.buffer_groups.values()):
            buffer.resize(bench_controller, ll)


class ExperienceBalancedBuffer(BalancedExemplarsBuffer):
    """
    Rehearsal buffer with samples balanced over experiences.

    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(self, max_size: int, adaptive_size: bool = True, num_experiences=None):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        """
        super().__init__(max_size, adaptive_size, num_experiences)

    def update(self, bench_controller, **kwargs):
        new_data = datasets.concatenate_datasets(
            list(
                bench_controller.experiences[
                    bench_controller.current_exp
                ].train_dataset.values()
            )
        )
        num_exps = bench_controller.current_exp + 1
        lens = self.get_group_lengths(num_exps)

        new_buffer = ReservoirSamplingBuffer(lens[-1])
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(bench_controller, ll)

    def update_selfgenerated(self, bench_controller, new_data, **kwargs):
        """Use new_data to create the buffer"""
        num_exps = bench_controller.current_exp + 1
        lens = self.get_group_lengths(num_exps)

        new_buffer = ReservoirSamplingBuffer(lens[-1])
        new_buffer.update_from_dataset(new_data)
        self.buffer_groups[num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(bench_controller, ll)

    def save_to_disk(self, bench_controller, prefix="", path=None):
        tok = bench_controller.tokenizer
        # implementation changed
