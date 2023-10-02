import math
import typing

class UpdateResult(typing.NamedTuple):
    is_best: bool
    should_stop: bool

class EarlyStoppingCriterion:
    """An API for implementing an early stopping criterion. The early stopping
    criterion can be tied to performance on a validation data set."""

    def update(self, value) -> UpdateResult:
        """Update the early stopping criterion with the latest value of a
        metric used to control early stopping (this should be validation
        performance), and check if training should stop now.

        :param value: The new value of the metric used for the stopping
            criterion.
        :return: A named tuple where ``is_best`` indicates whether this update
            had the best metric value seen so far (meaning the model
            parameters should probably be saved), and ``should_stop``
            indicates whether training should stop now.
        """
        raise NotImplementedError

class UpdatesWithoutImprovement(EarlyStoppingCriterion):
    """An early stopping criterion where training stops after a certain number
    of epochs/checkpoints without improvement in a certain metric (usually
    performance on a validation data set). Performance is always compared to
    the best value ever seen so far, not just the previous value."""

    def __init__(self, mode: str, patience: int=math.inf, goal=None):
        """
        :param mode: Either ``'min'`` or ``'max'``. ``min`` means that lower
            values are better; ``max`` means that higher values are better.
        :param patience: How many epochs without improvement will be tolerated
            before training is stopped. A patience of 1 will cause training to
            be stopped as soon as performance does not improve.
        :param goal: An optional value representing a perfect score which, if
            ever reached, will cause training to stop immediately.
        """
        super().__init__()
        if mode not in ('min', 'max'):
            raise ValueError
        self.mode = mode
        if mode == 'min':
            self.is_better = lambda x, y: x < y
            self.best = math.inf
        else:
            self.is_better = lambda x, y: x > y
            self.best = -math.inf
        self.patience = patience
        self.updates_since_improvement = 0
        if goal is None:
            if mode == 'min':
                goal = -math.inf
            else:
                goal = math.inf
        self.goal = goal

    def update(self, value):
        is_best = self.is_better(value, self.best)
        if is_best:
            self.best = value
            self.updates_since_improvement = 0
        else:
            self.updates_since_improvement += 1
        # Stop if patience has run out or if this update achieved a perfect
        # score.
        should_stop = (
            self.updates_since_improvement >= self.patience or
            not self.is_better(self.goal, value))
        return UpdateResult(is_best, should_stop)
