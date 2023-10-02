from typing import Any

import torch

class BasicComposable(torch.nn.Module):

    def __init__(self, main):
        super().__init__()
        self._composable_is_main = main

    def __or__(self, other: torch.nn.Module) -> 'Composed':
        if not isinstance(other, BasicComposable):
            other = Composable(other)
        return Composed(self, other)

class Composable(BasicComposable):

    def __init__(self, module: torch.nn.Module, main=False, tags=None, kwargs=None):
        super().__init__(main)
        self.module = module
        self._composable_tags = tags if tags is not None else set()
        self._composable_kwargs = kwargs if kwargs is not None else {}

    def forward(self, *args, tag_kwargs=None, **kwargs):
        if tag_kwargs:
            # TODO Nondeterministic iteration order might be a problem.
            for tag in self._composable_tags:
                if tag in tag_kwargs:
                    kwargs.update(tag_kwargs[tag])
        return self.module(*args, **self._composable_kwargs, **kwargs)

    def main(self) -> 'Composable':
        self._composable_is_main = True
        return self

    def tag(self, tag: str) -> 'Composable':
        self._composable_tags.add(tag)
        return self

    def kwargs(self, **kwargs: Any) -> 'Composable':
        self._composable_kwargs.update(kwargs)
        return self

class Composed(BasicComposable):

    def __init__(self, first: Composable, second: Composable):
        super().__init__(first._composable_is_main or second._composable_is_main)
        self.first = first
        self.second = second

    def forward(self, x, *args, tag_kwargs=None, **kwargs):
        first_args, first_kwargs = get_composed_args(self.first, args, kwargs, tag_kwargs)
        second_args, second_kwargs = get_composed_args(self.second, args, kwargs, tag_kwargs)
        return self.second(self.first(x, *first_args, **first_kwargs), *second_args, **second_kwargs)

def get_composed_args(composable, args, kwargs, tag_kwargs):
    new_args = []
    new_kwargs = {}
    if composable._composable_is_main:
        new_args.extend(args)
        new_kwargs.update(kwargs)
    if tag_kwargs:
        new_kwargs['tag_kwargs'] = tag_kwargs
    return new_args, new_kwargs
