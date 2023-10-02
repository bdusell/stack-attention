import collections.abc
import typing

import torch

from .layer import Layer

def default_fallback(name, data, generator):
    data.uniform_(generator=generator)

def _raise_return_error():
    raise TypeError(
        'initialize() should return an iterable of '
        '`torch.nn.Parameter`s containing the parameters that were '
        'initialized and that should be skipped by the fallback '
        'initializer (almost always this is just `module.parameters()`)')

def init_modules(
        module: torch.nn.Module,
        initialize: typing.Callable[
            [str, torch.nn.Module, typing.Optional[torch.Generator]],
            typing.Tuple[bool, typing.Optional[typing.Iterable[torch.nn.Parameter]]]
        ],
        fallback: typing.Callable[
            [str, torch.Tensor, typing.Optional[torch.Generator]],
            None
        ]=default_fallback,
        generator: typing.Optional[torch.Generator]=None
    ) -> None:
    """Recursively initialize the parameters of a module using a callback
    function that can be triggered on certain sub-modules.
    
    Recursion will stop when the callback is triggered, so parameters will not
    be initialized more than once by mistake. On the other hand, a fallback
    callback will be used to initialize all leftover parameters, so all
    parameters will be initialized exactly once.

    :param module: The module.
    :param initialize: A callback function that is called on each sub-module
        in ``module``. Its arguments are the name of the sub-module, the
        sub-module object itself, and a RNG. It should return a pair whose
        elements are a boolean indicating whether the callback actually
        initialized the parameters in the sub-module and did not decide to
        skip it, and an iterable of parameters that were initialized. If the
        iterable of parameters is ``None``, then all parameters in the
        sub-module will be used. Any parameters in the sub-module that were
        not returned will be treated as leftovers and passed to the fallback
        callback.
    :param fallback: A callback that will be called on all parameters not
        initialized by the ``initialize`` callback. Its arguments are the name
        of the parameter, the parameter's tensor object, and a RNG. By
        default, parameters are initialized uniformly from :math:`[0, 1]`.
    :param generator: Optional PyTorch RNG passed to callbacks.
    """
    seen = set()
    def recurse(module):
        for submodule_name, submodule in module.named_children():
            matched, params = initialize(submodule_name, submodule, generator)
            if matched:
                if params is None:
                    params = submodule.parameters()
                for param in params:
                    if not isinstance(param, torch.nn.Parameter):
                        _raise_return_error()
                    seen.add(id(param))
            else:
                recurse(submodule)
    recurse(module)
    for param_name, param in module.named_parameters():
        if id(param) not in seen:
            fallback(param_name, param.data, generator)

def init_modules_by_type(
        module: torch.nn.Module,
        callbacks: typing.Union[
            typing.Mapping[
                typing.Type[torch.nn.Module],
                typing.Callable[
                    [str, torch.nn.Module, torch.Generator],
                    typing.Optional[typing.Iterable[torch.nn.Parameter]]
                ]
            ],
            typing.Iterable[typing.Tuple[
                typing.Type[torch.nn.Module],
                typing.Callable[
                    [str, torch.nn.Module, torch.Generator],
                    typing.Optional[typing.Iterable[torch.nn.Parameter]]
                ]
            ]]
        ],
        fallback: typing.Callable[
            [str, torch.Tensor, typing.Optional[torch.Generator]],
            None
        ]=default_fallback,
        generator: typing.Optional[torch.Generator]=None
    ) -> None:
    """Recursively initialize the parameters of a module using callbacks that
    are triggered based on the type of each sub-module.
    
    Recursion will stop when the callback is triggered, so parameters will not
    be initialized more than once by mistake. On the other hand, a fallback
    callback will be used to initialize all leftover parameters, so all
    parameters will be initialized exactly once.

    :param module: The module.
    :param callbacks: A dict or list of pairs mapping module types to
        callbacks. Using a list is more flexible because it works on
        subclasses. The arguments to each callback are the name of the
        sub-module, the sub-module object itself, and a RNG. Each callback can
        optionally return a list of parameters that it initialized; all
        parameters in the sub-module not returned will be treated as leftovers
        and passed to the fallback callback. By default, all parameters in the
        sub-module are assumed to have been initialized, as if
        ``.parameters()`` were returned.
    :param fallback: A callback that will be called on all parameters not
        initialized by one of the other callbacks. See :py:func:`init_modules`.
    :param generator: Optional PyTorch RNG passed to callbacks.
    """
    if isinstance(callbacks, collections.abc.Mapping):
        def initialize(name, module, generator):
            func = callbacks.get(type(module))
            if func is not None:
                return True, func(name, module, generator)
            else:
                return False, None
    else:
        if not isinstance(callbacks, collections.abc.Sequence):
            callbacks = tuple(callbacks)
        def initialize(name, module, generator):
            for type_, func in callbacks:
                if isinstance(module, type_):
                    return True, func(name, module, generator)
            return False, None
    init_modules(module, initialize, fallback, generator)

def xavier_uniform_init(
        module: torch.nn.Module,
        generator: typing.Optional[torch.Generator]=None,
        fallback: typing.Callable[
            [str, torch.Tensor, typing.Optional[torch.Generator]],
            None
        ]=default_fallback,
    ) -> None:
    """Initialize all :py:class:`Layer`s in a module with
    Xavier uniform initialization, and use a default for all other parameters.

    :param module: The module.
    :param generator: Optional PyTorch RNG.
    :param fallback: A callback that will be called on all parameters not
        in a :py:class:`Layer`. See :py:func:`init_modules`.
    """
    def init_layer(name, module, generator):
        module.xavier_uniform_init(generator=generator)
    init_modules_by_type(module, [(Layer, init_layer)], fallback)

def smart_init(
        module: torch.nn.Module,
        generator: typing.Optional[torch.Generator]=None,
        fallback: typing.Callable[
            [str, torch.Tensor, typing.Optional[torch.Generator]],
            None
        ]=default_fallback,
        use_xavier_uniform_for_layers: bool=True,
        init_layer_norm: bool=True
    ) -> None:
    """Initialize all parameters in a module, using reasonable defaults for
    certain types of sub-module.

    :param module: The module.
    :param generator: Optional PyTorch RNG.
    :param fallback: A callback that will be called on all parameters not
        in a :py:class:`Layer`. See :py:func:`init_modules`.
    :param use_xavier_uniform_for_layers: Use Xavier uniform initialization for
        all :py:class:`Layer`s. Otherwise, use the fallback.
    :param init_layer_norm: For all :py:class:`~torch.nn.LayerNorm` modules,
        (re)initialize the weights to 1 and bias terms to 0. Otherwise, use the
        fallback.
    """
    callbacks = []
    if use_xavier_uniform_for_layers:
        def init_layer(name, module, generator):
            module.xavier_uniform_init(generator=generator)
        callbacks.append((Layer, init_layer))
    if init_layer_norm:
        def init_layer_norm(name, module, generator):
            if module.weight is not None:
                module.weight.data[...] = 1.0
            if module.bias is not None:
                module.bias.data[...] = 0.0
        callbacks.append((torch.nn.LayerNorm, init_layer_norm))
    init_modules_by_type(module, callbacks, fallback)

def uniform_fallback(scale):
    def fallback(name, param, generator):
        param.data.uniform_(-scale, scale, generator=generator)
    return fallback
