from typing import Union

import torch

from .unidirectional import ForwardResult

def unwrap_output_tensor(result: ForwardResult) -> Union[torch.Tensor, ForwardResult]:
    if result.extra_outputs or result.state is not None:
        return result
    else:
        return result.output

def ensure_is_forward_result(x: Union[torch.Tensor, ForwardResult]) -> ForwardResult:
    if isinstance(x, ForwardResult):
        return x
    else:
        return ForwardResult(x, (), None)
