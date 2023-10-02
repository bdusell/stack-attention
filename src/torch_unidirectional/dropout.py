from typing import Optional

import torch

from .simple import SimpleLayerUnidirectional

class DropoutUnidirectional(SimpleLayerUnidirectional):

    def __init__(self, dropout: Optional[float]):
        super().__init__(torch.nn.Dropout(dropout) if dropout else torch.nn.Identity())
