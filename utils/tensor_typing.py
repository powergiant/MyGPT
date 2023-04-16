from typing import TypeVar, Generic
from typing_extensions import TypeVarTuple
from torch import Tensor

Dim = TypeVarTuple('Dim')

class TensorNamed(Tensor, Generic[*Dim]):
    pass
