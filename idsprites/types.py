from collections.abc import Sequence
from typing import Generic, TypeVar

from numpy import typing as npt
from torch.utils.data import Subset as TorchSubset, Dataset

D = TypeVar('D', bound=Dataset)
T_co = TypeVar('T_co', covariant=True)

Floats = npt.NDArray[float] | list[float]
Shape = npt.NDArray[float]
Exemplar = npt.NDArray[float]


class Subset(TorchSubset, Generic[D]):
    dataset: D

    def __init__(self, dataset: D, indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)
