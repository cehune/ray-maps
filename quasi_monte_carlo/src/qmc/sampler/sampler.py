"""
interface for all samplers
"""

from abc import ABC, abstractmethod
class Sampler(ABC):
    @abstractmethod
    def setup_path(self, px, py, sample_index):
        # expect to just save the values and reset the dimension
        ...
    
    def next_1d(self) -> float:
        # should return 1 numbers mapped [0, 1)
        ...
    
    def next_2d(self) -> float:
        # should return 2 numbers mapped [0, 1)
        ...