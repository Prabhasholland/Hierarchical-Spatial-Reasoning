"""Base class for ARC solvers."""

from abc import ABC, abstractmethod
from typing import Any


Grid = list[list[int]]
Task = dict[str, Any]


class BaseSolver(ABC):
    """Abstract base class for ARC task solvers.
    
    All solvers should inherit from this class and implement
    the solve() method.
    """
    
    @abstractmethod
    def solve(self, task: Task) -> list[Grid]:
        """Solve an ARC task.
        
        Args:
            task: ARC task dictionary with 'train' and 'test' keys.
            
        Returns:
            List of predicted output grids, one per test input.
        """
        ...
    
    def name(self) -> str:
        """Return the name of this solver."""
        return self.__class__.__name__
