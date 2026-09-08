"""Core data structures and validation for ARC tasks.

This module defines the primary domain models for representing ARC (Abstraction
and Reasoning Corpus) grids, input/output pairs, and complete tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


ARC_COLORS: dict[int, str] = {
    0: "#000000",  # Black / Background
    1: "#0074D9",  # Blue
    2: "#FF4136",  # Red
    3: "#2ECC40",  # Green
    4: "#FFDC00",  # Yellow
    5: "#AAAAAA",  # Grey
    6: "#F012BE",  # Magenta / Fuchsia
    7: "#FF851B",  # Orange
    8: "#7FDBFF",  # Cyan / Teal
    9: "#870C25",  # Maroon / Brown
}

COLOR_NAMES: dict[int, str] = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon",
}

MIN_DIM: int = 1
MAX_DIM: int = 30
VALID_COLOR_VALUES: set[int] = set(range(10))


class ARCValidationError(ValueError):
    """Base exception for ARC data validation errors."""
    pass


class InvalidGridError(ARCValidationError):
    """Raised when a grid is malformed, non-rectangular, or contains invalid values."""
    pass


class MalformedTaskError(ARCValidationError):
    """Raised when an ARC task structure is invalid or missing required fields."""
    pass


@dataclass(frozen=True)
class Grid:
    """Immutable representation of a 2D ARC grid.
    
    Attributes:
        cells: Tuple of tuples containing integer color values (0-9).
        height: Number of rows (1-30).
        width: Number of columns (1-30).
    """
    cells: tuple[tuple[int, ...], ...]
    height: int = field(init=False)
    width: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.cells:
            raise InvalidGridError("Grid cannot be empty (0 rows).")
        
        num_rows = len(self.cells)
        if not (MIN_DIM <= num_rows <= MAX_DIM):
            raise InvalidGridError(
                f"Grid height {num_rows} out of bounds [{MIN_DIM}, {MAX_DIM}]."
            )
        
        num_cols = len(self.cells[0])
        if not (MIN_DIM <= num_cols <= MAX_DIM):
            raise InvalidGridError(
                f"Grid width {num_cols} out of bounds [{MIN_DIM}, {MAX_DIM}]."
            )

        for r_idx, row in enumerate(self.cells):
            if len(row) != num_cols:
                raise InvalidGridError(
                    f"Grid is not rectangular: row 0 has {num_cols} columns, "
                    f"but row {r_idx} has {len(row)} columns."
                )
            for c_idx, val in enumerate(row):
                if not isinstance(val, int) or isinstance(val, bool):
                    raise InvalidGridError(
                        f"Grid cell at ({r_idx}, {c_idx}) has non-integer value: {val!r}"
                    )
                if val not in VALID_COLOR_VALUES:
                    raise InvalidGridError(
                        f"Grid cell at ({r_idx}, {c_idx}) has invalid color value {val}. "
                        f"Must be an integer between 0 and 9."
                    )

        object.__setattr__(self, "height", num_rows)
        object.__setattr__(self, "width", num_cols)

    @classmethod
    def from_list(cls, raw_grid: Sequence[Sequence[Any]]) -> Grid:
        """Create a Grid instance from a nested list or sequence of ints."""
        if not raw_grid:
            raise InvalidGridError("Raw grid list cannot be empty.")
        tuple_grid = tuple(tuple(row) for row in raw_grid)
        return cls(cells=tuple_grid)

    @property
    def shape(self) -> tuple[int, int]:
        """Return (height, width) dimensions."""
        return (self.height, self.width)

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return (height, width) dimensions."""
        return self.shape

    @property
    def size(self) -> int:
        """Return total number of cells."""
        return self.height * self.width

    def to_list(self) -> list[list[int]]:
        """Convert grid back to standard nested list format."""
        return [list(row) for row in self.cells]

    def to_numpy(self) -> Any:
        """Convert grid to a NumPy 2D array of uint8."""
        import numpy as np
        return np.array(self.cells, dtype=np.uint8)

    def __getitem__(self, idx: int) -> tuple[int, ...]:
        return self.cells[idx]

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return iter(self.cells)

    def __len__(self) -> int:
        return self.height

    def get(self, row: int, col: int) -> int:
        """Get color value at (row, col)."""
        return self.cells[row][col]

    @property
    def unique_colors(self) -> set[int]:
        """Return set of distinct colors present in the grid."""
        colors = set()
        for row in self.cells:
            colors.update(row)
        return colors

    def color_counts(self) -> dict[int, int]:
        """Return frequency of each color present in the grid."""
        counts: dict[int, int] = {}
        for row in self.cells:
            for val in row:
                counts[val] = counts.get(val, 0) + 1
        return counts

    @property
    def background_color(self) -> int:
        """Default background color is 0, or the most frequent color."""
        counts = self.color_counts()
        return max(counts.keys(), key=lambda k: counts[k])

    def to_ascii(self) -> str:
        """Render a readable ASCII representation of the grid."""
        lines = []
        border = "+" + "-" * (self.width * 2 + 1) + "+"
        lines.append(border)
        for row in self.cells:
            row_str = " ".join(str(c) if c != 0 else "." for c in row)
            lines.append(f"| {row_str} |")
        lines.append(border)
        return "\n".join(lines)


@dataclass(frozen=True)
class TrainingPair:
    """Represents a demonstration input-output pair in an ARC task."""
    input: Grid
    output: Grid

    def __post_init__(self) -> None:
        if not isinstance(self.input, Grid):
            raise TypeError(f"input must be Grid instance, got {type(self.input)}")
        if not isinstance(self.output, Grid):
            raise TypeError(f"output must be Grid instance, got {type(self.output)}")

    @property
    def is_size_preserving(self) -> bool:
        """Check if input and output grids share identical dimensions."""
        return self.input.shape == self.output.shape

    @property
    def input_shape(self) -> tuple[int, int]:
        return self.input.shape

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.output.shape

    def to_dict(self) -> dict[str, list[list[int]]]:
        return {
            "input": self.input.to_list(),
            "output": self.output.to_list(),
        }


@dataclass(frozen=True)
class TestPair:
    """Represents a test input (and optional ground-truth output) in an ARC task."""
    __test__ = False
    input: Grid
    output: Grid | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input, Grid):
            raise TypeError(f"input must be Grid instance, got {type(self.input)}")
        if self.output is not None and not isinstance(self.output, Grid):
            raise TypeError(f"output must be Grid or None, got {type(self.output)}")

    @property
    def has_output(self) -> bool:
        """Check whether test pair includes ground truth output."""
        return self.output is not None

    @property
    def is_size_preserving(self) -> bool | None:
        if self.output is None:
            return None
        return self.input.shape == self.output.shape

    def to_dict(self) -> dict[str, list[list[int]]]:
        res: dict[str, list[list[int]]] = {"input": self.input.to_list()}
        if self.output is not None:
            res["output"] = self.output.to_list()
        return res


@dataclass
class ARCTask:
    """Complete representation of an ARC task with training and test pairs.
    
    Attributes:
        task_id: Unique identifier for the task (e.g. '007bbfb7').
        train: List of TrainingPair instances (typically 2-5).
        test: List of TestPair instances (typically 1-2).
        metadata: Optional dictionary with task source, file path, etc.
    """
    task_id: str
    train: list[TrainingPair]
    test: list[TestPair]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.train:
            raise MalformedTaskError(f"Task '{self.task_id}' has no training examples.")
        if not self.test:
            raise MalformedTaskError(f"Task '{self.task_id}' has no test examples.")

    @property
    def num_train(self) -> int:
        """Number of training demonstration pairs."""
        return len(self.train)

    @property
    def num_test(self) -> int:
        """Number of test evaluation pairs."""
        return len(self.test)

    @property
    def all_train_inputs(self) -> list[Grid]:
        return [p.input for p in self.train]

    @property
    def all_train_outputs(self) -> list[Grid]:
        return [p.output for p in self.train]

    @property
    def all_test_inputs(self) -> list[Grid]:
        return [p.input for p in self.test]

    @property
    def all_test_outputs(self) -> list[Grid]:
        return [p.output for p in self.test if p.output is not None]

    @property
    def all_grids(self) -> list[Grid]:
        """Return all grids present in the task."""
        grids = []
        for p in self.train:
            grids.extend([p.input, p.output])
        for p in self.test:
            grids.append(p.input)
            if p.output is not None:
                grids.append(p.output)
        return grids

    @property
    def unique_colors(self) -> set[int]:
        """Return set of all unique colors used across all task grids."""
        colors = set()
        for g in self.all_grids:
            colors.update(g.unique_colors)
        return colors

    @property
    def is_size_preserving(self) -> bool:
        """Check if all training pairs preserve grid dimensions."""
        return all(p.is_size_preserving for p in self.train)

    @property
    def has_fixed_output_dim(self) -> bool:
        """Check if all training outputs share identical dimensions."""
        out_shapes = {p.output.shape for p in self.train}
        return len(out_shapes) == 1

    def to_dict(self) -> dict[str, Any]:
        """Convert task back to standard ARC JSON dict format."""
        return {
            "train": [p.to_dict() for p in self.train],
            "test": [p.to_dict() for p in self.test],
        }

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics for this task."""
        train_in_shapes = [p.input.shape for p in self.train]
        train_out_shapes = [p.output.shape for p in self.train]
        test_in_shapes = [p.input.shape for p in self.test]
        test_out_shapes = [p.output.shape for p in self.test if p.output is not None]
        
        return {
            "task_id": self.task_id,
            "num_train": self.num_train,
            "num_test": self.num_test,
            "train_input_shapes": train_in_shapes,
            "train_output_shapes": train_out_shapes,
            "test_input_shapes": test_in_shapes,
            "test_output_shapes": test_out_shapes,
            "is_size_preserving": self.is_size_preserving,
            "colors_used": sorted(list(self.unique_colors)),
            "num_colors": len(self.unique_colors),
        }
