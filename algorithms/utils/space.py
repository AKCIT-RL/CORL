import json
import gymnasium as gym
import numpy as np
from minari.serialization import deserialize_space, serialize_space


class NumpySpace(gym.spaces.Space[np.ndarray]):
    def __init__(self, shape: tuple[int, ...], dtype: np.dtype, seed=None):
        super().__init__(shape, dtype, seed)

    def sample(self, mask=None) -> np.ndarray:
        return np.zeros(self.shape, dtype=self.dtype)

    def contains(self, x) -> bool:
        # Check type first is slightly safer
        return (
            isinstance(x, np.ndarray)
            and x.shape[-1] == self.shape[-1]
            and x.dtype == self.dtype
        )

    def __repr__(self) -> str:
        return f"NumpySpace(shape={self.shape}, dtype={self.dtype})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, NumpySpace)
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def seed(self, seed=None):
        super().seed(seed)

    def to_jsonable(self) -> dict:
        """Convert the space to a JSON serializable dictionary."""
        # This matches the structure our previous serialize function created
        return {
            "__type__": self.__class__.__name__,  # Use class name dynamically
            "shape": list(self.shape),
            "dtype": self.dtype.str,
        }

    @classmethod
    def from_jsonable(cls, jsonable: dict) -> "NumpySpace":
        """Create a NumpySpace instance from a JSON dictionary."""
        # This matches the structure our previous deserialize function used
        shape = tuple(jsonable["shape"])
        dtype = np.dtype(jsonable["dtype"])
        return cls(shape=shape, dtype=dtype)


@serialize_space.register(NumpySpace)
def serialize_space(space: NumpySpace, to_string=True):
    result = {}
    result["type"] = "NumpySpace"
    result["shape"] = list(space.shape)
    result["dtype"] = str(space.dtype)

    if to_string:
        result = json.dumps(result)
    return result


@deserialize_space.register("NumpySpace")
def deserialize_space(space_dict) -> NumpySpace:
    assert space_dict["type"] == "NumpySpace"
    shape = tuple(space_dict["shape"])
    dtype = np.dtype(space_dict["dtype"])
    return NumpySpace(shape=shape, dtype=dtype)
