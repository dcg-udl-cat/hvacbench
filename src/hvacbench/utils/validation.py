from typing import Tuple
from hvacbench.schemas import FloatArray

def validate_shape(arr: FloatArray, expected_shape: Tuple[int, ...], name: str) -> None:
    """Validates the shape of a numpy array.
    
    Args:
        arr: The array to validate.
        expected_shape: The expected shape tuple.
        name: Name of the array for error messages.
        
    Raises:
        ValueError: If shape is invalid.
    """
    if arr.shape != expected_shape:
        raise ValueError(
            f"Invalid shape for {name}. "
            f"Expected {expected_shape}, got {arr.shape}."
        )
