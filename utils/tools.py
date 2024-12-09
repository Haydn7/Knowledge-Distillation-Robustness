import torch.nn as nn
from typing import Optional


def decode_unit_string(uint: str):
    match uint.lower():
        case "kb":
            return 1024
        case "mb":
            return 1024 * 1024
        case "gb":
            return 1024 ** 3
        case _:
            return 1


def calculate_model_size(model: nn.Module, unit: Optional[str] = "MB") -> float:
    """
    Calculates the total memory size of all parameters in the PyTorch model.
    :param unit:    B, KB, MB, GB
    :return:        Total memory size
    """
    return sum(param.numel() * param.element_size() for param in model.parameters()) / decode_unit_string(unit)


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def parameter_item_count(model: nn.Module) -> int:
    return sum(1 for _ in model.parameters())


def round_up_to_power_of_2(n: int) -> int:
    """
    Rounds an integer up to the next power of 2.

    Args:
        :param n    The input integer.

    Returns:        The next power of 2 or n if n is already a power of 2.
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")

    # If n is already a power of 1 return n, otherwise compute the next power of 2
    if (n & (n - 1)) == 0:
        return n
    return 1 << (n - 1).bit_length()