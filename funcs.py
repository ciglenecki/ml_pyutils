import argparse
import inspect
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TypeVar

import numpy as np
import torch
import yaml

T = TypeVar("T")


def get_image_files(
    directory: Path,
    extensions: Optional[set[str]] = None,
) -> list[Path]:
    """Lists image file paths recursively in a directory.

    Args:
        directory: The root directory to search for image files.
        extensions: A list of file extensions to filter by (default: common image formats).

    Returns:
        A list of file paths matching the given extensions.

    Raises:
        ValueError: If the directory does not exist or is not a directory.
    """
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    # convert all extensions to lowercase for case-insensitive comparison
    extensions = {e.lower() for e in extensions}
    paths = [file for file in directory.rglob("*") if file.suffix.lower() in extensions]
    return paths


def min_max_scale(t: torch.Tensor | np.ndarray):
    return (t - t.min()) / (t.max() - t.min())


def dict_without_keys(d: dict, keys: list[str]):
    return {x: d[x] for x in d if x not in keys}


def dict_with_keys(d: dict, keys: list[str]):
    return {x: d[x] for x in d if x in keys}


def tensor_sum_of_elements_to_one(tensor: torch.Tensor, dim):
    """Scales elements of the tensor so that the sum is 1."""
    return tensor / torch.sum(tensor, dim=dim, keepdim=True)


def isfloat(x: str):
    try:
        float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(x: str):
    try:
        a = float(x)
        b = int(x)
    except (TypeError, ValueError):
        return False
    else:
        return a == b


def get_timestamp(format="%m-%d-%H-%M-%S"):
    return datetime.today().strftime(format)


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.mps.seed(seed)


def npy_to_image(npy):
    return npy.transpose((1, 2, 0))


def validate_image(
    image: np.ndarray | torch.Tensor,
    allowed_num_channels: tuple[int, ...] = (0, 1, 3, 4),
) -> None:
    """Validates that the input image has an appropriate shape and number of channels.

    Args:
        image: The input image, either as a NumPy array (..., H, W, C) or a Torch tensor (..., C, H, W).
        allowed_num_channels: A tuple of allowed channel numbers (default: (0, 1, 3, 4)).

    Raises:
        ValueError: If the image format or number of channels is invalid.
    """
    HELP_TEXT = (
        f"Input image must be a NumPy array (..., H, W, C) or a torch tensor (..., C, H, W). "
        f"Number of channels (C) must be one of {allowed_num_channels}."
    )

    if isinstance(image, np.ndarray):
        if image.ndim < 2:
            raise ValueError(HELP_TEXT)
        if image.ndim == 2 and 0 in allowed_num_channels:
            num_channels = 0  # Grayscale image without explicit channel dimension
        elif image.ndim == 3:
            _, _, num_channels = image.shape  # (H, W, C)
        else:
            raise ValueError(HELP_TEXT)
    elif isinstance(image, torch.Tensor):
        if image.ndim not in {3, 4}:  # Must be (C, H, W) or (B, C, H, W)
            raise ValueError(HELP_TEXT)
        num_channels = image.shape[-3]  # Extract channel dimension (C)
    else:
        raise TypeError("Image must be either a NumPy array or a Torch tensor.")

    if num_channels not in allowed_num_channels:
        raise ValueError(HELP_TEXT)


def is_between_0_1(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} not a floating-point literal")
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} not in range [0.0, 1.0]")
    return x


def is_positive_int(value):
    int_value = int(value)
    if int_value < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return int_value


def is_valid_dir(arg):
    if not os.path.isdir(arg):
        raise argparse.ArgumentError(arg, "Argument should be a path to directory")
    return arg


def add_prefix_to_keys(
    dict: dict,
    prefix: str,
    filter_fn: Optional[Callable] = None,
) -> dict:
    """
    Example:
        dict = {"a": 1, "b": 2}
        prefix = "text_"
        returns {"text_a": 1, "text_b": 2}

    Example:
        dict = {"abra": 1, "abrakadabra": 2, "nothing": 3}
        prefix = "text_"
        filter = lambda x: x.startswith("abra")
        returns {"text_abra": 1, "text_abrakadabra": 2, "nothing": 3}
    """
    return {
        (k if filter_fn is not None and filter_fn(k) else prefix + k): v
        for k, v in dict.items()
    }


def flatten(list):
    """
    Example:
        list = [[1, 2], [3, 4]]
        returns [1, 2, 3, 4]
    """
    return [item for sublist in list for item in sublist]


def to_yaml(data):
    return yaml.dump(data, allow_unicode=True, default_flow_style=False)


def to_json(data):
    return json.dumps(data, indent=4, default=str)


def save_yaml(data: object, path: Path):
    with open(path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def save_json(data: object, path: Path):
    with open(path, "w") as outfile:
        json.dump(data, outfile, indent=4, default=str)


def function_kwargs(func):
    """Returns function keyword arguments."""
    return inspect.getfullargspec(func)


def all_args(cls):
    """
    A decorator function that checks if all arguments are provided by the user when instantiating an object.
    Args:
        cls: The class to be wrapped.
    Returns:
        The wrapped class with argument checking.
    Raises:
        TypeError: If any of the required arguments are missing.
    Notes:
        - Required arguments are those that do not have a default value
    """

    def wrapper(*args, **kwargs):
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        if cls.__init__.__qualname__ != object.__init__.__qualname__:
            parent_sig = inspect.signature(super(cls).__init__)
            parent_params = parent_sig.parameters
            parent_args = parent_params.keys() - params.keys()
        else:
            parent_args = set()

        user_args = {**dict(zip(params.keys(), args)), **kwargs}
        missing_args = (
            set(params.keys()).union(parent_args)
            - set(user_args.keys())
            - {"self", "args", "kwargs"}
        )

        if missing_args:
            missing_args_list = ", ".join(missing_args)
            raise TypeError(f"Missing required argument(s): {missing_args_list}")

        return cls(*args, **kwargs)

    return wrapper


nato_alphabet = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliett",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
]

adjectives = [
    "agile",
    "ample",
    "avid",
    "awed",
    "best",
    "bonny",
    "brave",
    "brisk",
    "calm",
    "clean",
    "clear",
    "comfy",
    "cool",
    "cozy",
    "crisp",
    "cute",
    "deft",
    "eager",
    "eased",
    "easy",
    "elite",
    "fair",
    "famed",
    "fancy",
    "fast",
    "fiery",
    "fine",
    "finer",
    "fond",
    "free",
    "freed",
    "fresh",
    "fun",
    "funny",
    "glad",
    "gold",
    "good",
    "grand",
    "great",
    "hale",
    "handy",
    "happy",
    "hardy",
    "holy",
    "hot",
    "ideal",
    "jolly",
    "keen",
    "lean",
    "like",
    "liked",
    "loved",
    "loyal",
    "lucid",
    "lucky",
    "lush",
    "magic",
    "merry",
    "neat",
    "nice",
    "nicer",
    "noble",
    "plush",
    "prize",
    "proud",
    "pure",
    "quiet",
    "rapid",
    "rapt",
    "ready",
    "regal",
    "rich",
    "right",
    "roomy",
    "rosy",
    "safe",
    "sane",
    "sexy",
    "sharp",
    "shiny",
    "sleek",
    "slick",
    "smart",
    "soft",
    "solid",
    "suave",
    "super",
    "swank",
    "sweet",
    "swift",
    "tidy",
    "top",
    "tough",
    "vivid",
    "warm",
    "well",
    "wise",
    "witty",
    "worth",
    "young",
]


def random_codeword(num_numbers=2) -> str:
    """Examples: YoungAlpha55, WiseZulu17, CozyBravo53"""
    name = f"{random.choice(adjectives).upper()}_{random.choice(nato_alphabet).upper()}"
    if num_numbers == 0:
        return name
    max_num = 10**num_numbers
    number = str(random.randint(0, max_num - 1))
    return f"{name}{number.zfill(num_numbers)}"
