import argparse
import inspect
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from types import UnionType
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import yaml


def is_power_of_two(n: int) -> bool:
    """
    Check if a given number is a power of two.
        AND operation between two consecutive numbers should be 0
        because the binary representation of a power of 2 has only one 1
        when we subtract 1 from it, we get a number with all 1s
    """
    return n & (n - 1) == 0 and n != 0


def find_closest_multiple(
    target: float, multiple_of: int, direction: Literal["closest", "smaller", "larger"]
) -> int:
    """
    Finds the closest integer number to a given target that is a multiple of `n`.

    The `direction` parameter specifies whether to find the closest multiple that is smaller,
    larger, or simply the closest (irrespective of whether it is smaller or larger).

    Args:
        target: The target number to find the closest multiple for.
        n: The number whose multiple is to be found.
        direction (Literal["closest", "smaller", "larger"]): The direction to consider:
            - "closest": Finds the closest multiple of `n` to the target.
            - "smaller": Finds the closest multiple of `n` that is smaller than or equal to the target.
            - "larger": Finds the closest multiple of `n` that is larger than or equal to the target.

    Returns:
        int: The closest multiple of `n` to the target based on the specified direction.
    """

    if multiple_of == 0:
        raise ValueError("n must be a non-zero integer.")
    if direction == "closest":
        return round(target / multiple_of) * multiple_of
    elif direction == "smaller":
        return math.floor(target / multiple_of) * multiple_of

    elif direction == "larger":
        return math.ceil(target / multiple_of) * multiple_of
    else:
        raise ValueError(
            "Invalid direction. Choose from 'closest', 'smaller', or 'larger'."
        )


def is_error_or_caused_by(
    error: BaseException,
    error_type: type | UnionType | tuple[type[Exception], ...],
) -> bool:
    """
    Check if the given error is an instance of the specified type
    or if it was caused/reraised from an error of the specified type.

    Args:
        error (BaseException): The exception to check.
        error_type (type): The exception type to check against.

    Returns:
        bool: True if the error is an instance or was caused by the given type.
    """
    current_error = error
    while current_error:
        if isinstance(current_error, error_type):
            return True
        # Check the cause (explicitly reraised exception)
        if current_error.__cause__:
            current_error = current_error.__cause__
        # Check the context (implicitly reraised exception)
        elif current_error.__context__:
            current_error = current_error.__context__
        else:
            break
    return False


def add_note(e: Exception, note: str):
    """
    Adds a note to the exception message.
    """
    if len(e.args) >= 1:
        message = f"{e.args[0]}\n{note}"
        e.args = (message,) + e.args[1:]
    else:
        e.args = (note,)
    return e


def add_context(e: Exception, context: object):
    """
    Adds context to an exception by appending a JSON-encoded context string to the exception's message.
    """
    json_str = json.dumps(context, sort_keys=True, default=str)
    message = f"\n{json_str}"
    return add_note(e, message)


def get_file_size_mb(file_path: str | Path) -> float:
    """
    file_path: str, Path to the file
    Returns the size of the file in megabytes
    """
    return os.path.getsize(file_path) / float(1024 * 1024)


def ensure_json_obj(obj: Any) -> Any:
    """
    transforms the object so it can be serialized to json.
    e.g. datetime objects are converted to strings
    e.g. path objects are converted to strings ...
    """
    return json.loads(json.dumps(obj, default=str))


def get_files(
    directory: Path,
    extensions: set[str],
) -> list[Path]:
    """Get file paths recursively in a directory.

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

    # convert all extensions to lowercase for case-insensitive comparison
    extensions = {e.lower() for e in extensions}
    paths = [file for file in directory.rglob("*") if file.suffix.lower() in extensions]
    return paths


def dict_without_keys(d: dict, keys: list[str]):
    return {x: d[x] for x in d if x not in keys}


def dict_with_keys(d: dict, keys: list[str]):
    return {x: d[x] for x in d if x in keys}


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
    torch.mps.manual_seed(seed)


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


class HumanBytes:
    """
    Utility class to format bytes into human readable strings.
    Uses metric (SI) units by default (KB, MB, GB) rather than binary units (KiB, MiB, GiB).
    """

    METRIC_LABELS = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    BINARY_LABELS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]
    METRIC_UNIT = 1000.0
    BINARY_UNIT = 1024.0

    @staticmethod
    def format(num_bytes: int | float, metric: bool = True, precision: int = 1) -> str:
        """
        Format bytes into human readable string.

        Args:
            num_bytes: Number of bytes to format
            metric: If True, use metric (SI) units (KB, MB, GB), else use binary units (KiB, MiB, GiB)
            precision: Number of decimal places to show

        Returns:
            Formatted string like "1.5 MB" or "2 GiB"
        """
        unit = HumanBytes.METRIC_UNIT if metric else HumanBytes.BINARY_UNIT
        labels = HumanBytes.METRIC_LABELS if metric else HumanBytes.BINARY_LABELS

        if num_bytes < unit:
            return f"{num_bytes} {labels[0]}"

        exponent = min(int(math.log(num_bytes, unit)), len(labels) - 1)
        quotient = float(num_bytes) / (unit**exponent)

        if quotient.is_integer():
            precision = 0

        return f"{quotient:.{precision}f} {labels[exponent]}"


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
    """
    Examples: BraveMike09, NobleWhiskey17, CuteBravo53

    Might be useful for naming your children or pets! 👶🐶
    """
    name = f"{random.choice(adjectives).upper()}_{random.choice(nato_alphabet).upper()}"
    if num_numbers == 0:
        return name
    max_num = 10**num_numbers
    number = str(random.randint(0, max_num - 1))
    return f"{name}{number.zfill(num_numbers)}"
