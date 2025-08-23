"""
before:
my_dir
├── test0
├── test1
├── test2
├── test3
├── test4
├── test5
├── test6
├── test7
├── test8
├── test9
├── test10
├── test11
├── test12
├── test13
└── test14

after:
my_dir
├── 0
│   ├── 0
│   │   ├── test0
│   │   ├── test1
│   │   ├── test2
│   │   ├── test3
│   │   └── test4
│   └── 1
│       ├── test5
│       ├── test6
│       ├── test7
│       ├── test8
│       └── test9
├── 1
│   ├── 0
│   │   ├── test10
│   │   ├── test11
│   │   ├── test12
│   │   ├── test13
│   │   └── test14
"""

import argparse
import math
import shutil
from pathlib import Path
from typing import Callable


def calculate_depth(total_items: int, max_per_dir: int) -> int:
    """Calculate the depth required so that max_per_dir^depth >= total_items."""
    if total_items <= 0:
        return 0
    return math.ceil(math.log(total_items, max_per_dir)) - 1


def validate_main_directory(main_dir: Path) -> None:
    """Ensure main directory exists and does not contain integer-named entries."""
    if not main_dir.is_dir():
        raise ValueError(f"Provided path '{main_dir}' is not a valid directory.")

    for entry in main_dir.iterdir():
        if entry.name.isdigit():
            raise RuntimeError(
                f"Integer-named entry '{entry.name}' found in main directory. Aborting!"
            )


def get_target_dir(index: int, depth: int, max_per_dir: int, main_dir: Path) -> Path:
    """
    Compute the hierarchical target directory for a given item index based on the specified depth and maximum items per directory.

    How it works:
        1. Each `index` (0-based) represents an item in the sorted list of items to be moved.
        2. Items are grouped into "buckets" of size `max_per_dir`. For example,
           if `max_per_dir=10`, then:
               - Items 0–9 go into bucket 0
               - Items 10–19 go into bucket 1
               - Items 20–29 go into bucket 2, and so on.
        3. Each bucket number is converted to -`max_per_dir` with a fixed number of digits equal to `depth`. This base conversion determines the nested directory path.
           For example:
               - bucket=0  → [0, 0, 0] (for depth=3)
               - bucket=14 → [0, 1, 4] (for depth=3)
               - bucket=372 → [3, 7, 2]
        4. The resulting list of digits becomes the directory structure:
           `main_dir/0/1/4` for bucket 14.

    Args:
        index: Position of the item in the list (0-based).
        depth: The number of directory levels to create.
        max_per_dir: Maximum number of items allowed in a single directory.
        main_dir: The root directory under which the structure will be created.

    Returns:
        A Path object representing the target directory for the given index.

    Example:
        >>> get_target_dir(index=25, depth=3, max_per_dir=10, main_dir=Path("/data"))
        PosixPath('/data/0/0/2')
    """
    bucket = index // max_per_dir  # group items into chunks of max_per_dir
    # convert bucket to base max_per_dir representation with 'depth' digits
    parts = [
        str((bucket // (max_per_dir**i)) % max_per_dir) for i in reversed(range(depth))
    ]
    return main_dir.joinpath(*parts)


def reorganize_directory(
    main_path: str,
    max_per_dir: int = 10,
    filter_func: Callable[[Path], bool] | None = None,
) -> None:
    """
    Reorganize items in the specified directory into a nested structure.

    Args:
        main_path: Path to the main directory containing items to reorganize.
        max_per_dir: Maximum number of items allowed in a single directory.
        filter_func: Optional callable that returns True for items to be reorganized.

    Raises:
        ValueError: If main_path is not a directory.
        RuntimeError: If integer-named entries are found in the main directory.
    """
    assert max_per_dir > 1, "max_per_dir must be greater than 1"
    main_dir = Path(main_path)
    validate_main_directory(main_dir)

    filter_func = filter_func or (lambda _: True)

    items: list[Path] = sorted(
        [p for p in main_dir.iterdir() if filter_func(p)], key=lambda x: x.name
    )
    total_items = len(items)

    if total_items == 0:
        print("No matching items found.")
        return

    depth = calculate_depth(total_items, max_per_dir)
    if depth < 1:
        print("No reorganization needqed.")
        return

    print(f"Calculated depth: {depth}")

    for idx, item in enumerate(items):
        target_dir = get_target_dir(
            index=idx, depth=depth, max_per_dir=max_per_dir, main_dir=main_dir
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item), str(target_dir))
        print(f"Moved '{item.name}' -> '{target_dir / item.name}'")

    print("Reorganization complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reorganize files and directories into a nested structure."
    )
    parser.add_argument(
        "--main_path", required=True, help="Path to the main directory to reorganize."
    )
    parser.add_argument(
        "--max_per_dir",
        type=int,
        default=10,
        help="Maximum number of items per directory.",
    )
    parser.add_argument(
        "--only_dirs", action="store_true", help="Only move directories, skip files."
    )
    parser.add_argument(
        "--only_files", action="store_true", help="Only move files, skip directories."
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=None,
        help="Only move files with this extension (e.g., .txt).",
    )
    return parser.parse_args()


def build_filter_func(
    only_dirs: bool, only_files: bool, extension: str | None
) -> Callable[[Path], bool]:
    def filter_func(path: Path) -> bool:
        if only_dirs and not path.is_dir():
            return False
        if only_files and not path.is_file():
            return False
        if extension and path.is_file() and path.suffix != extension:
            return False
        return True

    return filter_func


def main() -> None:
    args = parse_args()
    filter_func = build_filter_func(
        only_dirs=args.only_dirs,
        only_files=args.only_files,
        extension=args.extension,
    )
    reorganize_directory(
        main_path=args.main_path,
        max_per_dir=args.max_per_dir,
        filter_func=filter_func,
    )


if __name__ == "__main__":
    main()
