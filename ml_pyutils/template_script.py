import argparse
import json
import random
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Template script for file processing.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input directory containing files to process."
    )
    return parser.parse_args()

def main(input: Path):
    
    pass

if __name__ == "__main__":
    args = parse_args()
    kwargs = vars(args)
    main(**kwargs)
    pass