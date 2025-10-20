import argparse
import os
import random
from typing import List


def shuffle_file_lines_preserve_bytes(input_file: str, output_file: str) -> int:
    """
    Shuffle CSV rows while preserving the file content exactly (quotes, spaces, line endings).
    Assumes the first line is a header and keeps it at the top.

    Returns:
        Number of shuffled data rows.
    """
    with open(input_file, 'rb') as f:
        lines: List[bytes] = f.readlines()

    if not lines:
        # Empty file -> just write nothing
        with open(output_file, 'wb') as out:
            pass
        return 0

    header = lines[:1]
    rows = lines[1:]

    random.shuffle(rows)

    with open(output_file, 'wb') as out:
        out.writelines(header + rows)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle CSV rows without changing content/quoting."
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output CSV file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    if args.seed is not None:
        random.seed(args.seed)

    count = shuffle_file_lines_preserve_bytes(args.input_file, args.output)
    print(f"Shuffled {count} rows from '{args.input_file}' to '{args.output}'")


if __name__ == "__main__":
    main()