import argparse
from pathlib import Path

from idsprites.io import Args, generate_dataset


class Namespace(Args):
    """For typing, we use a custom namespace for the argument parser
    that complies with 'Args'.
    """
    pass


def int_or_float(value):
    """Convert a string to an int or float."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError as err:
            raise argparse.ArgumentTypeError(
                f"{value} is not a valid int or float"
            ) from err


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=root / "data")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--shapes_per_task", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--train_split", type=int_or_float, default=0.98)
    parser.add_argument("--val_split", type=int_or_float, default=0.01)
    parser.add_argument("--test_split", type=int_or_float, default=0.01)
    parser.add_argument("--factor_resolution", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(namespace=Namespace)

    generate_dataset(args)
