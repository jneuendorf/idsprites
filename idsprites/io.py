import dataclasses
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, Literal
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from torch.utils.data import random_split
from tqdm.contrib.concurrent import process_map

import idsprites as ids
from idsprites import Factors
from idsprites.types import Shape, Floats, Subset

grouper = ids.ContinualBenchmark.grouper


class Args(Protocol):
    out_dir: Path
    num_tasks: int
    shapes_per_task: int
    img_size: int
    train_split: int | float
    val_split: int | float
    test_split: int | float
    factor_resolution: int
    overwrite: bool


@dataclass
class Task:
    dir: Path
    exemplars: list[Shape] = field(default_factory=list)
    shapes: list[Shape] = field(default_factory=list)

    def __post_init__(self):
        self.ensure_dirs()

    def write_shapes(self) -> None:
        np.save(self.dir / "shapes.npy", self.shapes)

    def write_exemplars(self) -> None:
        for i, exemplar in enumerate(self.exemplars):
            exemplar_img = to_image(exemplar)
            exemplar_img.save(self.exemplars_dir / f"exemplar_{i}.png")

    def write_split(
        self,
        name: Literal["train", "val", "test"],
        split: Iterable[tuple[Floats, Factors]],
    ) -> None:
        subdir = getattr(self, f"{name}_dir")
        split_factors = []
        labels = []
        for i, (image, factors) in enumerate(split):
            image: Floats
            factors: Factors

            split_factors.append(factors.replace(shape=None))
            path = subdir / f"sample_{i}.png"
            to_image(image).save(path)
            labels.append(f"{path.name} {factors.shape_id}")
        # split_factors = ids.Factors(*zip(*split_factors))
        np.savez(
            subdir / "factors.npz",
            **ids.Factors(*zip(*split_factors))._asdict(),
        )
        (subdir / "labels.txt").write_text("\n".join(labels))

    @property
    def exemplars_dir(self):
        return self.dir / "exemplars"

    @property
    def train_dir(self):
        return self.dir / "train"

    @property
    def val_dir(self):
        return self.dir / "val"

    @property
    def test_dir(self):
        return self.dir / "test"

    def ensure_dirs(self):
        self.exemplars_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# DATASET GENERATION

# TODO: Support all kwargs of 'InfiniteDSprites'
def generate_dataset(args: Args):
    dataset = ids.InfiniteDSprites(img_size=args.img_size)
    num_shapes = args.num_tasks * args.shapes_per_task
    shapes = [dataset.generate_shape() for _ in range(num_shapes)]
    exemplars = generate_exemplars(shapes, args.img_size)
    shape_ids = list(range(num_shapes))

    tasks_ids = list(range(args.num_tasks))
    task_shapes = list(grouper(shapes, args.shapes_per_task))
    task_shape_ids = list(grouper(shape_ids, args.shapes_per_task))
    task_exemplars = list(grouper(exemplars, args.shapes_per_task))

    # parallelize over tasks_ids
    process_map(
        write_task_files,
        tasks_ids,
        task_shapes,
        task_shape_ids,
        task_exemplars,
        [args] * args.num_tasks,
    )


def generate_exemplars(shapes, img_size: int):
    """Generate a batch of exemplars for training and visualization."""
    dataset = ids.InfiniteDSprites(
        img_size=img_size,
    )
    return [
        dataset.draw(
            ids.Factors(
                color=(1.0, 1.0, 1.0),
                shape=shape,
                shape_id=None,
                scale=1.0,
                orientation=0.0,
                position_x=0.5,
                position_y=0.5,
            ),
        )
        for shape in shapes
    ]


def write_task_files(task_id, task_shapes, task_shape_ids, task_exemplars, args: Args):
    """Writes all files related to task 'task_id' to 'args.out_dir'.

    Non-image files:
    - shapes.npy
    - */factors.npz
    - */labels.txt

    File structure of each task:
    task_*
    ├── exemplars
    │   └── exemplar_*.png
    ├── shapes.npy
    ├── test
    │   ├── factors.npz
    │   ├── labels.txt
    │   └── sample_*.png
    ├── train
    │   ├── factors.npz
    │   ├── labels.txt
    │   └── sample_*.png
    └── val
        ├── factors.npz
        ├── labels.txt
        └── sample_*.png
    """
    task_dir = args.out_dir / f"task_{task_id + 1}"

    if not args.overwrite and task_dir.exists():
        return

    task = Task(
        task_dir,
        exemplars=task_exemplars,
        shapes=task_shapes,
    )
    task.write_shapes()
    task.write_exemplars()

    dataset = create_continual_dsprites(args, task_shapes, task_shape_ids)
    train, val, test = random_split(
        dataset,
        lengths=[args.train_split, args.val_split, args.test_split],
    )
    task.write_split(name="train", split=train)
    task.write_split(name="val", split=val)
    task.write_split(name="test", split=test)


def create_continual_dsprites(args, task_shapes, task_shape_ids):
    """Create a dataset for a single task."""
    n = args.factor_resolution
    scale_range = np.linspace(0.5, 1.0, n)
    orientation_range = np.linspace(0, 2 * np.pi * n / (n + 1), n)
    position_x_range = np.linspace(0, 1, n)
    position_y_range = np.linspace(0, 1, n)
    return ids.ContinualDSpritesMap(
        img_size=args.img_size,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
        shapes=task_shapes,
        shape_ids=task_shape_ids,
    )


def to_image(array: np.ndarray):
    """Convert a tensor to an image."""
    array = array.transpose(1, 2, 0)
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


# ---------------------------------------------------------
# DATASET LOADING

def load_dataset(in_dir: Path):
    ...
