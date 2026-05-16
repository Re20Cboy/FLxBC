from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ArrayDataset:
    images: np.ndarray
    labels: np.ndarray
    num_classes: int
    task: str = "binary-class"

    def subset(self, indices: list[int] | np.ndarray) -> ArrayDataset:
        selected = np.asarray(indices, dtype=np.int64)
        return ArrayDataset(
            images=self.images[selected],
            labels=self.labels[selected],
            num_classes=self.num_classes,
            task=self.task,
        )


def _flatten_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim > 1:
        labels = labels[:, 0]
    return labels.astype(np.int64)


def partition_indices(
    labels: np.ndarray,
    *,
    num_clients: int,
    iid: bool,
    alpha: float = 0.5,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split training indices into simulated hospitals.

    IID mode shuffles indices into equal shards. Non-IID mode uses a per-class
    Dirichlet draw, then repairs empty shards so every hospital can participate.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    labels = _flatten_labels(labels)
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(labels))

    if iid:
        shuffled = rng.permutation(all_indices)
        chunks = np.array_split(shuffled, num_clients)
        return {
            f"hospital_{i + 1}": sorted(chunk.astype(int).tolist())
            for i, chunk in enumerate(chunks)
        }

    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        for client_id, chunk in enumerate(np.split(class_indices, split_points)):
            partitions[client_id].extend(chunk.astype(int).tolist())

    _repair_empty_partitions(partitions, rng)
    for part in partitions:
        part.sort()
    return {f"hospital_{i + 1}": part for i, part in enumerate(partitions)}


def _repair_empty_partitions(partitions: list[list[int]], rng: np.random.Generator) -> None:
    for empty_index, part in enumerate(partitions):
        if part:
            continue
        donor_candidates = [i for i, donor in enumerate(partitions) if len(donor) > 1]
        if not donor_candidates:
            raise ValueError("not enough samples to create non-empty client partitions")
        donor_index = int(rng.choice(donor_candidates))
        moved = partitions[donor_index].pop()
        partitions[empty_index].append(moved)


def load_medical_dataset(
    *,
    dataset: str,
    seed: int,
    max_train_samples: int | None,
    max_test_samples: int | None,
    use_synthetic: bool = False,
) -> tuple[ArrayDataset, ArrayDataset, ArrayDataset]:
    if use_synthetic or dataset.lower().startswith("synthetic"):
        return make_synthetic_medical_data(
            seed=seed,
            train_samples=max_train_samples or 240,
            val_samples=max(24, min(120, (max_test_samples or 120) // 2)),
            test_samples=max_test_samples or 120,
            is_3d=not dataset.lower().endswith("2d"),
        )
    return _load_medmnist(
        dataset=dataset,
        seed=seed,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )


def _load_medmnist(
    *,
    dataset: str,
    seed: int,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> tuple[ArrayDataset, ArrayDataset, ArrayDataset]:
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:  # pragma: no cover - exercised in real setup
        raise RuntimeError(
            "medmnist is not installed. Run `make setup-ml`, "
            "or pass `--synthetic` for a smoke demo."
        ) from exc

    aliases = {
        "nodulemnist3d": "nodulemnist3d",
        "nodule": "nodulemnist3d",
        "pneumoniamnist": "pneumoniamnist",
        "pneumonia": "pneumoniamnist",
    }
    flag = aliases.get(dataset.lower(), dataset.lower())
    if flag not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset `{dataset}`")

    info: dict[str, Any] = INFO[flag]
    dataset_cls = getattr(medmnist, info["python_class"])
    train_raw = dataset_cls(split="train", download=True)
    val_raw = dataset_cls(split="val", download=True)
    test_raw = dataset_cls(split="test", download=True)
    num_classes = len(info["label"])

    train = _from_medmnist(
        train_raw, num_classes=num_classes, max_samples=max_train_samples, seed=seed
    )
    val = _from_medmnist(
        val_raw, num_classes=num_classes, max_samples=max_test_samples, seed=seed + 1
    )
    test = _from_medmnist(
        test_raw, num_classes=num_classes, max_samples=max_test_samples, seed=seed + 2
    )
    return train, val, test


def _from_medmnist(
    raw: Any, *, num_classes: int, max_samples: int | None, seed: int
) -> ArrayDataset:
    images = np.asarray(raw.imgs)
    labels = _flatten_labels(np.asarray(raw.labels))
    if max_samples is not None and len(labels) > max_samples:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(np.arange(len(labels)), size=max_samples, replace=False))
        images = images[selected]
        labels = labels[selected]
    images = images.astype(np.float32)
    if images.max() > 1:
        images /= 255.0
    return ArrayDataset(images=images, labels=labels, num_classes=num_classes)


def make_synthetic_medical_data(
    *,
    seed: int,
    train_samples: int = 240,
    val_samples: int = 80,
    test_samples: int = 80,
    is_3d: bool = True,
) -> tuple[ArrayDataset, ArrayDataset, ArrayDataset]:
    rng = np.random.default_rng(seed)
    shape = (16, 16, 16) if is_3d else (28, 28)

    def build(count: int) -> ArrayDataset:
        labels = rng.integers(0, 2, size=count, dtype=np.int64)
        images = rng.normal(0.1, 0.2, size=(count, *shape)).astype(np.float32)
        if is_3d:
            n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
            if n0:
                signals = rng.normal(0.30, 0.15, size=n0).astype(np.float32)
                images[labels == 0, 5:11, 5:11, 5:11] += signals.reshape(-1, 1, 1, 1)
            if n1:
                signals = rng.normal(0.60, 0.15, size=n1).astype(np.float32)
                images[labels == 1, 5:11, 5:11, 5:11] += signals.reshape(-1, 1, 1, 1)
        else:
            n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
            if n0:
                signals = rng.normal(0.30, 0.15, size=n0).astype(np.float32)
                images[labels == 0, 9:19, 9:19] += signals.reshape(-1, 1, 1)
            if n1:
                signals = rng.normal(0.60, 0.15, size=n1).astype(np.float32)
                images[labels == 1, 9:19, 9:19] += signals.reshape(-1, 1, 1)
        images = np.clip(images, 0.0, 1.0)
        return ArrayDataset(images=images, labels=labels, num_classes=2)

    return build(train_samples), build(val_samples), build(test_samples)
