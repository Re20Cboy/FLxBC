import numpy as np

from flxbc.data import partition_indices


def test_iid_partition_covers_training_indices_without_overlap():
    labels = np.array([0, 1] * 20)

    partitions = partition_indices(labels, num_clients=5, iid=True, seed=7)

    flattened = [idx for part in partitions.values() for idx in part]
    assert sorted(flattened) == list(range(len(labels)))
    assert len(flattened) == len(set(flattened))
    assert all(len(part) == 8 for part in partitions.values())


def test_dirichlet_partition_is_reproducible_and_non_overlapping():
    labels = np.array([0] * 24 + [1] * 24 + [2] * 24)

    first = partition_indices(labels, num_clients=6, iid=False, alpha=0.4, seed=11)
    second = partition_indices(labels, num_clients=6, iid=False, alpha=0.4, seed=11)

    assert first == second
    flattened = [idx for part in first.values() for idx in part]
    assert sorted(flattened) == list(range(len(labels)))
    assert len(flattened) == len(set(flattened))
    assert all(len(part) > 0 for part in first.values())
