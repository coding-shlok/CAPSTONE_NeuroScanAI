from ml.training.dataset import PerDatasetBatchSampler


def test_batches_never_mix_datasets():
    dataset_ids = ["epilepsy"] * 10 + ["adhd"] * 7 + ["mci"] * 5
    sampler = PerDatasetBatchSampler(dataset_ids, batch_size=4, shuffle=True, seed=0)

    all_indices_seen = set()
    for batch in sampler:
        batch_dataset_ids = {dataset_ids[i] for i in batch}
        assert len(batch_dataset_ids) == 1, "every batch must be single-dataset"
        all_indices_seen.update(batch)

    assert all_indices_seen == set(range(len(dataset_ids)))


def test_sampler_length_matches_batch_count():
    dataset_ids = ["epilepsy"] * 10 + ["adhd"] * 7
    sampler = PerDatasetBatchSampler(dataset_ids, batch_size=4, shuffle=False)
    # Balanced-cycling, not a flat sum: each dataset is cycled up to the
    # largest dataset's own batch count (target = max(ceil(10/4), ceil(7/4))
    # = max(3, 2) = 3), then summed across the 2 datasets -> 3 * 2 = 6.
    assert len(sampler) == 6
    assert len(list(sampler)) == 6


def test_shuffle_is_seed_reproducible():
    dataset_ids = ["epilepsy"] * 10 + ["adhd"] * 7
    sampler_a = PerDatasetBatchSampler(dataset_ids, batch_size=4, shuffle=True, seed=42)
    sampler_b = PerDatasetBatchSampler(dataset_ids, batch_size=4, shuffle=True, seed=42)
    assert list(sampler_a) == list(sampler_b)
