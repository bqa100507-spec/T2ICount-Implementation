"""Shared deterministic selection for real limited-compute training subsets."""


def select_train_subset_indices(dataset_size, sample_limit, subset_seed):
    """Return the exact ordered indices used by ``--train-samples``.

    Torch is imported lazily so preprocessing paths that do not select an
    FSC147 training subset do not need to import the legacy training stack.
    """
    if dataset_size < 0:
        raise ValueError("dataset_size must be zero or greater")
    if sample_limit < 0:
        raise ValueError("sample_limit must be zero or greater")
    if sample_limit == 0:
        return list(range(dataset_size))

    import torch

    effective_samples = min(sample_limit, dataset_size)
    generator = torch.Generator()
    generator.manual_seed(subset_seed)
    return torch.randperm(
        dataset_size,
        generator=generator,
    )[:effective_samples].tolist()
