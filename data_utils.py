# encoding=utf-8
import logging
import os
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms, datasets

logger = logging.getLogger(__name__)

# ✅ Stronger augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

# ✅ Validation transform stays deterministic
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


def compute_class_weights(train_dataset):
    counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / c for c in counts]
    weights = torch.tensor(weights, dtype=torch.float)
    return weights


def get_loader(args):
    if not hasattr(args, 'data_dir') or args.data_dir is None:
        raise ValueError("You must set --data_dir to the path containing 'train' and 'test' folders.")

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    if not os.path.isdir(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise ValueError(f"Test directory not found: {test_dir}")

    logger.info(f"✅ Preparing datasets from: {args.data_dir}")
    logger.info(f"  ➜ Train dir: {train_dir}")
    logger.info(f"  ➜ Test dir: {test_dir}")

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Datasets
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform
    )

    # Samplers
    train_sampler = RandomSampler(train_dataset)
    test_sampler = RandomSampler(test_dataset)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    if args.local_rank in [-1, 0]:
        logger.info(f"✅ Number of training samples: {len(train_dataset)}")
        logger.info(f"✅ Number of test samples: {len(test_dataset)}")
        logger.info(f"✅ Classes: {train_dataset.classes}")
        logger.info(f"✅ Class-to-Index mapping: {train_dataset.class_to_idx}")

    # ✅ Compute class weights for balancing
    class_weights = compute_class_weights(train_dataset)

    return train_loader, test_loader, class_weights
