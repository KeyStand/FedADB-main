import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR10_truncated



def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=2),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),]
    )

    return train_transform, valid_transform


def get_all_targets_cifar10(root, train=True):
    dataset = CIFAR10_truncated(root=root, train=train)
    all_targets = dataset.targets
    return all_targets


def get_dataloader_cifar10(root, train=True, batch_size=50, dataidxs=None):
    train_transform, valid_transform = _data_transforms_cifar10()

    if train:
        dataset = CIFAR10_truncated(
            root, dataidxs, train=True, transform=train_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=3
        )
        dataset.targets

    else:
        dataset = CIFAR10_truncated(
            root, dataidxs, train=False, transform=valid_transform, download=False
        )
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=3
        )

    return dataloader
