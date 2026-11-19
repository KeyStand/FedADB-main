import os
import sys
import torch
import torch.nn as nn
import copy
import itertools
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer
from algorithms.measures import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

__all__ = ["ClientTrainer"]


# Custom dataset for class anchors
class AnchorDataset(Dataset):
    def __init__(self, data):
        self.data = data  # List of (image, label) tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label


# Custom dataset to combine local data and class anchors with weak_transform
class CombinedDataset(Dataset):
    def __init__(self, local_dataset, anchor_dataset, class_counts, transform, weak_transform, num_classes):
        self.local_dataset = local_dataset
        self.anchor_dataset = anchor_dataset
        self.class_counts = class_counts
        self.transform = transform
        self.weak_transform = weak_transform
        self.n_classes = num_classes
        self.min_count = min([count for count in class_counts if count > 0])
        self.data = []
        self.labels = []
        # Local data uses hard labels
        for idx in range(len(self.local_dataset)):
            img, label = self.local_dataset[idx]
            self.data.append((img, label, False))
            self.labels.append(label)

        # Class anchors use soft labels
        for idx in range(len(self.anchor_dataset)):
            img, soft_label = self.anchor_dataset[idx]
            self.data.append((img, soft_label, True))
            self.labels.append(soft_label.argmax().item())  # For BalancedSampler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label, is_anchor = self.data[idx]
        if is_anchor:
            img = self.weak_transform(img)
        else:
            img = self.transform(img)
            label = torch.tensor(label)
            label = F.one_hot(label, num_classes=self.n_classes).float()
        return img, label, is_anchor


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, n_classes=10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.indices_per_class = [[] for _ in range(n_classes)]
        for idx, label in enumerate(dataset.labels):
            self.indices_per_class[label].append(idx)
        self.length = max(len(indices) for indices in self.indices_per_class) * n_classes

    def __iter__(self):
        samples_per_class = self.batch_size // self.n_classes if self.batch_size // self.n_classes != 0 else 1
        for _ in range(self.length // self.batch_size):
            batch = []
            for c in range(self.n_classes):
                class_indices = self.indices_per_class[c]
                if len(class_indices) > 0:
                    selected = np.random.choice(class_indices,
                                                samples_per_class,
                                                replace=len(class_indices) < samples_per_class)
                    batch.extend(selected)
            np.random.shuffle(batch)
            yield from batch[:self.batch_size]

    def __len__(self):
        return self.length


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        FedADB client trainer.
        Uses class anchors for multi-balanced training.
        """
        # Use dictionary to store second classifier for all clients
        self.all_local_classifiers = {}
        self.client_id = None

    def calculate_class_anchor_ratio(self, class_counts, anchor_counts):
        """
        Calculate the ratio of class anchors for each class and return the average ratio
        """
        anchor_ratios = []
        valid_classes = 0
        
        for i in range(self.num_classes):
            total_count = class_counts[i] + anchor_counts[i]
            if total_count > 0:
                ratio = anchor_counts[i] / total_count
                anchor_ratios.append(ratio)
                valid_classes += 1
            else:
                anchor_ratios.append(0.0)
                
        avg_anchor_ratio = sum(anchor_ratios) / valid_classes if valid_classes > 0 else 0.0

        owned_classes = sum(1 for count in class_counts if count > 0)
        total_classes = self.num_classes if hasattr(self, 'num_classes') else len(class_counts)
        owned_class_ratio = owned_classes / total_classes if total_classes > 0 else 0.0

        return anchor_ratios, avg_anchor_ratio, owned_class_ratio

    def criterion_bal(self, outputs, labels, is_anchor):
        device = outputs.device
        labels = labels.to(device)
        loss = torch.tensor(0.0, device=device)
        batch_size = outputs.size(0)

        for i in range(batch_size):
            output = outputs[i].unsqueeze(0)
            label = labels[i]
            if is_anchor[i]:  # Use ross-entropy loss for class anchors
                log_probs = torch.log_softmax(output, dim=1)
                target_dist = label.to(device)  # Soft label
                loss -= (target_dist * log_probs).sum()
            else:  # Use cross-entropy loss for local data
                sum_criterion = nn.CrossEntropyLoss(reduction='sum')
                loss += sum_criterion(output, label.argmax().unsqueeze(0))
        return loss / batch_size

    def train(self, round_idx, class_anchors, anchor_counts, avg_confidence=None):
        """
        FedADB Local train
        """
        all_batch_size = self.task_args.get('data_setups')['batch_size']
        all_FedEpoch = self.task_args.get('train_setups').get('scenario')['n_rounds']

        batch_size = int(all_batch_size / 2)
        num_workers = 2

        """Local training"""
        if isinstance(self.trainloader.dataset.targets, list):
            targets = torch.tensor(self.trainloader.dataset.targets)
        elif isinstance(self.trainloader.dataset.targets, torch.Tensor):
            targets = self.trainloader.dataset.targets
        else:
            targets = torch.from_numpy(self.trainloader.dataset.targets)
        class_counts = torch.bincount(targets, minlength=self.num_classes).tolist()

        _, avg_anchor_ratio, owned_class_ratio = self.calculate_class_anchor_ratio(class_counts, anchor_counts)

        # Data preprocessing
        transform = transforms.Compose([
        ])

        # for cifar10
        weak_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])

        anchor_dataset = AnchorDataset(class_anchors)
        combined_dataset = CombinedDataset(
            self.trainloader.dataset, anchor_dataset,
            class_counts, transform, weak_transform, self.num_classes
        )

        # Create local dataset and loader for Classifier 2
        local_only_dataset = self.trainloader.dataset

        self.model.train()
        self.model.to(self.device)
        self.model.Both = True


        local_size = self.datasize

        # Calculate total training steps in normal training mode
        total_steps = self.local_epochs * len(self.trainloader)
        print(
            f"Local Train exec total_steps: {total_steps}=epochs={self.local_epochs}× batches_per_epoch={len(self.trainloader)}")

        # Use epoch-based iteration, recreate DataLoader for each epoch
        steps_per_epoch = len(self.trainloader)
        for epoch in range(self.local_epochs):
            # Recreate DataLoader for each epoch to sample data
            sampler = BalancedSampler(combined_dataset, batch_size, self.num_classes)
            combined_loader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )

            local_loader = DataLoader(
                local_only_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )

            # Use itertools.cycle to create infinite iterator
            combined_iter = itertools.cycle(combined_loader)
            local_iter = itertools.cycle(local_loader)

            for _ in range(steps_per_epoch):
                try:
                    local_inputs, local_labels = next(local_iter)
                    combined_inputs, combined_labels, is_anchor = next(combined_iter)

                    local_inputs, local_labels = local_inputs.to(self.device), local_labels.to(self.device)
                    combined_inputs, combined_labels, is_anchor = combined_inputs.to(
                        self.device), combined_labels, is_anchor.to(self.device)

                    self.optimizer.zero_grad()
                    output1, _ = self.model(combined_inputs)  # Classifier 1
                    loss1 = self.criterion_bal(output1, combined_labels, is_anchor)
                    _, output2 = self.model(local_inputs)  # Classifier 2

                    beta = (((round_idx+1) / (all_FedEpoch)) ** ((1-owned_class_ratio)*avg_confidence))

                    loss2 = self.criterion(output2, local_labels)

                    total_loss = loss1 + beta * loss2
                    total_loss.backward()
                    self.optimizer.step()
                except StopIteration:
                    print("Warning: Dataset is empty or insufficient samples, skipping current training step")
                    continue


        self.model.Both = False
        local_results = self._get_local_stats()

        return local_results, local_size

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.model.to(self.device)  # Ensure model is on the correct device
        self.optimizer.load_state_dict(server_optimizer)


    def upload_local(self):
        """Uploads local model's parameters"""
        local_weights = copy.deepcopy(self.model.state_dict())

        return local_weights

