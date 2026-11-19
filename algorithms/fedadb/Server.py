import copy
import os
import sys
import time
import math
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.fedadb.ClientTrainer import ClientTrainer
from algorithms.measures import evaluate_model

__all__ = ["Server"]


def generate_class_anchors(
    model, num_classes=100, anchor_per_class=1, image_size=(3, 32, 32),
    lr=0.01, steps=500, temperature=1.0, noise_scale=0.1, batch_split=1,batch_size=128
):

    model.eval()
    device = next(model.parameters()).device
    anchor_images = []
    anchor_targets = []

    print(f"Starting to generate class anchors, {anchor_per_class} anchors per class")

    for split in range(batch_split):
        start_class = split * (num_classes // batch_split)
        end_class = (split + 1) * (num_classes // batch_split) if split < batch_split - 1 else num_classes
        current_classes = end_class - start_class
        print(f"Optimizing class anchors for classes {start_class} to {end_class - 1}...")

        batch_images = torch.randn(current_classes * anchor_per_class, *image_size).to(device) * noise_scale
        batch_images.requires_grad_(True)
        batch_targets = []
        for c in range(current_classes):
            batch_targets.extend([start_class + c] * anchor_per_class)
        batch_targets = torch.tensor(batch_targets, dtype=torch.long).to(device)

        optimizer = torch.optim.Adam([batch_images], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = -torch.log_softmax(outputs, dim=1)[range(len(batch_targets)), batch_targets].mean()
            l2_reg = 0.01 * (batch_images ** 2).mean()
            total_loss = loss + l2_reg
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                batch_images.clamp_(-1.0, 1.0)
            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{steps}, Loss: {total_loss.item():.4f}")

        anchor_images.append(batch_images.detach().cpu())
        anchor_targets.append(batch_targets.cpu())

    anchor_images = torch.cat(anchor_images, dim=0)
    total_samples = anchor_images.shape[0]

    step = int(batch_size / num_classes) if int(batch_size / num_classes) > 1 else 1
    
    soft_labels = []
    confidences = []
    for i in range(0, total_samples, step):
        step_imgs = anchor_images[i:i+step].to(device)
        with torch.no_grad():
            outputs = model(step_imgs)
            soft_batch = torch.softmax(outputs / temperature, dim=1)
        for j in range(step_imgs.shape[0]):
            soft_label = soft_batch[j].cpu()
            confidence = soft_label.max().item()
            soft_labels.append(soft_label)
            confidences.append(confidence)
    class_anchors = []
    for i in range(total_samples):
        class_anchors.append((anchor_images[i], soft_labels[i]))

    avg_confidence = sum(confidences) / len(confidences)
    print(f"Class anchors generation completed! Total samples: {len(class_anchors)}\n")
    return class_anchors, avg_confidence


class Server(BaseServer):
    def __init__(
        self,
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        task_args,
        **kwargs,
    ):
        server_kwargs = dict(kwargs)

        super(Server, self).__init__(
            algo_params,
            model,
            data_distributed,
            optimizer,
            scheduler,
            task_args,
            **server_kwargs,
        )

        self.client = ClientTrainer(
            algo_params=algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
            task_args=task_args,
        )

        print("\n>>> FedADB Server initialized...\n")

    def run(self):
        """Run the FL experiment with FedADB logic"""
        self._print_start()
        for round_idx in range(self.n_rounds):
            print('----------Start Train Round :', round_idx, '------------------')
            # Initial Model Statistics
            if round_idx == 0:
                test_acc = evaluate_model(
                    self.model, self.testloader, device=self.device, num_classes=self.num_classes
                )
                self.server_results["test_accuracy"].append(test_acc)

            start_time = time.time()

            # Generate class anchors
            dataset_name = self.task_args['data_setups']['dataset_name']
            image_size = (3, 32, 32)
            # Get anchor_per_class from algorithm parameters
            anchor_per_class = self.algo_params.get('anchor_per_class', 1)

            class_anchors, avg_confidence = generate_class_anchors(
                self.model, num_classes=self.num_classes, anchor_per_class=anchor_per_class,
                image_size=image_size)
            
            # Calculate the number of class anchors for each class
            anchor_counts = [anchor_per_class] * self.num_classes

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            # Client training stage with FedADB (using class anchors for training)
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients, round_idx, class_anchors, avg_confidence, anchor_counts
            )

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)

            # Update global weights and evaluate statistics
            self._update_and_evaluate(ag_weights, round_results, round_idx, start_time)

        # Save final model after training
        self._save_final_model()

    def _clients_training(self, sampled_clients, round_idx, class_anchors, avg_confidence, anchor_counts):
        """Conduct local training with FedADB logic (using class anchors for training)"""
        updated_local_weights, client_sizes = [], []
        round_results = {}

        # Record L2 distances between each client and global model in this round
        l2_distances = [] if self.log_to_file else None

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:
            print("client ", client_idx, " start Train")
            # Fetch client datasets
            self._set_client_data(client_idx)

            # Set client ID
            self.client.client_id = client_idx

            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Local training with FedADB (using class anchors)
            local_results, local_size = self.client.train(
                round_idx, class_anchors, anchor_counts, avg_confidence
            )

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            # Calculate L2 distance to global model (only when logging is enabled)
            if l2_distances is not None:
                try:
                    local_w = updated_local_weights[-1]
                    total_sq = 0.0
                    for k in server_weights.keys():
                        # Exclude second classifier parameters (fc2)
                        if 'fc2' in k:
                            continue
                        diff = (server_weights[k].detach().cpu().float() - local_w[k].detach().cpu().float())
                        total_sq += float(torch.sum(diff * diff).item())
                    l2_distances.append(float(math.sqrt(total_sq)))
                except Exception:
                    pass

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        # Write distance statistics to round_results for subsequent logging
        if l2_distances is not None:
            if len(l2_distances) > 0:
                round_results["l2_distances"] = l2_distances
                round_results["l2_distance_mean"] = float(np.mean(l2_distances))
            else:
                round_results["l2_distances"] = []
                round_results["l2_distance_mean"] = 0.0

        return updated_local_weights, client_sizes, round_results

    def _save_final_model(self):
        """Save final model after training"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Final model saved with best accuracy: {self.best_accuracy:.4f}")

