import torch
import torch.nn as nn
import numpy as np
import copy
import time
import wandb
import os
import math
import json

from .measures import *

__all__ = ["BaseServer"]

from sklearn.cluster import KMeans


class BaseServer:
    def __init__(
            self,
            algo_params,
            model,
            data_distributed,
            optimizer,
            scheduler,
            task_args,
            n_rounds=200,
            sample_ratio=0.1,
            local_epochs=5,
            device="cuda:0",
            log_to_file=False,
            **kwargs,
    ):
        """
        Server class controls the overall experiment.
        """
        self.algo_params = algo_params
        self.num_classes = data_distributed["num_classes"]
        self.model = model
        self.data_distributed = data_distributed
        self.task_args = task_args
        dataset_name = self.task_args['data_setups']['dataset_name']
        print("dataset name :", dataset_name)

        self.testloader = self.data_distributed["global"]["test"]

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_ratio = sample_ratio
        self.n_rounds = n_rounds
        self.device = device
        self.n_clients = len(data_distributed["local"].keys())
        self.local_epochs = local_epochs
        self.server_results = {
            "client_history": [],
            "test_accuracy": [],
        }
        # Get model name
        self.model_name = self.model.__class__.__name__
        
        # Track best model
        self.best_accuracy = 0.0
        self.best_model_state = None

        try:
            cfg_log = bool(self.task_args["train_setups"]["scenario"].get("log_to_file", False))
        except Exception:
            cfg_log = False
        self.log_to_file = bool(log_to_file or cfg_log)

        self.log_file_path = None
        if self.log_to_file:
            ws = self.task_args.get("wandb_setups", {})
            project = str(ws.get("project", "proj"))
            group = str(ws.get("group", "group"))
            name = str(ws.get("name", "exp"))
            def _safe(s):
                return s.replace("/", "_").replace("\\", "_").replace(" ", "_")
            filename = f"{_safe(project)}__{_safe(group)}__{_safe(name)}.jsonl"
            log_dir = os.path.join("./logs")
            os.makedirs(log_dir, exist_ok=True)
            self.log_file_path = os.path.join(log_dir, filename)

        # Configuration for saving global model every 5 rounds (read from scenario, injected by main.py)
        try:
            scen = dict(self.task_args.get("train_setups", {}).get("scenario", {}))
        except Exception:
            scen = {}
        self.save_every5_enable = bool(scen.get("save_every5_enable", False))
        self.save_every5_dir = str(scen.get("save_every5_dir", "./saved_model_for_visualPseTsne"))
        if self.save_every5_enable:
            os.makedirs(self.save_every5_dir, exist_ok=True)

    def run(self):
        """Run the FL experiment"""
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

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients
            )
            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)

            # Update global weights and evaluate statistics
            self._update_and_evaluate(ag_weights, round_results, round_idx, start_time)


    def _clients_training(self, sampled_clients):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_results = {}



        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:
            print("client ", client_idx, " start Train")
            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Local training
            local_results, local_size = self.client.train()

            # Upload locals
            updated_local_weights.append(self.client.upload_local())



            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()



        return updated_local_weights, client_sizes, round_results

    def _client_sampling(self, round_idx):
        """Sample clients by given sampling ratio"""

        # make sure for same client sampling for fair comparison
        np.random.seed(round_idx)
        clients_per_round = max(int(self.n_clients * self.sample_ratio), 1)
        sampled_clients = np.random.choice(
            self.n_clients, clients_per_round, replace=False
        )

        return sampled_clients



    def _set_client_data(self, client_idx):
        """Assign local client datasets."""
        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"]
        self.client.testloader = self.data_distributed["global"]["test"]

    def _aggregation(self, w, ns):
        """Average locally trained model parameters"""
        prop = torch.tensor(ns, dtype=torch.float)
        prop /= torch.sum(prop)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * prop[0]

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * prop[i]

        return copy.deepcopy(w_avg)

    def _results_updater(self, round_results, local_results):
        """Combine local results as clean format"""

        for key, item in local_results.items():
            if key not in round_results.keys():
                round_results[key] = [item]
            else:
                round_results[key].append(item)

        return round_results

    def _print_start(self):
        """Print initial log for experiment"""

        if self.device == "cpu":
            return "cpu"

        if isinstance(self.device, str):
            device_idx = int(self.device[-1])
        elif isinstance(self.device, torch._device):
            device_idx = self.device.index

        device_name = torch.cuda.get_device_name(device_idx)
        print("")
        print("=" * 50)
        print("Train start on device: {}".format(device_name))
        print("=" * 50)

    def _print_stats(self, round_results, test_accs, round_idx, round_elapse):
        print(
            "[Round {}/{}] Elapsed {}s (Current Time: {})".format(
                round_idx + 1,
                self.n_rounds,
                round(round_elapse, 1),
                time.strftime("%H:%M:%S"),
            )
        )
        print(
            "[Local Stat (Train Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
                round_results["train_acc"],
                np.mean(round_results["train_acc"]),
                np.std(round_results["train_acc"]),
            )
        )

        print(
            "[Local Stat (Test Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
                round_results["test_acc"],
                np.mean(round_results["test_acc"]),
                np.std(round_results["test_acc"]),
            )
        )

        print("[Server Stat] Acc - {:2.2f}".format(test_accs))

    def _wandb_logging(self, round_results, round_idx):
        """Log on the W&B server"""

        # Local round results
        local_results = {
            "local_train_acc": np.mean(round_results["train_acc"]),
            "local_test_acc": np.mean(round_results["test_acc"]),
        }
        wandb.log(local_results, step=round_idx)

        # Server round results
        server_results = {"server_test_acc": self.server_results["test_accuracy"][-1]}
        wandb.log(server_results, step=round_idx)

    def _update_and_evaluate(self, ag_weights, round_results, round_idx, start_time):
        """Evaluate experiment statistics."""

        # Update Global Server Model
        self.model.load_state_dict(ag_weights)

        # Measure Accuracy Statistics
        print('\nEval Agg Model:')
        test_acc = evaluate_model_classwise(self.model, self.testloader, device=self.device,
                                            num_classes=self.num_classes)
        self.server_results["test_accuracy"].append(test_acc[1])
        
        # Update best model
        if test_acc[1] > self.best_accuracy:
            self.best_accuracy = test_acc[1]
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            print(f"Best MODEL ! Acc: {self.best_accuracy:.4f}")

        # Evaluate Personalized FL performance
        eval_results = get_round_personalized_acc(
            round_results, self.server_results, self.data_distributed
        )
        wandb.log(eval_results, step=round_idx)

        # Change learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        round_elapse = time.time() - start_time

        # Log and Print
        self._wandb_logging(round_results, round_idx)
        self._print_stats(round_results, test_acc[1], round_idx, round_elapse)
        print("-" * 50)

        # Append to log file
        if self.log_to_file and self.log_file_path is not None:
            try:
                raw_list = test_acc[0].view(-1).tolist()
                classwise_list = []
                for x in raw_list:
                    try:
                        fx = float(x)
                        if math.isnan(fx) or math.isinf(fx):
                            classwise_list.append(None)
                        else:
                            classwise_list.append(fx)
                    except Exception:
                        classwise_list.append(None)
                local_avg_test = float(np.mean(round_results.get("test_acc", []))) if len(round_results.get("test_acc", [])) > 0 else 0.0
                payload = {
                    "round": int(round_idx),
                    "global_classwise_acc": classwise_list,
                    "global_avg_acc": float(test_acc[1]),
                    "local_avg_test_acc": local_avg_test,
                }
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed to write log file: {e}")


