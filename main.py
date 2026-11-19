import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import algorithms
from train_tools import *
from utils import *

import numpy as np
import argparse
import warnings
import wandb
import random
import pprint
import json
import shutil

warnings.filterwarnings("ignore")
torch.set_printoptions(10)

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "fedadb": algorithms.fedadb.Server,

}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}


def _get_setups(args):
    """Get train configuration"""

    # Fix randomness for data distribution
    np.random.seed(1)
    random.seed(1)

    # Distribute the data to clients
    data_distributed = data_distributer(**args.data_setups)

    # Fix randomness for experiment
    _random_seeder(args.train_setups.seed)
    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name,
        **args.train_setups.model.params,
    )

    # Optimization setups
    optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)
    scheduler = None

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    # Algorith-specific global server container
    algo_params = args.train_setups.algo.params
    server = ALGO[args.train_setups.algo.name](
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        task_args=args,
        **args.train_setups.scenario,
    )

    return server


def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config_to_wandb(config, config_path):
    """Save configuration to wandb and locally"""
    # Save configuration as Python file format
    config_content = "# Auto-generated config file\n\n"
    config_content += "CONFIG = " + json.dumps(config, indent=4)
    
    # Create config.py file and save to wandb run directory
    config_py_path = os.path.join(wandb.run.dir, "config.py")
    with open(config_py_path, "w") as f:
        f.write(config_content)
    
    # Upload config file to wandb (but not model files)
    wandb.save(config_py_path, policy="now")
    
    # Also save original config file to wandb directory
    if os.path.exists(config_path):
        dest_path = os.path.join(wandb.run.dir, os.path.basename(config_path))
        shutil.copy(config_path, dest_path)
        wandb.save(dest_path, policy="now")
    
    print(f"Save config to: {config_py_path}")
    return config_py_path


def main(args):
    """Execute experiment"""

    # Load the configuration
    server = _get_setups(args)

    # Conduct FL
    server.run()




# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Configs")
parser.add_argument("--config_path", default="./config/fedavg.json", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--n_clients", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--partition_method", type=str)
parser.add_argument("--partition_s", type=int)
parser.add_argument("--partition_alpha", type=float)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_rounds", type=int)
parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--local_epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--algo_name", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--group", type=str)
parser.add_argument("--exp_name", type=str)
parser.add_argument("--log_to_file", action="store_true", default=True)
parser.add_argument("--save_every5_enable",  default=False,)
parser.add_argument("--save_every5_dir", type=str, default="./saved_model_for_visualPseTsne")
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    opt = ConfLoader(args.config_path).opt

    opt = config_overwriter(opt, args)

    # Print configuration dictionary pretty
    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    os.environ["WANDB_IGNORE_GLOBS"] = "*.pth,*.pt,*.bin,*.ckpt"

    # Initialize W&B
    wandb.init(config=opt, **opt.wandb_setups)

    # Save configuration to wandb and local
    config_py_path = save_config_to_wandb(opt, args.config_path)

    # How many batches to wait before logging training status
    wandb.config.log_interval = 10

    # Execute expreiment
    main(opt)
