import os
import json
import argparse
from shutil import copyfile
import torch
import torch.multiprocessing as mp

from src.tools.train.trainer_sttn import Trainer
from src.tools.train.utils_sttn import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-c', '--config', default='configs_sttn/youtube-vos.json', type=str)
parser.add_argument('-m', '--model', default='sttn', type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('-e', '--exam', action='store_true')
args = parser.parse_args()


def main_worker(rank, config):
    # If local_rank is not in config, assign rank to local_rank and global_rank
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank

    # If config specifies distributed training
    if config['distributed']:
        # Set CUDA device to GPU corresponding to current local_rank
        torch.cuda.set_device(int(config['local_rank']))
        # Initialize distributed process group via nccl backend
        torch.distributed.init_process_group(
            src='nccl',
            init_method=config['init_method'],
            world_size=config['world_size'],
            rank=config['global_rank'],
            group_name='mtorch'
        )
        # Print current GPU usage, outputting global and local ranks
        print('using GPU {}-{} for training'.format(
            int(config['global_rank']), int(config['local_rank']))
        )

    # Create model save directory path including model name and config name
    config['save_dir'] = os.path.join(
        config['save_dir'], '{}_{}'.format(config['model'], os.path.basename(args.config).split('.')[0])
    )

    # Set device to CUDA if available, else CPU
    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else:
        config['device'] = 'cpu'

    # If not distributed or if master node (rank 0)
    if (not config['distributed']) or config['global_rank'] == 0:
        # Create model save directory, ignore if exists
        os.makedirs(config['save_dir'], exist_ok=True)
        # Set config file save path
        config_path = os.path.join(
            config['save_dir'], config['config'].split('/')[-1]
        )
        # If config file doesn't exist, copy from original path
        if not os.path.isfile(config_path):
            copyfile(config['config'], config_path)
        # Print directory creation info
        print('[**] create folder {}'.format(config['save_dir']))

    # Initialize trainer with config and debug flag
    trainer = Trainer(config, debug=args.exam)
    # Start training
    trainer.train()


if __name__ == "__main__":
    # Load config file
    config = json.load(open(args.config))
    config['model'] = args.model  # Set model name
    config['config'] = args.config  # Set config file path

    # Set distributed training config
    config['world_size'] = get_world_size()  # Get world size, total GPUs participating in training
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"  # Set initialization method including master IP and port
    config['distributed'] = True if config['world_size'] > 1 else False  # Enable distributed training if world size > 1

    # Set distributed parallel training environment
    if get_master_ip() == "127.0.0.1":
        # If master IP is localhost, manually spawn multiple distributed processes
        mp.spawn(main_worker, nprocs=config['world_size'], args=(config,))
    else:
        # If processes are started by other tools like OpenMPI, no need to manually create them.
        config['local_rank'] = get_local_rank()  # Get local rank
        config['global_rank'] = get_global_rank()  # Get global rank
        main_worker(-1, config)  # Start main worker function
