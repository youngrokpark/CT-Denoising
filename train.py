import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from config import TrainerConfig, ModelConfig
from modules.trainer import Trainer

# Setup command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--device', type=int, default=None)
parser.add_argument('--supervised', '-s', action='store_true', default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Determine the computing device
    device = torch.device(f'cuda:{args.device}') if args.device is not None else torch.device('cpu')
    
    # Configuration and logging setup
    train_config = TrainerConfig()
    model_config = ModelConfig()
    logger = SummaryWriter(train_config.path_checkpoint)
    
    # Optional override for supervised training
    if args.supervised is not None:
        train_config.supervised = args.supervised

    # Initialize and run the training
    trainer = Trainer(train_config, model_config, device)
    trainer.run(logger)
