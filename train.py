import argparse
import time
import torch
from pytorch_lightning import seed_everything, Trainer
from torch.utils.data import DataLoader
from pathlib import Path
import torch.optim as optimizers
import torch.optim.lr_scheduler as lr_schedulers
import yaml

from mt_grad.data.dataset import MTDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lightning import LightningEngine
from mt_grad.model import MTParamModel


def parse_args():
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-c", "--config", help="path to yaml config", type=str, required=True
    )
    args, unknown = arguments_parser.parse_known_args()
    return args


def prepare_experiment(exp_path: Path, exp_name: str):
    exp_path.mkdir(exist_ok=True)
    experiment_path = exp_path / exp_name / str(time.ctime()).replace(' ', '_').replace(':', '_')
    logs_dir = experiment_path / 'logs'
    checkpoint_dir = experiment_path / '_checkpoints'
    experiment_path.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    return experiment_path, logs_dir, checkpoint_dir


def train(config: dict):

    experiment_path = Path(config['experiments_path'])
    experiment_path, logs_dir, checkpoint_dir = prepare_experiment(experiment_path, config['experiment_name'])

    seed_everything(config.get('seed', 1024))

    # Initialize model
    model = MTParamModel(
        **config['model_params']
    )

    # Initialize criterion
    criterion = torch.nn.MSELoss()

    # Initialize optimizer
    optimizer = getattr(optimizers, config['optimizer_name'])(model.parameters(), **config['optimizer_params'])

    # Initialize scheduler
    scheduler = getattr(lr_schedulers, config['scheduler_name'])(optimizer, **config['scheduler_params'])

    # Initialize train / val / test datasets
    train_dataset = MTDataset(
        **config['train_dataset_params']
    )
    valid_dataset = MTDataset(
        **config['valid_dataset_params']
    )

    # Initialize train & validation dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['dataloader_params']['train_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=config['dataloader_params']['num_workers']
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config['dataloader_params']['val_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config['dataloader_params']['num_workers']
    )

    # Initialize Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        **config['model_checkpoint_params']
    )
    logger = TensorBoardLogger(name=config['experiment_name'], save_dir=str(logs_dir))

    # Initialize lightning module
    module = LightningEngine(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )

    # Initialize trainer
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],  # , visualize],
        **config['trainer_params']
    )

    trainer.fit(module, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as file:
        config = yaml.load(file)
    train(config)
