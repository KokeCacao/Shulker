from typing import Union

from rich import print

from config.config_template import BaseConfigTemplate, DefaultConfig, TestConfig
from dataset import Dataset

import tyro

def train(config : BaseConfigTemplate):
    print("Training...")
    dataset = Dataset(config.dataset.root, config.dataset.folder)
    dataset.load()

def test(config : BaseConfigTemplate):
    print("Testing...")

def main(
    config: Union[DefaultConfig, TestConfig],
) -> None:
    if config.debug:
        print("[bold red]Debug mode is on[/bold red]")
        print(config)
    
    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)


if __name__ == "__main__":
    tyro.cli(main)
