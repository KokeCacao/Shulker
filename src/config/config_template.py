from dataclasses import dataclass
from typing_extensions import Annotated
from typing import Literal, Tuple
import tyro


@dataclass(frozen=True)
class AdamOptimizer:
    learning_rate: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True)
class Dataset:
    root: str
    folder: Literal["donut-c4-f3"]


@dataclass(frozen=True)
class BaseConfigTemplate:
    # whether to print debug information
    debug: bool

    # mode
    mode: Literal["train", "test"]

    # dataset to run experiment on
    dataset: Dataset

    # WARNING: not used
    optimizer: AdamOptimizer

    # random seed (don't touch, see: https://arxiv.org/abs/2109.08203)
    seed: int = 3407


DefaultConfig = Annotated[BaseConfigTemplate,
                          tyro.conf.subcommand(
                              name="default",
                              default=BaseConfigTemplate(
                                  debug=True,
                                    mode="train",
                                  dataset=Dataset(
                                      root="data",
                                      folder="donut-c4-f3",
                                  ),
                                  optimizer=AdamOptimizer(),
                              ),
                              description="Train a smaller model.",
                          ),]

# This config is not used
TestConfig = Annotated[BaseConfigTemplate,
                        tyro.conf.subcommand(
                            name="test",
                            default=BaseConfigTemplate(
                                mode="test",
                                debug=True,
                                dataset=Dataset(
                                    root="data",
                                    folder="donut-c4-f3",
                                ),
                                optimizer=AdamOptimizer(),
                                seed=0,
                            ),
                            description="Train a smaller model.",
                        ),]

