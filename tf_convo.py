#!/usr/bin/env python3
# coding: utf-8

'''
First attempt at implementing a convolutional NN from scratch with tensor flow.
TODO: THIS IS UNFINISHED. FINISH OR DELETE
'''

# ==================================================================================================
#                                             IMPORTS
# ==================================================================================================
# Future imports
from __future__ import annotations

# Generic python imports
import argparse
import pathlib
from typing import Tuple, Optional, Sequence
from dataclasses import dataclass

# Library specific imports
import numpy as np
import tensorflow as tf

# ==================================================================================================
#                             CONFIGURATION AND ARGUMENT PARSING
# ==================================================================================================


def arg_learning_rate(txt: str)-> float:
    try:
        val = float(txt)
        if not 0 < val < 1:
            raise argparse.ArgumentTypeError(
                'Sane learning rate value should be a positive float less than 1.0')
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Unable to convert "{txt}" to valid float')


def arg_positive_int(txt: str) -> int:
    try:
        val = int(txt)
        if not val > 0:
            raise argparse.ArgumentTypeError(
                'Argument value must be greater than 0')
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Unable to convert "{txt}" to valid integer')


def arg_int(txt: str) -> int:
    try:
        val = int(txt)
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Unable to convert "{txt}" to valid integer')

def arg_file_path(txt: str) -> pathlib.Path:
    path = pathlib.Path(txt)

    # If path do not exist throw an exception with absolute path so that we know
    # what exactly we tried to open.
    if not path.exists():
        raise argparse.ArgumentTypeError(
            f'File {path.resolve()} do not exist.')

    return path

@dataclass(frozen=True)
class Config:
    data_dir: str # Directory in which MNIST data is stored
    batch_size: int # Batch size
    epochs: int # Epochs
    learning_rate: float # Learning rate

    @staticmethod
    def from_args(argv: Optional[Sequence[str]] = None) -> Config:
        prsr = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Declare arguments
        prsr.add_argument('--data_dir', required=True,
                          type=str, help='Directory in which MNIST data is stored')
        prsr.add_argument('--batch_size', required=True,
                          type=arg_positive_int, help='Batch size')
        prsr.add_argument('--epochs', required=True,
                          type=arg_positive_int, help='Epochs')
        prsr.add_argument('--learning_rate', required=True,
                          type=arg_learning_rate, help='Learning rate')
        
        # Parsing arguments
        args: argparse.Namespace = prsr.parse_args(argv)

        # Returning config
        return Config(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
# ==================================================================================================
#                                           MNIST LOADER
# ==================================================================================================

class MNISTloader:

    data_folder: pathlib.Path

    def __init__(self, data_folder: str)-> None:
        self.data_folder = pathlib.Path(data_folder)