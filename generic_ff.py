#!/usr/bin/env python3
# coding: utf-8
'''
This scripts trains a simple feed forward neural network based on given train data CSV and
then tests it with data from test CSV. You have to give a comma separated list of input
columns as well as outut columns. Type of task (regression or classification) as well as 
the architecture of network are inferred from data.
'''
# ==================================================================================================
#                                             IMPORTS
# ==================================================================================================
# Future imports
from __future__ import annotations

# Basic python imports
import argparse
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Sequence, List
from datetime import datetime

# External libraries
import numpy as np
import tensorflow as tf
import pandas as pd

# ==================================================================================================
#                                      LITERALS AND CONSTS
# ==================================================================================================
REGRESSION_ID: str = 'regression'
BINARY_CLASSIFIER_ID: str = 'binary_classification'
MULTI_CLASSIFIER_ID: str = 'multiclass_classification'
TaskType = Literal[REGRESSION_ID, BINARY_CLASSIFIER_ID, MULTI_CLASSIFIER_ID]

# ==================================================================================================
#                                          LOGGING
# ==================================================================================================


class VerbosityLevel(Enum):
    ''' Verbosity levels compatiblile with Keras'''
    SILENT = 0,
    INFO = 1,
    DEBUG = 2

    def to_python_logging_verbosity(self) -> int:
        return {
            VerbosityLevel.SILENT: 60,  # Value higher than logging.CRITICAL
            VerbosityLevel.INFO: logging.INFO,
            VerbosityLevel.DEBUG: logging.DEBUG
        }[self]

# ==================================================================================================
#                             CONFIGURATION AND ARGUMENT PARSING
# ==================================================================================================


def arg_column_list(txt: str) -> Tuple[str, ...]:
    '''
    Function that converts comma separated list of strings into a
    tuple of strings. This is intended to be used by argparse 
    to deal with list of columns in dataset.
    '''
    column_names = [name.strip() for name in txt.split(',') if name.strip()]
    if not column_names:
        raise argparse.ArgumentTypeError(
            'Expected a nonepty, comma separated, list.')
    return tuple(column_names)


def arg_validation_split(txt: str) -> float:
    try:
        val = float(txt)
        if not (0.0 < val < 0.5):
            raise argparse.ArgumentTypeError('Validation split must be floating point value in'
                                             ' range <0.0;0.5>')
        return val
    except ValueError:
        raise argparse.ArgumentTypeError('Validation split must be a float, value '
                                         f'{txt} given instead.')


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


def arg_verbosity_level(txt: str) -> VerbosityLevel:
    try:
        return VerbosityLevel[txt.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f'Verbosity level {txt} not recognized available '
                                         f'values are {[e._name_ for e in VerbosityLevel]}')


@dataclass(frozen=True)
class Config:
    train_csv: str  # Path to CSV with training data (inputs and outputs)
    test_csv: str  # Path to CSV with test data (inputs and outputs)
    model_out: str  # Path to file in which model will be saved
    history_out: str | None  # Path to CSV where learning history will be stored,
    # if set to None history will not be saved.
    input_cols: Tuple[str, ...]  # Columns that will be used as input,
    # must exist in train and test CSV
    output_cols: Tuple[str, ...]  # Columns that will be used as output,
    # must exist in train and test CSV
    validation_split: float  # Part of data that will be used for validation,
    # must be between 0.0 and 0.5
    random_seed: int  # A seed used for all random generators, if set to 0 time will be used
    epochs: int  # Maximum number of epochs for which network will be trained.
    verbosity: VerbosityLevel  # Program verbosity

    @staticmethod
    def from_args(argv: Optional[Sequence[str]] = None) -> Config:
        prsr = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Declare arguments
        prsr.add_argument('--train_csv', required=True,
                          type=str, help='Path to training CSV')
        prsr.add_argument('--test_csv', required=True,
                          type=str, help='Path to test CSV')
        prsr.add_argument('--input_cols', required=True, type=arg_column_list,
                          help='Comma separated names of input feature columns.',
                          )
        prsr.add_argument('--output_cols', required=True, type=arg_column_list,
                          help='Comma separated names of output/target columns.',
                          )
        prsr.add_argument('--model_out', type=str, required=True,
                          help='Path to file where model will be saved.')
        prsr.add_argument('--validation_split', type=arg_validation_split, default=0.15,
                          help='Fraction of training data used for validation (defult: 0.15)')
        prsr.add_argument('--random_seed', type=arg_int, default=0,
                          help='Random seed for reproducibility, if not give timer is used as seed')
        prsr.add_argument('--epochs', type=arg_positive_int, default=100,
                          help='Maximum number of epochs with early stopping (default: 100)')
        prsr.add_argument('--history_out', type=str, default=None,
                          help='File to which learning history will be saved'
                          ', if this argument is not set history will not be saved at all.')
        prsr.add_argument('--verbosity', type=arg_verbosity_level,
                          default=VerbosityLevel.INFO,
                          help='Verbosity level, available value are '
                          f'{[e._name_ for e in VerbosityLevel]}'
                          ' (default: INFO)')

        # Parsing arguments
        args: argparse.Namespace = prsr.parse_args(argv)

        # Returning config
        return Config(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            input_cols=args.input_cols,
            output_cols=args.output_cols,
            validation_split=args.validation_split,
            random_seed=args.random_seed,
            epochs=args.epochs,
            verbosity=args.verbosity,
            model_out=args.model_out,
            history_out=args.history_out
        )

# ==================================================================================================
#                                      HELPER FUNCTIONS
# ==================================================================================================


def set_global_random_seed(seed: int) -> None:
    if seed == 0:
        seed = int(datetime.now().timestamp())
    np.random.seed(seed)
    tf.random.set_seed(seed)


def check_for_missing_columns(data: pd.DataFrame, cols: Sequence[str]) -> Sequence[str]:
    ''' Takes a list of columns and a dataframe and then returns list of those columns that 
     while given as argument while not present in dataframe '''
    return [col for col in cols if col not in data.columns]


def is_numeric(s: pd.Series) -> bool:
    ''' For better redability of final code (why there is no s.is_numeric() :( )'''
    return pd.api.types.is_numeric_dtype(s)


def attempt_one_hot_encoding(data: pd.DataFrame,
                             categorical_cols: List[str],
                             categories: Optional[List[str]] = None
                             ) -> pd.DataFrame:
    if not categorical_cols:
        return data.copy() # Using copy to be consistent with other cases in this function
    encoded =  pd.get_dummies(data, columns=categorical_cols, dummy_na=False)
    if not categories:
        return encoded
    # Add missing categories
    for category in  categories:
        if category not in encoded.columns:
            encoded[category] = 0
    # Remove extra categories
    valid_columns: List[str] = list(data.columns) + categories 
    to_drop = [col for col in encoded.columns if col not in valid_columns]
    if to_drop:
        encoded.drop(columns=to_drop, inplace=True)
    return encoded
    

# ==================================================================================================
#                                           MAIN
# ==================================================================================================


def main():
    cfg: Config = Config.from_args()
    print(cfg)
    print(cfg.verbosity.to_python_logging_verbosity())


if __name__ == '__main__':
    main()
