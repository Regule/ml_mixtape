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
import math
import sys
from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, List, Final, Dict, Any
from datetime import datetime

# External libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import keras as krs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# ==================================================================================================
#                                          CONSTANS
# ==================================================================================================
MIN_LAYER_SIZE: Final[int] = 16
MAX_LAYER_SIZE: Final[int] = 512
FEATURES_TO_SIZE_FACTOR: Final[float] = 2.5
MAX_LAYER_COUNT: Final[int] = 3
# TODO: Name following two better
SAMPLES_TO_LAYER_BASE: Final[int] = 1000
SAMPLES_TO_LAYER_STEP_MULTIPLIER: Final[int] = 10
LEARNING_RATE: float = 1e-3  # Make this argument later
MIN_LEARNING_RATE: float = 1e-6  # Make this argument later

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

    @staticmethod
    def parse_string(txt: str) -> VerbosityLevel:
        try:
            return VerbosityLevel[txt.upper()]
        except KeyError:
            raise KeyError(f'String "{txt}" is not valid for VerbosityLevel '
                           f'enum. Valid values are {[e.name for e in VerbosityLevel]}')

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.name}'

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
        return VerbosityLevel.parse_string(txt)
    except KeyError as e:
        raise argparse.ArgumentTypeError(str(e))


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
#                                           HELPERS
# ==================================================================================================


def set_global_random_seed(seed: int) -> None:
    if seed == 0:
        seed = int(datetime.now().timestamp())
    np.random.seed(seed)
    tf.random.set_seed(seed)


def panic(msg: str) -> None:
    print(str)  # Maybe log
    sys.exit()


def check_for_missing_columns(data: pd.DataFrame, cols: Sequence[str], filename: str) -> None:
    missing: List[str] = [col for col in cols if col not in data.columns]
    if missing:
        raise KeyError(
            f'File {filename} is missing following columns f{missing}')


def is_numeric(s: pd.Series) -> bool:
    ''' For better redability of final code (why there is no s.is_numeric() :( )'''
    return pd.api.types.is_numeric_dtype(s)


class NetworkTaskType(Enum):
    REGRESSION = auto(),
    BINARY_CLASSIFICATION = auto(),
    MULTICLASS_CLASSIFICATION = auto()

    def __repr__(self) -> str:
        return f'{self.name}'

    @staticmethod
    def infer_from_data(data_out: pd.DataFrame) -> NetworkTaskType:

        # If there are multiple columns it can only be regression, if there are non numeric
        # columns an ValueError is thrown.
        if data_out.shape[1] > 1:
            if any(not is_numeric(data_out[col]) for col in data_out.columns):
                raise ValueError('Non numeric data detected for multi column output, '
                                 ' this script can only handle classification for single '
                                 ' column output.')
            return NetworkTaskType.REGRESSION

        # We now that there is only a single column so we can focus on it
        y: pd.Series = data_out.iloc[:, 0]

        # If there is a single column with non numeric data this is clearly classification
        # we only need to check if there are only two classes or more.
        if not is_numeric(y):
            if y.astype('category').cat.categories.size == 2:
                return NetworkTaskType.BINARY_CLASSIFICATION
            else:
                return NetworkTaskType.MULTICLASS_CLASSIFICATION

        # If there is a single column with numeric value we must decide if those numers encode
        # categories or are values for regression. We will concider numbers as category id's if
        # values are positive integers and there is less unique numbers than some given threshold.
        # What is important if there is a larger number, like 15 we assume that all values from
        # 0 to 15 can be set even if it is not present.
        CLASS_COUNT_THRESHOLD: Final[int] = 20
        if pd.api.types.is_float_dtype(y) or any(y < 0):
            return NetworkTaskType.REGRESSION
        class_count = max(y.nunique(dropna=True), y.max())
        if class_count == 2:
            return NetworkTaskType.BINARY_CLASSIFICATION
        elif class_count < CLASS_COUNT_THRESHOLD:
            return NetworkTaskType.MULTICLASS_CLASSIFICATION
        else:
            return NetworkTaskType.REGRESSION

# ==================================================================================================
#                                    DATA PREPROCESSING
# ==================================================================================================


class OneHotCategoryDecoder:

    idx_to_category_: Dict[int, str] = {}

    def __init__(self, encoded: OneHotEncodedData) -> None:
        for col in encoded.encoding_columns:
            if col not in encoded.data.columns:
                raise KeyError(f'Column {col} not present in data')
            idx: Any = encoded.data.columns.get_loc(col)
            if not isinstance(idx, (int, np.integer)):
                raise ValueError('Numpy column indexes are not integers.')
            self.idx_to_category_[int(idx)] = col

    def decode(self, idx: int) -> str:
        try:
            return self.idx_to_category_[idx]
        except KeyError:
            raise KeyError(
                f'Index {idx} not present in encoding, valid indexes f{self.idx_to_category_.keys()}')


@dataclass
class OneHotEncodedData:
    data: pd.DataFrame  # Dataframe with categorical columns repleaced with encoded ones
    encoding_columns: List[str]  # Columns generated by one hot encoding
    # Names of original categorical columns that are no longer presend in data
    categorical_columns: List[str]

    @staticmethod
    def encode(data: pd.DataFrame,
               categorical_cols: List[str],
               categories: Optional[List[str]] = None
               ) -> OneHotEncodedData:
        if not categorical_cols:
            return OneHotEncodedData(data=data.copy(), encoding_columns=[], categorical_columns=categorical_cols)
        encoded = pd.get_dummies(
            data, columns=categorical_cols, dummy_na=False)
        if not categories:
            generated_columns = [
                col for col in encoded.columns if col not in data.columns]
            return OneHotEncodedData(data=encoded, encoding_columns=generated_columns, categorical_columns=categorical_cols)
        # Add missing categories
        for category in categories:
            if category not in encoded.columns:
                encoded[category] = 0
        # Remove extra categories
        valid_columns: List[str] = list(data.columns) + categories
        to_drop = [col for col in encoded.columns if col not in valid_columns]
        if to_drop:
            encoded.drop(columns=to_drop, inplace=True)
        return OneHotEncodedData(data=encoded, encoding_columns=categories, categorical_columns=categorical_cols)

    @staticmethod
    def encode_like(data: pd.DataFrame, reference: OneHotEncodedData) -> OneHotEncodedData:
        return OneHotEncodedData.encode(data, reference.categorical_columns, reference.encoding_columns)


class BinaryCategoryDecoder:

    # Category names corresponding to encoding 0 and 1
    category_: Tuple[str, str]

    def __init__(self, data: BinaryEncodedData) -> None:
        self.category_ = data.category

    def decode(self, code: int):
        if code not in [0, 1]:
            raise KeyError(
                f'Binary encoding only uses keys 0 and 1, {code} given instead')
        return self.category_[code]


@dataclass
class BinaryEncodedData:
    data: np.ndarray  # Array with 0 and 1 that indicates binary classification
    # Category names corresponding to encoding 0 and 1
    category: Tuple[str, str]
    name: str  # Name of trait that is encoded

    @staticmethod
    def encode(data: pd.Series[pd.CategoricalDtype], name: str) -> BinaryEncodedData:
        if len(data.cat.categories) != 2:
            raise ValueError(
                'Attempted to use binary encoding on data with more than 2 categories.')
        encoded_data: np.ndarray = data.cat.codes.to_numpy(dtype=np.float32)
        categories: Tuple[str, str] = (
            data.cat.categories[0], data.cat.categories[1])
        return BinaryEncodedData(data=encoded_data, category=categories, name=name)

    @staticmethod
    def encode_like(data: pd.Series[pd.CategoricalDtype], reference: BinaryEncodedData) -> BinaryEncodedData:
        if len(data.cat.categories) != 2:
            raise ValueError(
                'Attempted to use binary encoding on data with more than 2 categories.')

        # Check if category mappings are same as in reference, if they are inverted we can fix it
        # but if categories are different we must throw an exception.
        invert_encoding: bool = False
        if data.cat.categories[0] != reference.category[0]:
            if data.cat.categories[0] == reference.category[1] and data.cat.categories[1] == reference.category[0]:
                invert_encoding = True
            else:
                raise ValueError(
                    'Incompatibile categories between new data and reference.')

        # Encode new data, all other fields are taken from reference.
        encoded_data: np.ndarray = data.cat.codes.to_numpy(dtype=np.float32)
        if invert_encoding:
            encoded_data = 1-encoded_data
        return BinaryEncodedData(encoded_data, reference.category, reference.name)


@dataclass
class InputData:

    train_input: np.ndarray
    test_input: np.ndarray
    scaler: StandardScaler
    columns: List[str]

    def get_training_sample_count(self) -> int:
        return self.train_input.shape[0]

    def get_test_sample_count(self) -> int:
        return self.test_input.shape[0]

    def get_feature_count(self) -> int:
        return self.train_input.shape[1]

    @staticmethod
    def prepare_inputs(data_train: pd.DataFrame, data_test: pd.DataFrame, input_cols: Sequence[str]) -> InputData:
        train_raw: pd.DataFrame = data_train[list(input_cols)].copy()
        test_raw: pd.DataFrame = data_test[list(input_cols)].copy()

        categorical_cols: List[str] = [
            col for col in input_cols if not is_numeric(train_raw[col])]
        train_encoded: OneHotEncodedData = OneHotEncodedData.encode(
            train_raw, categorical_cols)
        test_encoded: OneHotEncodedData = OneHotEncodedData.encode_like(
            test_raw, train_encoded)
        train_np: np.ndarray = train_encoded.data.to_numpy(
            dtype=np.float32, copy=True)
        test_np: np.ndarray = test_encoded.data.to_numpy(
            dtype=np.float32, copy=True)

        scaler: StandardScaler = StandardScaler().fit(train_np)
        train_np = scaler.transform(train_np)
        test_np = scaler.transform(test_np)

        return InputData(train_input=train_np,
                         test_input=test_np,
                         scaler=scaler,
                         columns=list(train_encoded.data.columns))


@dataclass
class OutputData:
    train_out: np.ndarray
    test_out: np.ndarray
    decoder: BinaryCategoryDecoder | OneHotCategoryDecoder | None

    def get_output_size(self) -> int:
        return self.train_out.shape[1]

    def get_category_count(self):
        return self.train_out.shape[1] if self.train_out.shape[1] > 1 else 2

    @staticmethod
    def prepare_outputs(data_train: pd.DataFrame,
                        data_test:  pd.DataFrame,
                        output_cols: Sequence[str],
                        task: NetworkTaskType) -> OutputData:
        train_raw: pd.DataFrame = data_train[list(output_cols)].copy()
        test_raw: pd.DataFrame = data_test[list(output_cols)].copy()

        # Regression task do not require any preprocessing
        if task == NetworkTaskType.REGRESSION:
            train_np: np.ndarray = train_raw.to_numpy(
                dtype=np.float32, copy=True)
            test_np: np.ndarray = test_raw.to_numpy(
                dtype=np.float32, copy=True)
            return OutputData(train_np, test_np, None)

        # For classification task there must be only a single output column
        if len(output_cols) > 1:
            raise ValueError(
                'Classification task cannot have more than one output column.')
        out_col = output_cols[0]

        train_categorical = train_raw[out_col].astype('category')
        test_categorical = test_raw[out_col].astype('category')
        if len(train_categorical.cat.categories) == 2:
            train_encoded = BinaryEncodedData.encode(
                train_categorical, out_col)
            test_encoded = BinaryEncodedData.encode_like(
                test_categorical, train_encoded)
            decoder = BinaryCategoryDecoder(train_encoded)
            return OutputData(train_encoded.data, test_encoded.data, decoder)

        # For one hot encoding we use train and test raw as OneHotEncodedData was designed for
        # more general cases than BinaryEncodedData
        train_encoded = OneHotEncodedData.encode(train_raw, list(output_cols))
        test_encoded = OneHotEncodedData.encode(test_raw, list(output_cols))
        decoder = OneHotCategoryDecoder(train_encoded)
        return OutputData(train_encoded.data.to_numpy(dtype=np.float32), test_encoded.data.to_numpy(dtype=np.float32), decoder)

# ==================================================================================================
#                                        NETWORK MODEL
# ==================================================================================================


class NeuralNetwork:

    data_in: InputData
    data_out: OutputData
    task: NetworkTaskType
    model: krs.Model
    history: None | krs.callbacks.History

    def __init__(self, data_in: InputData, data_out: OutputData, task: NetworkTaskType):
        self.data_in = data_in
        self.data_out = data_out
        self.task = task
        self.history = None

        # We infer network architecture based on traning data size
        feature_count = data_in.get_feature_count()
        sample_count = data_in.get_training_sample_count()
        output_size = data_out.get_output_size()
        layer_sizes = NeuralNetwork.infer_architecture_from_data(
            sample_count, feature_count)

        # We create input layer
        input_layer = krs.Input(shape=(feature_count,), name='inputs')

        # Now we chain input and hidden layers, each hidden layer will have a
        # corresponding dropout and batch normalization
        last_layer = input_layer
        for layer_id, layer_size in enumerate(layer_sizes, start=1):
            last_layer = Dense(layer_size,
                               activation='relu',
                               name=f'dense_{layer_id}')(last_layer)
            last_layer = BatchNormalization(
                name=f'batch_norm_{layer_id}')(last_layer)
            last_layer = Dropout(0.1, name=f'dropout_{layer_id}')(last_layer)

        # Finally we add an output layer
        activation: str = {NetworkTaskType.REGRESSION: 'linear',
                           NetworkTaskType.BINARY_CLASSIFICATION: 'sigmoid',
                           NetworkTaskType.MULTICLASS_CLASSIFICATION: 'softmax'}[task]
        loss: str = {NetworkTaskType.REGRESSION: 'mse',
                     NetworkTaskType.BINARY_CLASSIFICATION: 'binary_crossentropy',
                     NetworkTaskType.MULTICLASS_CLASSIFICATION: 'categorical_crossentropy'}[task]
        metrics: List[Any] = {NetworkTaskType.REGRESSION: ['mae'],
                              NetworkTaskType.BINARY_CLASSIFICATION: ['accuracy', krs.metrics.AUC(name='auc')],
                              NetworkTaskType.MULTICLASS_CLASSIFICATION: ['accuracy']}[task]
        output_layer = Dense(output_size, activation,
                             name='output')(last_layer)

        self.model = krs.Model(
            inputs=input_layer, outputs=output_layer, name='generic_ffn')
        self.model.compile(optimizer=krs.optimizers.Adam(learning_rate=LEARNING_RATE),  # type: ignore
                           loss=loss,
                           metrics=metrics)

    def train(self, model_out: str, validation_split: float, epochs: int) -> None:
        if self.history:
            raise ValueError(f'Attempting to train already trained network, '
                             'this functionality is not provided by this implementation')
        callbacks: List[krs.callbacks.Callback] = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=0, verbose=0, min_lr=MIN_LEARNING_RATE,),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ModelCheckpoint(monitor='val_loss', filepath=model_out, save_best_only=True, verbose=0)
        ]
        batch_size = NeuralNetwork.get_reasonable_batch_size(self.data_in.get_training_sample_count())
        self.history = self.model.fit(self.data_in.train_input,
                                      self.data_out.train_out,
                                      validation_split=validation_split,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      callbacks=callbacks,
                                      shuffle=True
                                      )

    def get_history(self)-> krs.callbacks.History:
        if self.history:
            return self.history
        raise ValueError('Attempted to acces history before training network.')

    def plot_history(self):
        if not self.history:
            raise ValueError('Attempted to acces history before training network.')
        history_dict = self.history.history
        
        metrics = [m for m in history_dict.keys() if not m.startswith('val_')]

        for metric in metrics:
            plt.figure(figsize=(8, 5))
            
            plt.plot(history_dict[metric], label=f'Train {metric}')
            if f'val_{metric}' in history_dict:
                plt.plot(history_dict[f'val_{metric}'], label=f'Validation {metric}')
            
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.title(f'Training and Validation {metric.capitalize()}')
            plt.legend()
            plt.grid(True)
            plt.show()

    @staticmethod
    def get_reasonable_batch_size(sample_count: int) -> int:
        return int(max(32, min(256, sample_count//20)))

    @staticmethod
    def infer_architecture_from_data(sample_count: int, feature_count: int) -> List[int]:
        # For small data there is no reason for complex networks, we will use smalles one possible.
        if sample_count < 200:
            return [MIN_LAYER_SIZE]

        # Generate layer sizes. Layer count depend on number of samples but is not greater
        # than MAX_LAYER_COUNT and not less than 1. Each consecutive layer is half size of
        # pervious one but not smaller than MIN_LAYER_SIZE. Size of first layer is  calculated
        # based on feature count.
        base_layer_size = max(MIN_LAYER_SIZE, min(
            MAX_LAYER_SIZE, math.floor(feature_count*FEATURES_TO_SIZE_FACTOR)))
        sizes = [base_layer_size]
        sample_size_per_layer_limit = SAMPLES_TO_LAYER_BASE
        while sample_count < sample_size_per_layer_limit and len(sizes) <= MAX_LAYER_COUNT:
            sizes.append(max(MIN_LAYER_SIZE, sizes[-1]//2))
            sample_size_per_layer_limit *= SAMPLES_TO_LAYER_STEP_MULTIPLIER

        return sizes


# ==================================================================================================
#                                           MAIN
# ==================================================================================================


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg: Config = Config.from_args()

    # Reading data
    train_dataframe: pd.DataFrame = pd.read_csv(cfg.train_csv)
    test_dataframe: pd.DataFrame = pd.read_csv(cfg.test_csv)

    # Checking for missing columns
    try:
        check_for_missing_columns(
            train_dataframe, cfg.input_cols, cfg.train_csv)
        check_for_missing_columns(
            train_dataframe, cfg.output_cols, cfg.train_csv)
        check_for_missing_columns(
            test_dataframe, cfg.input_cols, cfg.test_csv)
        check_for_missing_columns(
            test_dataframe, cfg.output_cols, cfg.test_csv)
    except KeyError as err:
        panic(f'Missing columns found - {err}')

    task: NetworkTaskType = NetworkTaskType.infer_from_data(
        train_dataframe[list(cfg.output_cols)])
    input_data: InputData = InputData.prepare_inputs(
        train_dataframe, test_dataframe, cfg.input_cols)
    output_data: OutputData = OutputData.prepare_outputs(
        train_dataframe, test_dataframe, cfg.output_cols, task)
    net: NeuralNetwork = NeuralNetwork(input_data, output_data, task)
    print(f'Inferred task : {task}')
    if task in [NetworkTaskType.BINARY_CLASSIFICATION, NetworkTaskType.MULTICLASS_CLASSIFICATION]:
        print(f'Category count : {output_data.get_category_count()}')
    print(f'Train samples - {input_data.get_training_sample_count()}')
    print(f'Test samples - {input_data.get_test_sample_count()}')
    if cfg.history_out:
        print(f'Training history will be saved to {cfg.history_out}')
    else:
        print(f'Training history will not be saved.')
    print(f' TRAIN OUT SHAPE = {output_data.train_out.shape}')
    if not input('Proceed with training (y/n)>')=='y':
        sys.exit()    
    net.train(cfg.model_out, cfg.validation_split, cfg.epochs)
    net.plot_history()
    if cfg.history_out:
        history = pd.DataFrame(net.get_history().history)
        history.to_csv(cfg.history_out)

if __name__ == '__main__':
    main()
