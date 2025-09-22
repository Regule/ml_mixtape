#!/usr/bin/env python3
#coding: utf-8

'''
This script is my first attempt at interacting with brain activity to speach 
dataset.
TODO: ACTIVE PROJECT
'''

#===================================================================================================
#                                             IMPORTS
#===================================================================================================
# Future imports
from __future__ import annotations

# Standard python imports
from dataclasses import dataclass
from typing import List, Final, Any, Dict
from pathlib import Path

# Specific library related imports
import h5py
import numpy as np
import numpy.typing as npt


#===================================================================================================
#                                         BRAIN SCAN
#===================================================================================================
@dataclass
class ScanRange:

    start: int # First channel in range
    end: int # Last channel in range

@dataclass
class BrainScan:


    phrase: str # Spoken phrase corresponding to brain ativity
    channels: npt.NDArray[np.float32] # Two dimensional array with multi channel scan


#===================================================================================================
#                                      BRAIN SCAN MANAGER 
#===================================================================================================

class BrainScanManager:

    CHANNELS_FEATURE_KEY: str = 'input_features'
    PHRASE_FEATURE_KEY: str = 'sentence_label'

    scans: List[BrainScan] = [] 

    def load_from_h5py(self, filename: str)-> None:
        h5py_path: Path = Path(filename)
        if not h5py_path.exists():
            raise ValueError(f'File {h5py_path.resolve()} do not exists')
        with h5py.File(h5py_path,'r') as h5py_file:

            # Hdf5 file consist of multiple individual scans 
            for scan_name, scan_data in h5py_file.items(): 
                if not scan_data.attrs:
                    continue
                phrase:str = scan_data.attrs[self.PHRASE_FEATURE_KEY][:]
                channels: npt.NDArray[np.float32] = scan_data[self.CHANNELS_FEATURE_KEY][:]
                self.scans.append(BrainScan(phrase, channels))


#===================================================================================================
#                                        MAIN 
#===================================================================================================

def main()-> None:
    a = BrainScanManager() 
    a.load_from_h5py('data/data_train.hdf5')

if __name__ == '__main__':
    main()
