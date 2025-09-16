#!/usr/bin/env python3
# coding: utf-8

'''
This scripts creates a mask that shows which part of video frame are worth
analyzing. The idea is that only part that undergo change should be considered
interesting and for other we can assume that they contain same info as last time.
This is simillar in underlying concept to time redundancy compression algorithms
like MPEG.
'''

# ==================================================================================================
#                                             IMPORTS
# ==================================================================================================
# Future imports
from __future__ import annotations

# Basic python imports
import argparse
import pathlib
from dataclasses import dataclass
from typing import Optional, Sequence

# ==================================================================================================
#                             CONFIGURATION AND ARGUMENT PARSING
# ==================================================================================================

def arg_file_path(txt: str)-> pathlib.Path:
    path = pathlib.Path(txt)
    
    # If path do not exist throw an exception with absolute path so that we know 
    # what exactly we tried to open.
    if not path.exists():
        raise argparse.ArgumentTypeError(f'File {path.resolve()} do not exist.')
    
    return path

@dataclass(frozen=True)
class Config:

    video_path: pathlib.Path # Path to video that will be processed

    @staticmethod
    def from_args(argv: Optional[Sequence[str]] = None) -> Config:
        prsr = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Declare arguments
        prsr.add_argument('--video_path', required=True,
                          type=arg_file_path, help='Path to video that will be processed')
        
         # Parsing arguments
        args: argparse.Namespace = prsr.parse_args(argv)

        # Returning config
        return Config(
            video_path=args.video_path
        )

# ==================================================================================================
#                                           MAIN
# ==================================================================================================


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg: Config = Config.from_args()

if __name__ == '__main__':
    main()