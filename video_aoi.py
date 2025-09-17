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
import time
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

# Library specific imports
import numpy as np
import numpy.typing as npt
import cv2 as cv

# ==================================================================================================
#                             CONFIGURATION AND ARGUMENT PARSING
# ==================================================================================================


def arg_file_path(txt: str) -> pathlib.Path:
    path = pathlib.Path(txt)

    # If path do not exist throw an exception with absolute path so that we know
    # what exactly we tried to open.
    if not path.exists():
        raise argparse.ArgumentTypeError(
            f'File {path.resolve()} do not exist.')

    return path


def arg_fps(txt: str) -> int:
    try:
        val: int = int(txt)
        if val < 1:
            raise argparse.ArgumentTypeError(
                f'FPS must be a positive value, value {val} given instead')
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Unable to convert "{txt}" to valid integer')


@dataclass(frozen=True)
class Config:

    video_path: pathlib.Path  # Path to video that will be processed
    fps_limit: int  # Maximum number of frames per second in playback, if set to zero no limit applied

    @staticmethod
    def from_args(argv: Optional[Sequence[str]] = None) -> Config:
        prsr = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Declare arguments
        prsr.add_argument('--video_path', required=True,
                          type=arg_file_path, help='Path to video that will be processed')
        prsr.add_argument('--fps_limit', default=0,
                          type=arg_fps, help='Maximum number of frames per second in playback')

        # Parsing arguments
        args: argparse.Namespace = prsr.parse_args(argv)

        # Returning config
        return Config(
            video_path=args.video_path,
            fps_limit=args.fps_limit
        )

# ==================================================================================================
#                                         HELPERS
# ==================================================================================================


def panic(msg: str) -> None:
    print(f'Critical - {msg}')
    sys.exit()

# ==================================================================================================
#                                         VIDEO FRAME
# ==================================================================================================


@dataclass(frozen=True)
class VideoFrame:
    data: npt.NDArray[np.uint8]  # A numpy array with frame data
    timestamp: float  # Timestamp of frame cration in seconds
    nr: int  # Frame number, if frames are not counted set to zero

    def copy(self):
        copied_data = self.data.copy()
        copied_data.setflags(write=False)
        return VideoFrame(copied_data, self.timestamp, self.nr)

    @staticmethod
    def from_ndarray(frame_array: npt.NDArray[np.uint8], nr: Optional[int] = None):
        data = frame_array.copy()
        data.setflags(write=False)
        return VideoFrame(data=data, timestamp=time.time(), nr=nr if nr else 0)

# ==================================================================================================
#                                     FRAME PROCESSORS
# ==================================================================================================


class FrameProcessor:
    ''' Instances of this class are used for processing frames. The correct usage is:

        processor.prepare(frame)
        processed_frame = processor.process(frame)

        First step adjusts dynamic parameters of processor based on difference between frames
        and second one applies changes to frame.
        IMPORTANT - processors do not change frame given as an argument but return updated version.
    '''

    # Last frame data. It can be None when processing first frame or reset was forced
    last_frame_: npt.NDArray[np.uint8] | None = None

    def prepare(self, frame: VideoFrame) -> None:
        ''' If processor have any dynamic properties that must be adjusted for new frame
         this is a place to set them. For most simple processors this method does nothing '''
        pass

    def process(self, frame: VideoFrame) -> VideoFrame:
        ''' Dummy processor simply returns a copy of source frame'''
        return frame.copy()

# ==================================================================================================
#                                    VIDEO SOURCE AND SINK
# ==================================================================================================


class VideoSource:

    vid_: cv.VideoCapture  # Source from which video will be read
    filename_: str  # Name of video file, used for verbose exceptions
    # Minimum time between updated in seconds (fps_limit)
    min_update_period_: float
    frame: VideoFrame  # Last frame read

    def __init__(self, filename: str, fps_limit: int) -> None:

        # Opening video file
        self.filename_ = filename
        self.vid_ = cv.VideoCapture(filename)
        if not self.vid_.isOpened():
            raise FileNotFoundError(
                f'File {filename} is not valid video file.')

        # Attempting initial capture, nr of first frame is set to 1
        capture_sucess, frame = self.vid_.read()
        if not capture_sucess:
            raise Exception(
                f'Opened file {self.filename_} sucessfuly but unable to read frames from it.')
        self.frame = VideoFrame.from_ndarray(
            np.asarray(frame, dtype=np.uint8), 1)

        # Setting FPS limit, if 0 is given no limit is set
        if fps_limit < 0:
            raise ValueError(
                f'FPS limit must be positive value or zero if disabled. Value{fps_limit} given instead.')
        elif fps_limit == 0:
            self.min_update_period_ = 0.0
        else:
            self.min_update_period_ = 1.0/fps_limit

    def spin(self) -> None:
        if time.time() - self.frame.timestamp < self.min_update_period_:
            return
        self.update_frame_()

    def update_frame_(self):
        capture_sucess, frame = self.vid_.read()
        if not capture_sucess:
            if self.frame.nr == 1:
                raise EOFError(
                    f'Reached end of video file {self.filename_} on frame {self.frame.nr}'
                    ' this may indicate that an image file was opened as video source.')
            else:
                raise EOFError(
                    f'Reached end of video file {self.filename_} on frame {self.frame.nr}')
        self.frame = VideoFrame.from_ndarray(
            np.asarray(frame, dtype=np.uint8), self.frame.nr+1)


class SimpleDisplay:

    window_name_: str  # Name of opencv window which this display will use
    frame_: VideoFrame  # Frame that will be displayed

    def __init__(self, window_name: str) -> None:
        self.window_name_ = window_name

    def __del__(self):
        cv.destroyWindow(self.window_name_)

    def spin(self):
        cv.imshow(self.window_name_, self.frame_.data)

        # wait 1 ms for a key; quit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt('User pressed "q" to close GUI.')

    def set_frame(self, frame: VideoFrame)->None:
            self.frame_ = frame

# ==================================================================================================
#                                           MAIN
# ==================================================================================================


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg: Config = Config.from_args()

    try:
        source: VideoSource = VideoSource(str(cfg.video_path.resolve()), cfg.fps_limit)
        display: SimpleDisplay = SimpleDisplay('Video (press "q" to quit)')

        try:
            while True:
                source.spin()
                display.set_frame(source.frame)
                display.spin()

        except (EOFError, KeyboardInterrupt) as e:
            print(f'Display stopped, reason : {e}')
    except Exception as e:
        # An exception happend that we are unable to handle
        panic(str(e))


if __name__ == '__main__':
    main()
