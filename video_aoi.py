#!/usr/bin/env python3
# coding: utf-8

'''
This scripts creates a mask that shows which part of video frame are worth
analyzing. The idea is that only part that undergo change should be considered
interesting and for other we can assume that they contain same info as last time.
This is simillar in underlying concept to time redundancy compression algorithms
like MPEG.
CONCEPT FAILED BUT A LOT OF USEFULL CODE SNIPPETS - KEEP
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
from typing import Optional, Sequence, Tuple

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


def arg_scale(txt: str) -> float:
    try:
        val: float = float(txt)
        if 0.0 < val < 1.0:
            return val
        raise argparse.ArgumentTypeError(
            f'Scale argument must be in range between 0 and 1, value {val} given instead.')
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Unable to convert "{txt}" to valid float.')


@dataclass(frozen=True)
class Config:

    video_path: pathlib.Path  # Path to video that will be processed
    fps_limit: int  # Maximum number of frames per second in playback, if set to zero no limit applied
    # Scale of AOI map related to source image (must be between 0.0 an 1.0)
    map_scale: float

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
        prsr.add_argument('--map_scale', default=1.0,
                          type=arg_scale, help='Scale of AOI map related to source image (must be between 0.0 an 1.0)')

        # Parsing arguments
        args: argparse.Namespace = prsr.parse_args(argv)

        # Returning config
        return Config(
            video_path=args.video_path,
            fps_limit=args.fps_limit,
            map_scale=args.map_scale
        )

# ==================================================================================================
#                                         HELPERS
# ==================================================================================================


def panic(msg: str) -> None:
    print(f'Critical - {msg}')
    sys.exit()


@dataclass
class ImageSize:

    x: int  # Horizontal size (width) of image in pixels
    y: int  # Vertical size (height) of image in pixels

    def to_cv_tuple(self) -> Tuple[int, int]:
        return (self.y, self.x)
    
    def to_np_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @staticmethod
    def from_tuple(src: Tuple[int, ...]) -> ImageSize:
        if len(src) < 2:
            raise ValueError(
                f'Attempting to generate image size for one dimensional vector.')
        return ImageSize(src[0], src[1])

    def can_encapsulate(self, other: ImageSize) -> bool:
        return self.x >= other.x and self.y >= other.y

    def is_same_ratio(self, other: ImageSize) -> bool:
        return self.x//other.x == self.y//other.y

    def scale(self, scale: float) -> ImageSize:
        return ImageSize(x=int(self.x*scale), y=int(self.y*scale))

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

    def get_size(self) -> ImageSize:
        return ImageSize.from_tuple(self.data.shape)

    @staticmethod
    def from_ndarray(frame_array: npt.NDArray[np.uint8], nr: Optional[int] = None):
        data = frame_array.copy()
        data.setflags(write=False)
        return VideoFrame(data=data, timestamp=time.time(), nr=nr if nr else 0)

# ==================================================================================================
#                                     FRAME PROCESSORS
# ==================================================================================================


class FrameProcessor:

    def process(self, frame: VideoFrame) -> VideoFrame:
        ''' Dummy processor simply returns a copy of source frame'''
        return frame.copy()

class BGRtoGrayscale(FrameProcessor):

    def process(self, frame: VideoFrame) -> VideoFrame:
        gray = cv.cvtColor(frame.data, cv.COLOR_BGR2GRAY)
        return VideoFrame(data=np.asarray(gray, dtype=np.uint8),
                          timestamp=frame.timestamp,
                          nr=frame.nr)

class AOIdetector(FrameProcessor):

    # Last frame data. It can be None when processing first frame or reset was forced
    last_frame_: npt.NDArray[np.uint8] | None = None
    map_size_: ImageSize  # Target size of map and downscaled frame
    # Number of frames after which we again consider whole image AOI, if set to zero attention mechanism is disabled
    attention_reset_period_: int
    # Number of frames remining before whole image will be analyzed
    attention_remining_: int
    diff_threshold_: int # Difference in brightnes for which we consider pixel a AOI

    def __init__(self, map_size: ImageSize, diff_threshold:int, attention_reset_period: int = 0) -> None:
        super().__init__()
        self.map_size_ = map_size
        self.attention_reset_period_ = attention_reset_period
        self.attention_remining_ = attention_reset_period
        self.diff_threshold_ = diff_threshold

    def process(self, frame: VideoFrame) -> VideoFrame:
        # Scaling recieved frame to required map size, if image would have
        # to be upscaled or have different ratio we throw ValueError
        source_size: ImageSize = frame.get_size()
        if not source_size.can_encapsulate(self.map_size_):
            raise ValueError(
                f'AOImapper - map size {self.map_size_} cannot be encapsulate by image with size of {source_size}')
        if not self.map_size_.is_same_ratio(source_size):
            raise ValueError(
                f'AOImapper - map size {self.map_size_} is of different ratio than source {source_size}')
        resized_frame = cv.resize(frame.data, self.map_size_.to_cv_tuple())

        # If this is first frame we simply return whole frame as AOI
        if self.last_frame_ is None:
            self.last_frame_ = resized_frame # type: ignore
            return VideoFrame(data=np.asarray(np.ones(self.map_size_.to_np_tuple())*255, dtype=np.uint8),
                              timestamp=frame.timestamp,
                              nr=frame.nr)

        # If attention mechanism is active and we reached the limit of attention frames we return
        # whole frame as AOI
        if self.attention_reset_period_ > 0 and self.attention_remining_ <= 0:
            self.attention_remining_ = self.attention_reset_period_
            self.last_frame_ = resized_frame # type: ignore
            return VideoFrame(data=np.asarray(np.ones(self.map_size_.to_np_tuple())*255, dtype=np.uint8),
                              timestamp=frame.timestamp,
                              nr=frame.nr)

        # If neither of pervious special cases occured we decrement remining attention,
        # if attention mechanism is used, and calculate AOI. Pixels for which brightness value 
        # changed over given threshold will be considered AOI
        if self.attention_reset_period_ > 0:
            self.attention_remining_-= 1
        mask = np.abs(self.last_frame_ - resized_frame) > self.diff_threshold_
        aoi_map = np.where(mask, 255, 0).astype(np.uint8)
        self.last_frame_ = resized_frame # type: ignore
        return VideoFrame(data=np.asarray(aoi_map, dtype=np.uint8),
                          timestamp=frame.timestamp,
                          nr=frame.nr)

        

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

    def spin(self):
        cv.imshow(self.window_name_, self.frame_.data)

        # wait 1 ms for a key; quit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt('User pressed "q" to close GUI.')

    def set_frame(self, frame: VideoFrame) -> None:
        self.frame_ = frame

# ==================================================================================================
#                                           MAIN
# ==================================================================================================


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg: Config = Config.from_args()

    source: VideoSource = VideoSource(
        str(cfg.video_path.resolve()), cfg.fps_limit)
    display: SimpleDisplay = SimpleDisplay('Video (press "q" to quit)')
    to_grayscale: BGRtoGrayscale = BGRtoGrayscale()
    aoi_detector: AOIdetector = AOIdetector(
        source.frame.get_size().scale(cfg.map_scale), 240)

    try:
        while True:
            source.spin()
            gray: VideoFrame = to_grayscale.process(source.frame)
            aoi_map: VideoFrame = aoi_detector.process(gray)
            display.set_frame(aoi_map)
            display.spin()

    except (EOFError, KeyboardInterrupt) as e:
        print(f'Display stopped, reason : {e}')



if __name__ == '__main__':
    main()
