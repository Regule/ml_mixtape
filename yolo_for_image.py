#!/usr/bin/env python3
# coding: utf-8

'''
A very simple example of how to use pretrained YOLO, pretrained weights can be downloaded 
from https://pjreddie.com/media/files/yolov3.weights
'''

# ==================================================================================================
#                                             IMPORTS
# ==================================================================================================
# Future imports
from __future__ import annotations

# Basic python imports
import pathlib
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# Library specific imports
import cv2
import numpy as np

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

    input_image: str # Path to input image
    yolo_config: str # Path to yolo config file
    weights_file: str # Path to file with pretrained weights
    classes_file: str # Path to file with classes 
    scale: float # Scale of image that will be fed to network

    @staticmethod
    def from_args(argv: Optional[Sequence[str]] = None) -> Config:
        prsr = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Declare arguments
        prsr.add_argument('--input_image', required=True,
                          type=arg_file_path, help='Path to input image')
        prsr.add_argument('--yolo_config', required=True,
                          type=arg_file_path, help='Path to yolo config file')
        prsr.add_argument('--weights_file', required=True,
                          type=arg_file_path, help='Path to file with pretrained weights')
        prsr.add_argument('--classes_file', required=True,
                          type=arg_file_path, help='Path to file with classes')
        prsr.add_argument('--scale', default=0.004,
                          type=arg_file_path, help='Scale of image that will be fed to network')


        # Parsing arguments
        args: argparse.Namespace = prsr.parse_args(argv)

        # Returning config
        return Config(
            input_image=args.input_image,
            yolo_config=args.yolo_config,
            weights_file=args.weights_file,
            classes_file=args.classes_file,
            scale=args.scale
        )

# ==================================================================================================
#                                         HELPERS
# ==================================================================================================


def panic(msg: str) -> None:
    print(f'Critical - {msg}')
    sys.exit()

# ==================================================================================================
#                                         IMAGE SIZE
# ==================================================================================================

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
#                                           MAIN
# ==================================================================================================

# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, COLORS, classes):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg: Config = Config.from_args()

    image = cv2.imread(cfg.input_image)
    if image is None:
        panic(f'Unable to open {cfg.input_image} as image')
    image_size: ImageSize = ImageSize.from_tuple(image.shape)# type: ignore

    # We creating blob for network from our image. We have to use "ignore" for
    # typehints as linter do not understand that image cannot be None at this point
    blob = cv2.dnn.blobFromImage(image,  # type: ignore
                                 scalefactor=cfg.scale, 
                                 size=(416,416),
                                 mean=(0,0,0),
                                 swapRB=True,
                                 crop=False,
                                 ddepth=cv2.CV_32F) # type: ignore


    # read class names from text file
    classes = None
    with open(cfg.classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # read pre-trained model and config file
    net = cv2.dnn.readNet(cfg.weights_file, cfg.yolo_config)

    # set input blob for the network
    net.setInput(blob)

    #--------------------
    # Running inference 

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net)) # type: ignore 

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image_size.y)
                center_y = int(detection[1] * image_size.x)
                w = int(detection[2] * image_size.y)
                h = int(detection[3] * image_size.x)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), COLORS, classes)

    # display output image    
    cv2.imshow("object detection", image) # type: ignore

    # wait until any key is pressed
    cv2.waitKey()
        

    # release resources
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
