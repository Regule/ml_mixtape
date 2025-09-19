#!/usr/bin/env python3
# coding: utf-8

'''
A very simple example of how to use pretrained YOLO.
Pretrained weights may be downloaded from https://www.kaggle.com/datasets/shivam316/yolov3-weights/data
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
from typing import Optional, Sequence, Tuple, List

# Library specific imports
import cv2
import numpy as np
import numpy.typing as npt
import cv2.typing as cvt

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

    input_image: str  # Path to input image
    yolo_config: str  # Path to yolo config file
    weights_file: str  # Path to file with pretrained weights
    classes_file: str  # Path to file with classes
    scale: float  # Scale of image that will be fed to network

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
class PixelCoord:

    x: int
    y: int

    def to_cv_tuple(self) -> Tuple[int, int]:
        return (self.y, self.x)

    def to_np_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class ImageSize(PixelCoord):

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
#                                          YOLO WRAPPER
# ==================================================================================================


@dataclass(frozen=True)
class YoloCategory:

    name: str  # Name of class
    colour: Tuple[int, int, int]


@dataclass
class YoloDetectedObject:

    category: YoloCategory
    confidence: float
    position: PixelCoord
    size: ImageSize


class YoloWrapper:

    YOLO_INPUT_SIZE: ImageSize = ImageSize(x=416, y=416)
    YOLO_INPUT_MEAN: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    categories_: List[YoloCategory]

    def __init__(self, yolo_config: str, weights_file: str, classes_file: str) -> None:

        # Reading class names from file and assigning colors to each of them
        categories: List[str] = []
        with open(classes_file, 'r') as f:
            categories = [line.strip() for line in f.readlines()]
        if not categories:
            raise ValueError(f'Unable to read class names from {classes_file}')
        colours: npt.NDArray[np.uint8] = np.random.randint(
            0, 255, size=(len(categories), 3), dtype=np.uint8)
        self.categories_ = []
        for name, colour in zip(categories, colours):
            self.categories_.append(YoloCategory(name, (colour[0], colour[1], colour[2])))

        # TODO: Error handling for reading net
        self.net_: cv2.dnn.Net = cv2.dnn.readNet(weights_file, yolo_config)
        layer_names = self.net_.getLayerNames()
        self.output_layers_ = [layer_names[i - 1]
                               for i in self.net_.getUnconnectedOutLayers()]

    def detect_objects(self, img: npt.NDArray[np.uint8], scale: float = 1.0)-> List[YoloDetectedObject]:
        # Prepare input based on image, assuring correct typing
        blob_raw: cvt.MatLike = cv2.dnn.blobFromImage(img,
                                                      scalefactor=scale,
                                                      size=YoloWrapper.YOLO_INPUT_SIZE.to_cv_tuple(),
                                                      mean=YoloWrapper.YOLO_INPUT_MEAN,
                                                      swapRB=True,
                                                      crop=False,
                                                      ddepth=cv2.CV_32F)
        blob: npt.NDArray[np.float32] = np.asarray(blob_raw, dtype=np.float32)

        # Run network
        self.net_.setInput(blob)
        # TODO: Assure return type
        output: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32],
                      npt.NDArray[np.float32]] = self.net_.forward(self.output_layers_)

        # Build object list
        detected_objects: List[YoloDetectedObject] = []
        image_size: ImageSize = ImageSize.from_tuple(img.shape)
        for out in output:
            for detection in out:
                category_id = np.argmax(detection[5:])
                confidence: float = detection[5:][category_id]
                if confidence > 0.5:
                    # TODO: Make this clearer 
                    center_x = int(detection[0] * image_size.y)
                    center_y = int(detection[1] * image_size.x)
                    w = int(detection[2] * image_size.y)
                    h = int(detection[3] * image_size.x)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    detected_objects.append(YoloDetectedObject(
                        category=self.categories_[category_id],
                        confidence=confidence,
                        position=PixelCoord(x,y),
                        size=ImageSize(w,h)
                    ))
        return detected_objects

    def run_network_(self):
        pass

# ==================================================================================================
#                                           MAIN
# ==================================================================================================

def new_draw_bounding_box(img: npt.NDArray[np.uint8], obj: YoloDetectedObject):

    print(obj.category.colour)
    cv2.rectangle(img, obj.position.to_cv_tuple(), obj.size.to_cv_tuple(),
                  color=(0,0,0), thickness=2)

    text_location = PixelCoord(x= obj.position.x-10, y=obj.position.y-10)
    cv2.putText(img, obj.category.name, text_location.to_cv_tuple(),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

def new_main():
    cfg: Config = Config.from_args()

    image = cv2.imread(cfg.input_image)
    if image is None:
        panic(f'Unable to open {cfg.input_image} as image')
    
    image = np.asarray(image, dtype=np.uint8)
    yolo = YoloWrapper(cfg.yolo_config, cfg.weights_file, cfg.classes_file)
    objects = yolo.detect_objects(image, cfg.scale)
    for obj in objects:
        new_draw_bounding_box(image, obj)
 
    cv2.imshow("object detection", image) 
    cv2.waitKey()


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

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg: Config = Config.from_args()

    image = cv2.imread(cfg.input_image)
    if image is None:
        panic(f'Unable to open {cfg.input_image} as image')
    image_size: ImageSize = ImageSize.from_tuple(image.shape)  # type: ignore

    # We creating blob for network from our image. We have to use "ignore" for
    # typehints as linter do not understand that image cannot be None at this point
    blob = cv2.dnn.blobFromImage(image,  # type: ignore
                                 scalefactor=cfg.scale,
                                 size=(416, 416),
                                 mean=(0, 0, 0),
                                 swapRB=True,
                                 crop=False,
                                 ddepth=cv2.CV_32F)  # type: ignore

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

    # --------------------
    # Running inference

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))  # type: ignore

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    print(f'TYPE OF OUTS - {type(outs)}')
    for out in outs:
        print(f'TYPE OF OUT - {type(out)}')
        for detection in out:
            # print(f'TYPE OF DETECTION - {type(detection)} ({detection.dtype})')
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
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(image, class_ids[i], confidences[i], round(
            x), round(y), round(x+w), round(y+h), COLORS, classes)

    # display output image
    cv2.imshow("object detection", image)  # type: ignore

    # wait until any key is pressed
    cv2.waitKey()

    # release resources
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
