# references
# - https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/detect_picamera.py
"""Example using TF Lite to detect objects with the Raspberry Pi camera."""

from __future__ import absolute_import, division, print_function

import argparse
import re
import time

import cv2
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def annotate_objects(cv2_frame, results, labels):
    """Draws the bounding box and label for each object in the results."""
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        # Overlay the box, label, and score on the camera preview
        cv2_frame = cv2.rectangle(cv2_frame, (xmin, ymin), (xmax, ymax),
                                  color=(0, 0, 255),
                                  thickness=3)
        cv2_frame = cv2.putText(
            cv2_frame,
            f'{labels[obj["class_id"]]} {obj["score"]:.2f}',
            (xmin + 5, ymin + 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 0, 255),
            thickness=2)

    return cv2_frame


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',
                        help='File path of .tflite file.',
                        required=True)
    parser.add_argument('--labels',
                        help='File path of labels file.',
                        required=True)
    parser.add_argument('--threshold',
                        help='Score threshold for detected objects.',
                        required=False,
                        type=float,
                        default=0.5)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details(
    )[0]['shape']

    capture = cv2.VideoCapture(0)

    while (True):
        ret, cv2_frame = capture.read()
        image = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).resize((input_width, input_height),
                                              Image.ANTIALIAS)

        start_time = time.monotonic()
        results = detect_objects(interpreter, image, args.threshold)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        print(f'elapsed [ms]: {elapsed_ms:.1f}')
        cv2_frame = annotate_objects(cv2_frame, results, labels)

        cv2.imshow('detection result', cv2_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
