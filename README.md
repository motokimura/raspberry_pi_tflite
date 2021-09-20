# raspberry_pi_tflite

This repository is a slightly modified version of
[TensorFlow Lite Python object detection example with Pi Camera](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi).

An example to use [TensorFlow Lite](https://tensorflow.org/lite)
on a Raspberry Pi to perform real-time object detection using images
streamed from the Pi Camera. It draws a bounding box around each detected
object in the camera preview (when the object score is above a given threshold).

## Requirements

- Raspberry Pi (tested with Raspberry Pi 4 model 4GB)
- UbuntuOS (tested with Ubuntu Desktop 21.04)
- Raspberry Pi Camera (test with V2 camera module)

May work with other versions.

## Setup your hardware

Install UbuntuOS on your Raspberry Pi.

You also need to [connect and configure the Pi Camera](
https://www.raspberrypi.org/documentation/configuration/camera.md)..
Since the image stream from the camera is processed by OpenCV `VideoCapture()`,
USB cameras can also be used.

## Install TensorFlow Lite runtime

In this project, all you need from the TensorFlow Lite API is the `Interpreter`
class. So instead of installing the large `tensorflow` package, we're using the
much smaller `tflite_runtime` package.

To install `tflite_runtime` package:

```
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

If you use Rasberry Pi OS (OS other than Ubuntu), follow
[Python quickstart](https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python).

You also have to install additional Python packages:

```
pip3 install -r requirements.txt
```

## Download the example files

Use [TensorFlow Lite Python object detection example with Pi Camera script](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/download.sh)
to download the MobileNet model and labels file:

```
# The script takes an argument specifying where you want to save the model files
bash download.sh ./models
```

## Run the example

```
python3 detect.py \
  --model ./models/detect.tflite \
  --labels ./models/coco_labels.txt \
  --threshold 0.5
```

You should see the camera feed appear on the monitor attached to your Raspberry
Pi. Put some objects in front of the camera, like a coffee mug or keyboard, and
you'll see boxes drawn around those that the model recognizes, including the
label and score for each. It also prints the amount of time it took
to perform each inference in milliseconds at the top-left corner of the screen.

For more information about executing inferences with TensorFlow Lite, read
[TensorFlow Lite inference](https://www.tensorflow.org/lite/guide/inference).
