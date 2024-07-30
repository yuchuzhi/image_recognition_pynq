#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from pynq_dpu import DpuOverlay
from PyCameraList.camera_device import list_video_devices


print(" ################################################################## ")
print("image recognision in TensorFlow2, use model of kr260_cifar10_tf2_resnet18.xmodel")
print(" ################################################################## ")


# input files
cnn_xmodel  = os.path.join("./", "kr260_cifar10_tf2_resnet18.xmodel")
images_dir  = os.path.join("./", "test_images")


# ***********************************************************************
# CIFAR10 Labels
# ***********************************************************************
labelNames = { "airplane" : 0, "automobile" : 1, "bird" : 2, "cat" : 3, "deer" : 4, "dog" : 5,
                "frog" : 6, "horse" : 7, "ship" : 8, "truck" : 9}

overlay = DpuOverlay("dpu.bit")
overlay.load_model(cnn_xmodel)
dpu = overlay.runner


def predict_label(softmax):
    keynames = list(labelNames.keys())
    return keynames[np.argmax(softmax)]

# see https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def calculate_softmax(data):
    result = np.exp(data)
    return result

def Normalize(image):
    x_test  = np.asarray(image)
    x_test = x_test.astype(np.float32)
    x_test = x_test/255.0
    x_test = x_test -0.5
    out_x_test = x_test *2
    return out_x_test

def preprocess_fn(image_filename):
    image=cv2.imread(image_filename)
    image = np.asarray(image)
    image2 = Normalize(image)
    return image2

def get_images(camera_des="USB Camera"):
    cameras = list_video_devices()
    my_camera = None
    for camera in cameras:
        if "USB Camera" in camera:
            my_camera = camera

    if my_camera == None:
        return

    timeout = 30
    cap = cv2.VideoCapture(my_camera[0]) # check this
    if cap.isOpened():
        while(timeout):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # write the resulting frame to file.
            cv2.imwrite(f"./prediction_images/image_{timeout}.png", frame)

            time.sleep(1)
            timeout -= 1
    else:
        print("camera not open!!!")

    # When everything done, release the capture
    cap.release()

def self_test(image_path):
    """DPU to Make Predictions on specify image

    Args:
        image_path (str): image path
        display (bool, optional): Defaults to False.
    """
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    shapeIn = tuple(inputTensors[0].dims)
    shapeOut = tuple(outputTensors[0].dims)
    outputSize = int(outputTensors[0].get_data_size() / shapeIn[0])
    print("shapeIn   : {}".format(shapeIn))
    print("shapeOut  : {}".format(shapeOut))
    print("outputSize: {}".format(outputSize))

    softmax = np.empty(outputSize)
    output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]
    input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
    image = input_data[0]

    preprocessed = preprocess_fn(image_path)
    image[0,...] = preprocessed.reshape(shapeIn[1:])
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    temp = [j.reshape(1, outputSize) for j in output_data]
    softmax = calculate_softmax(temp[0][0])
    print("\ninput image ", os.path.basename(image_path))
    print(f"Classification result: {predict_label(softmax)}, prediction: {softmax.argmax()}.")


def test(images_dir):
    """DPU to Make Predictions on specify images

    Args:
        image_dir (str): images dir
    """
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    shapeIn = tuple(inputTensors[0].dims)
    shapeOut = tuple(outputTensors[0].dims)
    outputSize = int(outputTensors[0].get_data_size() / shapeIn[0])
    print("shapeIn   : {}".format(shapeIn))
    print("shapeOut  : {}".format(shapeOut))
    print("outputSize: {}".format(outputSize))

    original_images = [f"{images_dir}/{i}" for i in os.listdir(images_dir) if i.endswith("png")]
    results = []
    predictions = np.empty(len(original_images))
    softmax = np.empty(outputSize)
    output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]
    input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
    image = input_data[0]

    time1 = time.time()
    for i, image_path in enumerate(original_images):
        preprocessed = preprocess_fn(cv2.imread(image_path))
        image[0,...] = preprocessed.reshape(shapeIn[1:])
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)
        temp = [j.reshape(1, outputSize) for j in output_data]
        softmax = calculate_softmax(temp[0][0])
        results.append(predict_label(softmax))
        predictions[i] = softmax.argmax()
        print("\ninput image ", os.path.basename(image_path))
        print("Classification result: ", predict_label(softmax), " ", softmax.argmax())

    time2 = time.time()
    execution_time = time2-time1
    print("  Execution time: {:.4f}s".format(execution_time))
    
    label_name = "cat"
    counter = 0
    for label in labelNames:
        tmp = results.count(label)
        if tmp > counter:
            label_name = label
            counter = tmp

    predict_percent = results.count(label_name)/len(original_images)
    print(f"The input image {predict_percent} predicted to be {label_name}, ")


#self_test("./test_images/cat_7970.png")

get_images()
test("./prediction_images")

del overlay
del dpu
