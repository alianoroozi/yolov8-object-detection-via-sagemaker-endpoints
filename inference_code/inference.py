import os
from PIL import Image
import numpy as np
import boto3
import json
from six import BytesIO
from ultralytics import YOLO


s3 = boto3.client('s3')


def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'yolov8x.pt')
    model = YOLO(model_path)
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/x-npy'
    data_npy = np.load(BytesIO(request_body))
    data = json.loads(data_npy.item())

    img_local_path = '/tmp/img.png'
    s3.download_file(data['img_bucket'], data['img_key'], img_local_path)
    img = Image.open(img_local_path).convert('RGB')
    return img


def predict_fn(input_object, model):
    results = model(input_object, stream=True)
    # results is a generator, so we need to convert it to a list
    # there is just one image in the list, so we need to get the first element
    # use .cpu() to get the tensor from the GPU
    detections = np.array(list(results)[0].cpu().boxes.boxes)
    return detections
