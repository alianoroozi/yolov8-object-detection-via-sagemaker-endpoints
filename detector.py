import json
import numpy as np
from sagemaker.pytorch.model import PyTorchPredictor


class Detector:
    def __init__(self, endpoint_name, labels_path):
        self.predictor = PyTorchPredictor(endpoint_name=endpoint_name)
        self.labels = self.read_labels(labels_path)
        
    def read_labels(self, labels_path):
        lines = open(labels_path).read().strip().split("\n")
        labels = [l.split(': ')[-1] for l in lines]
        return labels
        
    def detect(self, img_bucket, img_key, detection_classes):
        """
        Args:
            detections: a numpy array
            detection_classes: a list
        """
        data = {'img_bucket': img_bucket, 'img_key': img_key}
        request_body = json.dumps(data)
        detections = self.predictor.predict(request_body)
        results = self.process_torch_prediction(detections=detections, 
                                              detection_classes=detection_classes)
        return results
    
    def process_torch_prediction(self, detections, detection_classes):
        """
        Process torch model output and return coordinates of detected objects and prediction scores.
        Args:
            detections: a numpy array
            detection_classes: a list
        Returns:
            boxes: an array of coordinates of detected objects
            obj_scores: an array of prediction scores
        """
        if detections.shape[0] == 0:
            raise Exception("No object detected!")
        
        # Indexes of predicted classes
        detected_labels_ind = np.array([det[-1] for det in detections])
        # prediction scores
        scores = np.array([det[-2] for det in detections])
        
        # Indexs of desired classes
        desired_labels_ind = [self.labels.index(c) for c in detection_classes]

        # Filter detected objects to contain only desired objects
        filter_ind = np.isin(detected_labels_ind, desired_labels_ind)
        obj_detections = detections[filter_ind]
        obj_scores = scores[filter_ind]

        # get coordinates
        boxes = obj_detections[:, :4].round().astype(int)
        if len(boxes) == 0:
            raise Exception("No desired object detected!")

        return boxes, obj_scores
