import cv2
import PIL
import numpy as np

from ultralytics import YOLO
from collections import Counter
from .convert import parse_xywh_and_class
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

class BrailleSegmentation:

    def __init__(self, yolo_weight="weights/yolov8_braille.pt"):
        self.conf = 0.15
        self.image_dim = (100, 150)
        self.yolo_weight = yolo_weight

    def segment_braille(self, image_path):
        """Segment Braille Letters from Image"""

        image = cv2.imread(image_path)
        yolo_model = YOLO(self.yolo_weight)
        results = yolo_model.predict(image, conf=self.conf, max_det=9999)
        
        boxes = results[0].boxes
        list_boxes = parse_xywh_and_class(boxes)
        
        return results, list_boxes
    
    def get_distance(self, xs):
        """Get distance between each x center"""

        distances = np.diff(xs)
        common_distance = Counter(distances).most_common(1)[0][0]
        
        return distances, common_distance

    def get_box_properties(self, row):
        """
        Get x coordinates, classes, horizontal distance between each boxes,
        and the common distance from each boxes.
        """

        xs = [box[0] for box in row.astype(int)]
        classes = [box[-1] for box in row.astype(int)]
        distances, common = self.get_distance(xs)

        return xs, classes, distances, common
    
    def clean_bboxes(self, boxes):
        """Remove overlapping bounding boxes"""
        
        for i, row in enumerate(boxes):
            xs, _, distances, common = self.get_box_properties(row)
            for j, _ in enumerate(xs):
                if j > 0:
                    if distances[j-1] < common / 2:
                        row = np.delete(row, j-1, 0)
            boxes[i] = row

        return boxes