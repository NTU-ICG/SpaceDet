import os
import yaml
import numpy as np
import cv2
import random
import torch
from PIL import Image
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics import YOLO
from torchvision import transforms
from torchvision.transforms import Resize
from FeatureExtraction import Hog_descriptor
from FeatureExtraction import SIFT_descriptor

class DetectionResults:
    def __init__(self, detections):
        self.detections = detections
        self.conf = np.array([det[5] for det in detections])
        self.xyxy = np.array([self._to_xyxy(det[1:5]) for det in detections])
        self.cls = np.array([det[0] for det in detections])
        self.feature_maps = []

    @staticmethod
    def _to_xyxy(bbox):
        x_center, y_center, width, height = bbox
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]

class Args:
    def __init__(self, config):
        self.tracker_type = config['tracker_type']
        self.track_high_thresh = config['track_high_thresh']
        self.track_low_thresh = config['track_low_thresh']
        self.new_track_thresh = config['new_track_thresh']
        self.track_buffer = config['track_buffer']
        self.match_thresh = config['match_thresh']
        self.gmc_method = config['gmc_method']
        self.proximity_thresh = config['proximity_thresh']
        self.appearance_thresh = config['appearance_thresh']
        self.with_reid = config['with_reid']
        self.similarity_method = config['similarity_method']

class ObjectTracker:
    def __init__(self, model_path, image_folder, label_folder, config_path, img_size=4418):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.config_path = config_path
        self.img_size = img_size

        self.feature_model = YOLO(model_path)
        self.torch_model = self.feature_model.model.model

    def load_yaml_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def load_images_and_labels(self,merged_boxes,image_path):
        data = []
        boxes_np = np.array(merged_boxes)
        class_ids = boxes_np[:, 1].astype(float)
        confidence_scores = boxes_np[:, 6].astype(float)
        coordinates = boxes_np[:, 2:6].astype(float) * self.img_size
        processed_boxes = np.column_stack((class_ids, coordinates, confidence_scores))
        # Add processed boxes to data
        data.append((image_path, processed_boxes.tolist()))

        return data

    def track_objects(self, merged_boxes,image_path,frame_id,bot_tracker):
        """ Track objects using BoT-SORT algorithm. """
        data = self.load_images_and_labels(merged_boxes,image_path)
        tracks = []
        args = self.load_yaml_config()
        for frame_number, (img_path, detections) in enumerate(data):
            # Process detections and update tracker
            # Note: Detections format [class_id, x_center, y_center, width, height]
            # Tracker expects format [x1, y1, x2, y2, score, class_id]
            det_for_tracker = [det[:] for det in detections]  # Add a dummy score of 1.0
            results = DetectionResults(det_for_tracker)

            image_array_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if args['feature_method'] == 'yolo': image_array = image_array_rgb
            else: image_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            layer_outputs = {}
            def get_layer_output(module, input, output):
                layer_outputs['layer_20'] = output
            hook = self.feature_model.model.model[20].register_forward_hook(get_layer_output)

            if args['with_reid']:
                target_size = (640, 640)
                bboxes = results.xyxy
                for i, bbox in enumerate(bboxes):
                    bbox = [int(x) for x in bbox]
                    segment_array = image_array[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    resized_segment_array = cv2.resize(segment_array, target_size)

                    if args['feature_method'] == 'yolo':
                        # Feature extraction method based on YOLOv8
                        new_shape = np.expand_dims(resized_segment_array, axis=0)
                        new_shape = new_shape[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
                        new_shape = np.ascontiguousarray(new_shape)  # contiguous
                        new_shape = torch.from_numpy(new_shape)
                        new_shape = new_shape.float()
                        new_shape /= 255
                        self.feature_model(new_shape)
                        results.feature_maps.append(torch.mean(layer_outputs['layer_20'], dim=[2, 3]).cpu())
                    elif args['feature_method'] == 'hog':
                        # Feature extraction method based on HOG
                        hog_descriptor = Hog_descriptor(resized_segment_array)
                        hog_vector = hog_descriptor.extract()
                        results.feature_maps.append(hog_vector)
                    elif args['feature_method'] == 'sift':
                        # Feature extraction method based on SIFT
                        sift_descriptor = SIFT_descriptor(resized_segment_array)
                        sift_tensor = sift_descriptor.extract()
                        results.feature_maps.append(sift_tensor.cpu())
            tracked_items = bot_tracker.update(results, frame_id, image_array_rgb)

            for track in tracked_items:
                class_id, bbox = int(track[5]), track[:4]
                # Convert bbox from [x1, y1, x2, y2] to [x_center, y_center, width, height]
                x_center, y_center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
                width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tracks.append([img_path, track[4], class_id, x_center, y_center, width, height])  # Adding object ID
        return tracks

















