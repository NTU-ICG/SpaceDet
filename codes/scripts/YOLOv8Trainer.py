# Class Two: Used for YOLOv8 training, prediction and then output tracking result

import os                      # export PYTHONPATH="${PYTHONPATH}:/home/space/Code/Space-YOLO"
import time
import numpy as np
import sys
import yaml
from datetime import datetime, timedelta
import math
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)
os.chdir(root_dir)
ultralytics_path = os.path.join(root_dir, 'codes')
sys.path.insert(0, ultralytics_path)
try:
    from ultralytics import YOLO, settings
    print("Successfully imported YOLO and settings from ultralytics.")
except ModuleNotFoundError as e:
    print(f"Error importing ultralytics: {e}")

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)

class YOLOv8Trainer:
    def __init__(self,dataset_result,runs_dir,picture_base_name,data_folder,mode,distance_threshold,angle_threshold,Iou_threshold,conf_threshold,track_low_thresh,config_path,pretrained_model,model_path=None,epochs=100,imgsz=640,device=[0, 1],Focal_length=35e-6,width=4418,height=4418,width_mm=14.14,height_mm=14.14,cutting_size=260,overlap=0.2):
        self.model_path = model_path
        self.pretrained_model = pretrained_model
        self.epochs = epochs
        self.imgsz = imgsz
        self.device = device
        self.mode = mode
        self.config_path = config_path
        self.dataset_result = dataset_result
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.Iou_threshold = Iou_threshold
        self.conf_threshold = conf_threshold
        self.track_low_thresh = track_low_thresh
        self.picture_base_name = picture_base_name
        self.Focal_length = Focal_length
        self.width = width
        self.height = height
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.cutting_size = cutting_size
        self.overlap = overlap
        self.data_folder = data_folder
        self.segment_root_path = os.path.join(self.dataset_result, self.mode, f'segment_{cutting_size}_{overlap}')
        self.original_image_path = os.path.join(self.dataset_result, self.mode, 'original')
        self.dataset_path = self.segment_root_path + "/dataset.yaml"
        # self.tracking_data_file = self.data_folder + '/Tracking_Data_predict.csv'
        # self.actual_tracking_file = self.data_folder + '/Tracking_Data.csv'
        settings.update({'runs_dir': runs_dir, 'tensorboard': True})

        if self.mode == 'train':
            # Load a pretrained model (recommended for training)
            self.model = YOLO(self.pretrained_model,task='detect')
            # self.model = YOLO(model_path)
            self.valid_dataset = self.segment_root_path + '/test'
            self.test_dataset = self.segment_root_path + '/test/images'
            # self.results_dir = self.original_image_path + '/test/predict'
        elif self.mode == 'validate' or self.mode == 'predict':
            if self.model_path:
                # Load a specific model
                self.model = YOLO(self.model_path)
                self.valid_dataset = self.segment_root_path
                self.test_dataset = self.segment_root_path + '/images'
                # self.results_dir = self.original_image_path + '/predict'
            else:
                print('No model provided')
                sys.exit(1)
        else:
            print('The mode you provid is invalid,please choose from \'train\', \'validate\' or \'predict\'')
            sys.exit(1)

    def create_yaml_file(self, mode, val_paths):
        data_dict = {}
        segment_root_path_abs = os.path.abspath(self.segment_root_path)
        if mode == 'train':
            # train_paths = self.segment_root_path + '/train/images'
            # # val_paths = self.segment_root_path + '/valid/images'
            # test_paths = self.segment_root_path + '/test/images'
            train_paths = os.path.join(segment_root_path_abs, 'train/images')
            test_paths = os.path.join(segment_root_path_abs, 'test/images')
            val_paths = os.path.abspath(val_paths)
            data_dict = {
                'train': train_paths,
                'val': val_paths,
                'test': test_paths,
                'nc': 3,
                'names': ['Low', 'Medium', 'High']
            }
        elif mode == 'validate':
            # val_paths = self.segment_root_path + '/images'
            val_paths = os.path.abspath(val_paths)
            data_dict = {
                'train': segment_root_path_abs,
                'val': val_paths,
                'test': segment_root_path_abs,
                'nc': 3,
                'names': ['Low', 'Medium', 'High']
            }

        with open(self.dataset_path, 'w') as file:
            yaml.dump(data_dict, file)

    def predict_segment_dataset(self,image_path):
        start_time = time.time()
        # results = self.model.predict(source=self.test_dataset,conf=0.5,iou=0,imgsz=640,stream=True)
        one_image = [[image_path], self.cutting_size, self.overlap, self.width, self.height]
        results = self.model.predict(source=one_image,conf=self.track_low_thresh,iou=0.3,imgsz=640,stream=True)

        all_boxes = []
        for result in results:
            image_path = result.path
            for box in result.boxes:
                class_id = box.cls.item()
                x_center_rel, y_center_rel, width_rel, height_rel = box.xywhn.cpu().numpy().flatten()
                confidence_score = box.conf.item()
                all_boxes.append([image_path, class_id, x_center_rel, y_center_rel, width_rel, height_rel, confidence_score])
        end_time = time.time()
        print(f"Prediction time: {end_time - start_time} seconds")
        
        return all_boxes
    
    def extract_time_from_seg_filename(self, filename):
        # Assume the file name format is "Raw_Observation0000_xxx.tiff"
        # Remove the "Raw_Observation" prefix first, then extract the numeric part
        picture_base_name_without_slash = self.picture_base_name.lstrip('/')
        number_part_str = filename.split(picture_base_name_without_slash)[1].split('_')[0]
        number_part = int(number_part_str)
        base_time = datetime(2023, 1, 1)
        # Treat the number part as seconds and calculate the time
        timestamp = base_time + timedelta(seconds=number_part)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')

    def extract_time_from_ori_filename(self, filename):
        # Assume the file name format is "Raw_Observation0000.tiff"
        # Remove the "Raw_Observation" prefix first, then extract the numeric part
        picture_base_name_without_slash = self.picture_base_name.lstrip('/')
        number_part_str = filename.split(picture_base_name_without_slash)[1].split('_')[0]
        number_part = int(number_part_str[:4])
        base_time = datetime(2023, 1, 1)
        # Treat the number part as seconds and calculate the time
        timestamp = base_time + timedelta(seconds=number_part)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    #Used to realize the conversion from pixel coordinate system to bearingangle
    def pixel_angles(self, px, py):
        pixel_per_mm_x = self.width / self.width_mm
        pixel_per_mm_y = self.height / self.height_mm
        cx = self.width // 2
        cy = self.height // 2
        x_pixel = px - cx  
        y_pixel = cy - py
        x_mm = x_pixel / pixel_per_mm_x
        y_mm = y_pixel / pixel_per_mm_y
        P_picture = [x_mm*10**-6,y_mm*10**-6,1]

        rot_body_picture = np.array([[self.Focal_length/1, 0, 0],
                        [0, self.Focal_length/1, 0],
                        [0, 0, 1/1]])
        rot_picture_body = np.linalg.inv(rot_body_picture)
        P_body = rot_picture_body @ P_picture
        tan_A1 = P_body[0] / P_body[2]
        tan_A2 = P_body[1] / P_body[2]
        A1_rad = math.atan(tan_A1)
        A2_rad = math.atan(tan_A2)
        A1 = (A1_rad / math.pi) * 180
        A2 = (A2_rad / math.pi) * 180

        return A1, A2

    def calculate_distance(self, box1, box2):
        """Calculate a specific distance between two bounding box center points"""
        x_center1 = box1[2]; y_center1 = box1[3]
        x_center2 = box2[2]; y_center2 = box2[3]
        # Make sure box1 is on the left
        if x_center1 < x_center2:
            point_1 = [x_center2 - box2[4] / 2, y_center2]
            point_2 = [x_center1 + box1[4] / 2, y_center1]
        else:
            point_1 = [x_center1 - box1[4] / 2, y_center1]
            point_2 = [x_center2 + box2[4] / 2, y_center2]
        distance = np.linalg.norm(np.array(point_1) - np.array(point_2))

        return distance

    def merge_seg_target(self, boxes):
        """
        Merge overlapping bounding boxes based on center point distance.
        The format of boxes: [[time_str, class_id, x_center_rel, y_center_rel, width_rel, height_rel], ...]
        """
        merged_boxes = []

        while boxes:
            current_box = boxes.pop(0)
            overlapping_indices = []
            x_min = current_box[2] - current_box[4] / 2
            y_min = current_box[3] - current_box[5] / 2
            x_max = current_box[2] + current_box[4] / 2
            y_max = current_box[3] + current_box[5] / 2
            max_confidence_score = current_box[6]

            for i, box in enumerate(boxes):
                distance = self.calculate_distance(box, current_box)
                if distance < self.distance_threshold and distance >= 0:
                    overlapping_indices.append(i)
                    x_min = min(x_min, box[2] - box[4] / 2)
                    y_min = min(y_min, box[3] - box[5] / 2)
                    x_max = max(x_max, box[2] + box[4] / 2)
                    y_max = max(y_max, box[3] + box[5] / 2)
                    max_confidence_score = max(max_confidence_score, box[6])  # Retain the higher confidence score

            for i in sorted(overlapping_indices, reverse=True):
                boxes.pop(i)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            merged_boxes.append([current_box[0], current_box[1], x_center, y_center, width, height, max_confidence_score])

        return merged_boxes

    
    def extract_tile_position(self, image_path, full_tiles_x):
        # Implement the extraction of x_tile and y_tile from the file path
        # Assume the file name format is "Raw_Observation0010_3.txt" where 3 is the tile index
        base_name = os.path.basename(image_path)
        tile_index = int(base_name.split('_')[-1].split('.')[0])
        y_tile = tile_index // full_tiles_x
        x_tile = tile_index % full_tiles_x
        return x_tile, y_tile

    def merge_boxes(self, all_boxes):
        full_tiles_x = math.ceil((self.width - self.cutting_size) / (self.cutting_size * (1 - self.overlap))) + 1
        overlap_size = int(self.cutting_size * self.overlap)

        boxes_by_image = {}
        for image_path, class_id, x_center_rel, y_center_rel, width_rel, height_rel, confidence_score in all_boxes:
            x_tile, y_tile = self.extract_tile_position(image_path, full_tiles_x)
            # Extract time information
            time_str = self.extract_time_from_seg_filename(os.path.basename(image_path))
            # Convert to coordinates under original image size
            tile_x_start = x_tile * (self.cutting_size - overlap_size)
            tile_y_start = y_tile * (self.cutting_size - overlap_size)
            x_center_abs = (x_center_rel * self.cutting_size) + tile_x_start
            y_center_abs = (y_center_rel * self.cutting_size) + tile_y_start
            width_abs = width_rel * self.cutting_size
            height_abs = height_rel * self.cutting_size

            # Transform regression normalized coordinates
            x_center_rel = x_center_abs / self.width
            y_center_rel = y_center_abs / self.width
            width_rel = width_abs / self.width
            height_rel = height_abs / self.width

            # Extract the name of the original image
            base_name = os.path.basename(image_path)
            original_image_name = '_'.join(base_name.split('_')[:-1])
            boxes_by_image.setdefault(original_image_name, []).append([time_str, class_id, x_center_rel, y_center_rel, width_rel, height_rel,confidence_score])

        return boxes_by_image

    def predict(self,image_path,results_dir):
        # Calculate the size of the overlapping area
        all_boxes = self.predict_segment_dataset(image_path)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        boxes_by_image = self.merge_boxes(all_boxes)
        # Merge the bounding boxes of each image and write them to the corresponding file
        for image_name, boxes in boxes_by_image.items():
            merged_boxes = self.merge_seg_target(boxes)
            output_file = os.path.join(results_dir, f"{image_name}.txt")
            with open(output_file, 'w') as file:
                for box in merged_boxes:
                    # time_str = box[0]
                    class_id = int(box[1])
                    x_center = box[2];y_center = box[3];width = box[4];height = box[5];confidence_score = box[6]
                    if confidence_score >= self.conf_threshold:
                        file.write(f"{class_id} {x_center} {y_center} {width} {height} {confidence_score}\n")
        if len(boxes_by_image) == 0:
            empty_output_file = os.path.join(results_dir, f"{os.path.basename(image_path).split('.')[0]}.txt")
            open(empty_output_file, 'w').close()  # 创建一个空的 txt 文件
            return 1

        return merged_boxes

    def tracking(self,tracked_objects):
        # realize tracking and save to csv file
        tracking_data = []
        for obj in tracked_objects:
            time_str = self.extract_time_from_ori_filename(os.path.basename(obj[0]))
            object_ID = int(obj[1]); class_ID = int(obj[2]); x_center = obj[3];y_center = obj[4];width = obj[5];height = obj[6]
            x_right_top = x_center + obj[5]/2; y_right_top = y_center - obj[6]/2
            angel1, angle2 = self.pixel_angles(x_right_top, y_right_top)
            tracking_data.append([time_str, object_ID, angel1, angle2])
            # x_center_norm = x_center/4418; y_center_norm = y_center/4418
            # width_norm = width/4418; height_norm = height/4418
            # tracking_data.append([time_str, object_ID, x_center_norm, y_center_norm, width_norm, height_norm])

        return tracking_data

    @staticmethod
    def annotate_image(image_path, tracking_data):
        """
        Annotate the image with bounding boxes and IDs.
        Parameters:
        - image_path: Path to the image file.
        - tracking_data: List of tracking data for each object in the format
                        [image_path, ID, _, x_center, y_center, width, height].
        - font_scale: Font scale for ID text.
        Returns:
        - Annotated image as a cv2 or tensor variable.
        """
        # Read the 16-bit grayscale image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Convert the 16-bit image to 8-bit
        image_8bit = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
        # Convert to RGB
        image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
        # Predefined colors for IDs, ensuring visibility on a black background
        font_scale = 3
        colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 0, 255)]
        for index, obj in enumerate(tracking_data):
            _, obj_id, _, x_center, y_center, width, height = obj
            # Calculate bounding box top-left and bottom-right corners
            start_point = (int(x_center - width / 2), int(y_center - height / 2))
            end_point = (int(x_center + width / 2), int(y_center + height / 2))
            # Draw the bounding box in red
            cv2.rectangle(image_rgb, start_point, end_point, (0, 0, 255), 4)
            # Choose color for the ID text based on object index
            color = colors[index % len(colors)]
            # Display the object ID above the bounding box
            cv2.putText(image_rgb, str(int(obj_id)), (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        return image_rgb


    # def process(self):
    #     if self.mode == 'train':
    #         label_folder = self.original_image_path + '/test/labels'
    #         val_paths = self.segment_root_path + '/valid/images'
    #         self.create_yaml_file(self.mode, val_paths)
    #         self.model.train(data=self.dataset_path, epochs=self.epochs, imgsz=self.imgsz, device=self.device)
    #         self.predict()
    #         evaluator = ObjectDetectionEvaluator(self.results_dir, label_folder, self.Iou_threshold)
    #         evaluator.evaluate_detection()
    
    #     if self.mode == 'validate':
    #         label_folder = self.original_image_path + '/labels'
    #         # val_paths = self.segment_root_path + '/test/images'
    #         # self.create_yaml_file(self.mode, val_paths)
    #         # metrics = self.model.val(data=self.dataset_path,imgsz=640,conf=0.8,iou=0)
    #         # print("metrics.box.map50:",metrics.box.map50)
    #         self.predict()
    #         evaluator = ObjectDetectionEvaluator(self.results_dir, label_folder,self.Iou_threshold)
    #         evaluator.evaluate_detection()
    #         tracker = ObjectTracker(self.picture_path, self.results_dir, self.config_path, self.width)
    #         tracked_objects = tracker.track_objects()
    #         self.tracking(tracked_objects)
    #         # Tracking evaluating
    #         evaluator = TrackingEvaluator(self.tracking_data_file, self.actual_tracking_file, self.angle_threshold)
    #         evaluator.process_tracking_data()
    #         evaluator.print_results()
    #         evaluator.print_detailed_results()
    
    #     if self.mode == 'predict':
    #         tracker = ObjectTracker(self.picture_path, self.results_dir, self.config_path, self.width)
    #         tracking_data_all = []
    #         for image_path in self.segments_info[0]:
    #             merged_boxes = self.predict(image_path)
    #             tracked_objects = tracker.track_objects(merged_boxes,image_path)
    #             tracking_data = self.tracking(tracked_objects)
    #             tracking_data_all.extend(tracking_data)
    #             print("tracking_data:", tracking_data)
    #         # Convert tracking_data to DataFrame
    #         print("tracking_data_all:", tracking_data_all)
    #         tracking_df = pd.DataFrame(tracking_data_all, columns=['Timestamp', 'Object', 'Angle1', 'Angle2'])
    #         # Save DataFrame to CSV file
    #         tracking_df.to_csv(self.tracking_data_file, index=False)


# if __name__ == "__main__":
#     model_path = "/home/rangya/Pycharm/Pycharm_Projects/YOLO_V8/datasets/best.pt"
#     dataset_result = "/home/rangya/Pycharm/Pycharm_Projects/YOLO_V8/datasets/YOLO_form/Cam2_yolo"
#     data_folder = '/home/rangya/Pycharm/Pycharm_Projects/YOLO_V8/datasets/original/New_Version/Cam_2-Az_90'   #Used to store csv files
#     picture_base_name = '/Raw_Observation'  # '/Raw_Observation','/binary_image_', '/binary_'
#     config_path = "/home/rangya/Pycharm/Pycharm_Projects/YOLO_V8/datasets/YOLO_form/Cam2_yolo/botsort.yaml"
#     Focal_length = 35e-6; width = 4418; height = 4418; width_mm = 14.14; height_mm = 14.14
#     cutting_size = 260; overlap = 0
#     mode = 'predict'  # mode = 'train', 'validate' or 'predict'
#     distance_threshold = 60/4418
#     angle_threshold = 0.2
#     Iou_threshold = 0.5
#     epochs = 100
#     imgsz = 640
#     device = [0, 1]
#
#     trainer = YOLOv8Trainer(dataset_result, picture_base_name, data_folder, mode, distance_threshold, angle_threshold, Iou_threshold, config_path, model_path, epochs, imgsz, device,Focal_length,width,height,width_mm,height_mm,cutting_size,overlap)
#     trainer.process()



