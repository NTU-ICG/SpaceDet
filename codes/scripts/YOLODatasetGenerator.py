# Class One: used to generate segmented dataset(Used for train or predict)
import pandas as pd
from PIL import Image
import math
import random
import os
import glob
import shutil
import sys

class YOLODatasetGenerator:
    
    def __init__(self, camera_names, data_folder, dataset_result, picture_path_all, mode, image_type, overlap, cutting_size,desired_ratio, picture_base_name,focal_length=35, train_part=0.7, valid_part=0.2, test_part=0.1,width_default = 4418,height_default = 4418,width_mm_default = 14.14,height_mm_default = 14.14,H_number = 14.1376):
        self.data_folder = data_folder
        self.picture_path_all = picture_path_all
        self.camera_names = camera_names
        self.camera_data = [os.path.join(self.data_folder, camera_name) for camera_name in self.camera_names]
        self.camera_picture_path = [os.path.join(self.picture_path_all, camera_name) for camera_name in self.camera_names]
        self.dataset_result = dataset_result
        self.focal_length = focal_length
        self.train_part = train_part
        self.valid_part = valid_part
        self.test_part = test_part
        self.width_default = width_default
        self.height_default = height_default
        self.width_mm_default = width_mm_default
        self.height_mm_default = height_mm_default
        self.H_number = H_number
        self.mode = mode
        self.image_type = image_type
        self.overlap = overlap
        self.cutting_size = cutting_size
        self.desired_ratio = desired_ratio
        self.picture_base_name = picture_base_name
        self.segment_root_path = os.path.join(self.dataset_result, self.mode, f'segment_{cutting_size}_{overlap}/')
        self.original_image_path = os.path.join(self.dataset_result, self.mode, 'original')

    #transform from bearing angles to pixel coordinate(just for an item)
    def image_to_pixel_coordinates(self,A1, A2, range_value):
        A1_rad = math.radians(A1)
        A2_rad = math.radians(A2)
        tan_A1 = math.tan(A1_rad)
        tan_A2 = math.tan(A2_rad)
        x_pixel = (tan_A1 * self.focal_length/self.H_number + 0.5) * self.width_default
        y_pixel = (tan_A2 * self.focal_length/self.H_number - 0.5) * (-self.width_default)
        P_pixel = [x_pixel,y_pixel]
        return P_pixel
    
    # label is defined based on SMA
    def ang_trans_pixels(self, data_file):
        Tracking_Data_path = os.path.join(data_file, 'Tracking_Data.csv')
        Target_States_path = os.path.join(data_file, 'Target_States.csv')
        data = pd.read_csv(Tracking_Data_path)
        data_Target = pd.read_csv(Target_States_path)
        target_dict = data_Target.set_index(['Object', 'Timestamp']).to_dict()['SMA']
        results = []
        for index, row in data.iterrows():
            pixel_coords = self.image_to_pixel_coordinates(row['Angle1'], row['Angle2'], row['Range'])
            timestamp = row['Timestamp']
            obj_name = row['Object']
            Range = row['Range']
            SMA = target_dict.get((obj_name, timestamp), 0)
            results.append([timestamp, obj_name, pixel_coords[0], pixel_coords[1],Range,SMA])
        return results
    
    #generate YOLOv8 form of labels for original images
    def output_bb(self, results,camera_name):
        data_dict = {}
        for timestamp, obj_name, x, y, Range,SMA in results:
            if timestamp not in data_dict:
                data_dict[timestamp] = {}
            data_dict[timestamp][obj_name] = [x, y, Range,SMA]
        base_time = "2023-01-01 00:00:00"
        time_points = list(data_dict.keys())
        time_points.sort()

        label_all_figures = []
        for i in range(len(time_points) - 1):
            start_time_str = time_points[i]
            end_time_str = time_points[i+1]

            common_objects = set(data_dict[start_time_str].keys()) & set(data_dict[end_time_str].keys())
            label_one_figure = []
            for obj in common_objects:
                coord_start = data_dict[start_time_str][obj][:2]
                coord_end = data_dict[end_time_str][obj][:2]
                Range = data_dict[start_time_str][obj][2]
                SMA = data_dict[start_time_str][obj][3]
                # if Range <=200: object_label = 0
                # else: continue
#                 if Range <=1600: object_label = 0
#                 elif Range <=3000: object_label = 1
#                 else: object_label = 2
# #                     continue
                if SMA <=8413: object_label = 0
                elif SMA <=42240: object_label = 1
                else: object_label = 2
#                     continue
                # Label with center point and width and height format
                x_min = min(coord_start[0], coord_end[0]) - 2
                y_min = min(coord_start[1], coord_end[1]) - 2
                x_max = max(coord_start[0], coord_end[0])
                y_max = max(coord_start[1], coord_end[1])
                x_center = ((x_min + x_max) / 2) / self.width_default
                y_center = ((y_min + y_max) / 2) / self.height_default
                width = (x_max - x_min) / self.width_default
                height = (y_max - y_min) / self.height_default
                target_one_obj = [object_label, x_center, y_center, width, height]
                label_one_figure.append(target_one_obj)

            time_difference = pd.Timestamp(start_time_str) - pd.Timestamp(base_time)
            elapsed_seconds = int(time_difference.total_seconds())
            formatted_time = "{:04}".format(elapsed_seconds)
            # label_all_figures.append([formatted_time, label_one_figure])
            number_targets = len(common_objects)
            # if number_targets < 15:
            #     task_ID = 0
            # elif 15 <= number_targets < 30:
            #     task_ID = 1
            # elif 30 <= number_targets < 50:
            #     task_ID = 2
            # else:
            #     task_ID = 3
            if camera_name == 'camera1':
                task_ID = 0
            elif camera_name == 'camera2':
                task_ID = 1
            elif camera_name == 'camera3':
                task_ID = 2
            elif camera_name == 'camera4':
                task_ID = 3
            else:
                task_ID = 4
            label_all_figures.append([formatted_time, label_one_figure, task_ID])

        return label_all_figures

    def generate_labels_train(self, label_all_figures_all, picture_path_all, camera_names):
        def save_to_txt(file_name, data):
            with open(file_name, 'w') as file:
                for line in data[0]:
                    file.write(' '.join(map(str, line)) + ' ' + str(data[1]) + '\n')
                    # for item in line:
                    #     file.write(' '.join(map(str, item)) + '\n')

        def process_train_subset(files, label_dict, subset):
            for i, file in enumerate(files):
                filename = os.path.basename(file)
                image_index = str(filename[len(self.picture_base_name)-1:-len(self.image_type)+1])  # Extract time index
                image_format = self.image_type.strip('*')
                path_parts = file.split(os.sep)
                camera_part = path_parts[-2]
                label = label_dict.get(camera_part, {}).get(image_index, None)
                if subset == 'test':
                    label_file = os.path.join(self.original_image_path, subset + '_' + camera_part, 'labels', filename[:-len(self.image_type)+1] + '_' + camera_part + '.txt')
                    picture_save_path = os.path.join(self.original_image_path, subset + '_' + camera_part, 'images', filename[:-len(self.image_type)+1] + '_' + camera_part + image_format)
                else:
                    label_file = os.path.join(self.original_image_path, subset, 'labels', filename[:-len(self.image_type)+1] + '_' + camera_part + '.txt')
                    picture_save_path = os.path.join(self.original_image_path, subset, 'images', filename[:-len(self.image_type)+1] + '_' + camera_part + image_format)
                
                if label is not None:
                    save_to_txt(label_file, label)
                else:
                    open(label_file, 'w').close()
                shutil.copy(file, picture_save_path)
                if i == 0 or i % 100 == 0:
                    print("Generate ", subset, " dataset:", i+1, "/", len(files))
                
        # Handling based on data_mode
        if self.mode == 'train':
            epsilon = 1e-10
            if abs(self.train_part + self.valid_part + self.test_part - 1.0) > epsilon:
                print("Warning: The sum of the training, validation, and test dataset ratios is not 1 and will be set to the default value of 7:2:1")
                self.train_part = 0.7
                self.valid_part = 0.2
                self.test_part = 0.1
            picture_files_all = []
            combined_label_dict = {}
            for i, camera_name in enumerate(camera_names):
                # Create a dictionary with time index as key and label data as value
                label_dict = {entry[0]: entry[1:] for entry in label_all_figures_all[i]}
                combined_label_dict[camera_name] = label_dict
                # Get all .tiff pictures under self.picture_path
                picture_files = [f for f in os.listdir(picture_path_all[i]) if f.endswith(self.image_type[1:])]
                full_paths = [os.path.join(picture_path_all[i], f) for f in picture_files]
                picture_files_all.extend(full_paths)

            # Divide training, validation and test sets according to image file names
            random.shuffle(picture_files_all)
            total_pictures = len(picture_files_all)
            train_size = int(self.train_part * total_pictures)
            val_size = int(self.valid_part * total_pictures)
            train_files = picture_files_all[:train_size]
            valid_files = picture_files_all[train_size:train_size + val_size]
            test_files = picture_files_all[train_size + val_size:]

            train_path = os.path.join(str(self.original_image_path), 'train');valid_path = os.path.join(str(self.original_image_path), 'valid')
            test_paths = [os.path.join(str(self.original_image_path), 'test_'+camera_name) for camera_name in self.camera_names]
            train_images = train_path + '/images';valid_images = valid_path + '/images';test_images = [test_path + '/images' for test_path in test_paths]
            train_labels = train_path + '/labels';valid_labels = valid_path + '/labels';test_labels = [test_path + '/labels' for test_path in test_paths]
            directories = [train_path, valid_path, train_images, train_labels, valid_images, valid_labels]
            directories.extend(test_paths); directories.extend(test_images); directories.extend(test_labels)
            for directory in directories:
                if not os.path.exists(directory):
                    os.makedirs(directory)

            process_train_subset(train_files, combined_label_dict, 'train')
            process_train_subset(valid_files, combined_label_dict, 'valid')
            process_train_subset(test_files, combined_label_dict, 'test')

    def generate_labels_valid(self, label_all_figures, picture_path, camera_name):
        # def save_to_txt(file_name, data):
        #     with open(file_name, 'w') as file:
        #         for line in data:
        #             for item in line:
        #                 file.write(' '.join(map(str, item)) + '\n')
        def save_to_txt(file_name, data):
            with open(file_name, 'w') as file:
                for line in data[0]:
                    file.write(' '.join(map(str, line)) + ' ' + str(data[1]) + '\n')
        def process_valid_subset(files, label_dict, camera_name):
            for i, file in enumerate(files):
                image_index = str(file[len(self.picture_base_name) - 1:-len(self.image_type) + 1])  # Extract time index
                label_file = os.path.join(self.original_image_path, 'labels_' + camera_name,
                                          file[:-len(self.image_type) + 1] + '.txt')
                image_format = self.image_type.strip('*')
                picture_save_path = os.path.join(self.original_image_path, 'images',
                                                 file[:-len(self.image_type) + 1] + image_format)
                label = label_dict.get(image_index)
                if i == 0 or i % 100 == 0:
                    print("Generate ", camera_name, " dataset:", i + 1, "/", len(files))
                # if label is not None and label[0]:
                if label is not None:
                    save_to_txt(label_file, label_dict[image_index])
                    # shutil.copy(os.path.join(picture_path, file), picture_save_path)
                else:
                    open(label_file, 'w').close()

        # Use all data for validation
        valid_label = os.path.join(str(self.original_image_path), 'labels_' + camera_name)
        if not os.path.exists(valid_label):
            os.makedirs(valid_label)
        # Create a dictionary with time index as key and label data as value
        label_dict = {str(entry[0]).zfill(4): entry[1:] for entry in label_all_figures}
        # Get all .tiff pictures under self.picture_path
        picture_files = [f for f in os.listdir(picture_path) if f.endswith(self.image_type[1:])]
        process_valid_subset(picture_files, label_dict, camera_name)

    #Segment images
    def split_image_fixed(self,image_path):
        image = Image.open(image_path)
        width, height = image.size
        stride = self.cutting_size - int(self.cutting_size * self.overlap)
        full_tiles_x = math.ceil((width - self.cutting_size) / stride) + 1
        full_tiles_y = math.ceil((height - self.cutting_size) / stride) + 1
        tiles = []

        for y in range(full_tiles_y):
            for x in range(full_tiles_x):
                left = min(x * stride, width - self.cutting_size)
                upper = min(y * stride, height - self.cutting_size)
                right = left + self.cutting_size
                lower = upper + self.cutting_size
                tile = image.crop((left, upper, right, lower))
                position = (x, y)
                tiles.append((tile, position))

        return tiles, full_tiles_x, full_tiles_y
    
    #Segment labels
    def adjust_labels_yolov8(self, tiles, base_filename, full_tiles_x, label_file, save_dir_labels):
        with open(label_file, 'r') as file:
            labels = file.readlines()
            
        overlap_size = self.cutting_size * self.overlap
        
        for tile, (x_tile, y_tile) in tiles:
            adjusted_labels = []
            for label in labels:
                # class_id, x_left_upper, y_left_upper, width, height = map(float, label.split())
                class_id, x_left_upper, y_left_upper, width, height, task = map(float, label.split())
                # Convert coordinates to absolute coordinates
                x_left_upper_abs = x_left_upper * self.width_default
                y_left_upper_abs = y_left_upper * self.width_default
                width_abs = width * self.width_default
                height_abs = height * self.width_default
                # Calculate the starting absolute coordinates of the slice
                tile_x_start = x_tile * (self.cutting_size - overlap_size)
                tile_y_start = y_tile * (self.cutting_size - overlap_size)
                tile_x_end = tile_x_start + self.cutting_size
                tile_y_end = tile_y_start + self.cutting_size
                # Check if the label is at least partially inside the current slice
                if (x_left_upper_abs + width_abs > tile_x_start and x_left_upper_abs < tile_x_end) and \
                   (y_left_upper_abs + height_abs > tile_y_start and y_left_upper_abs < tile_y_end):
                    # Convert to coordinates relative to the current slice
                    x_left_upper_rel = max(x_left_upper_abs - tile_x_start, 0)
                    y_left_upper_rel = max(y_left_upper_abs - tile_y_start, 0)
                    width_abs = width_abs + x_left_upper_abs - tile_x_start - x_left_upper_rel
                    height_abs = height_abs + y_left_upper_abs - tile_y_start - y_left_upper_rel
                    # Make sure the bounding box does not extend beyond the slice
                    width_rel = min(width_abs, tile_x_end - x_left_upper_rel)
                    height_rel = min(height_abs, tile_y_end - y_left_upper_rel)
                    # Convert back to relative coordinates
                    x_left_upper_rel /= self.cutting_size
                    y_left_upper_rel /= self.cutting_size
                    width_rel /= self.cutting_size
                    height_rel /= self.cutting_size
                    # Format the adjusted label
                    adjusted_label = f"{class_id} {x_left_upper_rel} {y_left_upper_rel} {width_rel} {height_rel} {task}\n"
                    # adjusted_label = f"{class_id} {x_left_upper_rel} {y_left_upper_rel} {width_rel} {height_rel}\n"
                    adjusted_labels.append(adjusted_label)

            # Save adjusted labels
            tile_label_filename = f"{save_dir_labels}/{base_filename}_{y_tile * full_tiles_x + x_tile:03d}.txt"
            with open(tile_label_filename, 'w') as label_file:
                label_file.writelines(adjusted_labels)
                
    def save_tiles(self, tiles, base_filename, save_dir_images,full_tiles_x):
        for tile, (x_tile, y_tile) in tiles:
            tile_filename = f"{save_dir_images}/{base_filename}_{y_tile * full_tiles_x + x_tile:03d}{self.image_type[1:]}"
            tile.save(tile_filename)
    
    #Used to generate segment dataset
    def process_dataset(self, subset):
        images_path = os.path.join(str(self.original_image_path), str(subset), 'images')
        labels_path = os.path.join(str(self.original_image_path), str(subset), 'labels')
        save_dir_images = os.path.join(str(self.segment_root_path), str(subset), 'images')
        save_dir_labels = os.path.join(str(self.segment_root_path), str(subset), 'labels')
        os.makedirs(save_dir_images, exist_ok=True)
        os.makedirs(save_dir_labels, exist_ok=True)
        image_files = glob.glob(os.path.join(images_path, self.image_type))

        for i, image_file in enumerate(image_files):
            if i == 0 or i % 100 == 0:
                print("Segment ",subset," dataset: ",i + 1,"/",len(image_files))
            base_filename = os.path.splitext(os.path.basename(image_file))[0]
            label_file = os.path.join(labels_path, f"{base_filename}.txt")
            if not os.path.exists(label_file):
                continue
            tiles, full_tiles_x, full_tiles_y = self.split_image_fixed(image_file)
            self.save_tiles(tiles, base_filename, save_dir_images, full_tiles_x)
            self.adjust_labels_yolov8(tiles, base_filename,full_tiles_x,label_file, save_dir_labels=save_dir_labels)

    def process_images(self,picture_path):
        images_path = picture_path
        image_files = glob.glob(os.path.join(images_path, self.image_type))
        image_files = sorted(image_files)

        segment_info = []
        for image_file in image_files:
            base_filename = os.path.splitext(os.path.basename(image_file))[0]
            image_filename = f"{images_path}/{base_filename}{self.image_type[1:]}"
            segment_info.append(image_filename)
        segments_info = [segment_info, self.cutting_size, self.overlap, self.width_default,self.height_default]

        return segments_info

    #Calculate the ratio of segment images with traget and pure background        
    def calculate_label_ratio(self, labels_path):
        label_files = glob.glob(os.path.join(labels_path, '*.txt'))
        labeled_count = 0
        unlabeled_count = 0

        for label_file in label_files:
            with open(label_file, 'r') as file:
                labels = file.readlines()
            if labels:
                labeled_count += 1
            else:
                unlabeled_count += 1

        return labeled_count, unlabeled_count
    
    #Delete images with pure background based on a constant ratio
    def delete_unlabeled_images(self, subset):
        images_path = os.path.join(self.segment_root_path, subset, 'images')
        labels_path = os.path.join(self.segment_root_path, subset, 'labels')
        labeled_count, unlabeled_count = self.calculate_label_ratio(labels_path)
        print("labeled_count:",labeled_count,"unlabeled_count:",unlabeled_count)
        desired_unlabeled_count = math.ceil(labeled_count * self.desired_ratio)
        delete_count = max(0, unlabeled_count - desired_unlabeled_count)

        unlabeled_files = [f for f in glob.glob(os.path.join(labels_path, '*.txt')) if not open(f).read().strip()]
        i = 0
        for label_file in random.sample(unlabeled_files, int(delete_count)):
            if i == 0 or i % 10000 == 0:
                print("Delete unlabeled ",subset, " images", i + 1, "/",unlabeled_count - desired_unlabeled_count)
            i += 1
            base_filename = os.path.splitext(os.path.basename(label_file))[0]
            image_file = f"{base_filename}{self.image_type[1:]}"
            image_path = os.path.join(images_path, image_file)
            if os.path.exists(image_path):
                os.remove(image_path)  # Delete image
                os.remove(label_file)  # Delete label file

    #Conduct all the process of this class
    def process(self):
        segments_info = []
        if self.mode == 'train':
            #Step 1: generate YOLOv8 dataset
            label_all_figures_allcamera = []
            for i, camera_name in enumerate(self.camera_names):
                results = self.ang_trans_pixels(self.camera_data[i])
                label_all_figures = self.output_bb(results,camera_name)
                label_all_figures_allcamera.append(label_all_figures)
                # self.generate_labels(label_all_figures,self.camera_picture_path[i],camera_name)
            self.generate_labels_train(label_all_figures_allcamera,self.camera_picture_path,self.camera_names)
            for i, camera_name in enumerate(self.camera_names):
                test_picture_path = os.path.join(str(self.original_image_path), 'test_' + camera_name, 'images')
                segment_info = self.process_images(test_picture_path)
                segments_info.append(segment_info)
            #Step 2: generate YOLOv8 segment dataset
            for subset in ['train', 'valid']:
                self.process_dataset(subset)
                self.delete_unlabeled_images(subset)
            subset = 'test_camera2'
            self.process_dataset(subset)
            return segments_info

        elif self.mode == 'validate':
            # Step 1: generate YOLOv8 dataset
            for i, camera_name in enumerate(self.camera_names):
                results = self.ang_trans_pixels(self.camera_data[i])
                label_all_figures = self.output_bb(results,camera_name)
                self.generate_labels_valid(label_all_figures,self.camera_picture_path[i],camera_name)
                picture_path = os.path.join(self.picture_path_all, camera_name)
                segment_info = self.process_images(picture_path)
                segments_info.append(segment_info)
            return segments_info

        elif self.mode == 'predict':
            # generate segmented predict dataset
            for i, camera_name in enumerate(self.camera_names):
                picture_path = os.path.join(self.picture_path_all, camera_name)
                segment_info = self.process_images(picture_path)
                segments_info.append(segment_info)
            return segments_info
        else:
            print("Data mode is wrong,please input a valid data mode (select from ['train', 'validate', 'predict'])")
            sys.exit(1)


        # if __name__ == "__main__":
#     # default value
#     focal_length=35
#     train_part=0.7
#     valid_part=0.2
#     test_part=0.1
#     width_default = 4418
#     height_default = 4418
#     width_mm_default = 14.14
#     height_mm_default = 14.14
#     cutting_size = 260
#     overlap = 0
#     image_type = '*.tiff'
#     mode = 'validate'         # 'train','validate' or 'predict':train is used to generate training dataset(with train, valid and test),'predict' is used to generate predicted dataset(just segment)
#     picture_base_name = '/Raw_Observation'  # '/Raw_Observation','/binary_image_', '/binary_'
#     desired_ratio = 0.1 # Desired ratio of unlabeled to labeled images(just used for data_mode = train)
#
#     #path need to be changed according to your env
#     data_path = '/home/rangya/Pycharm/Pycharm_Projects/YOLO_V8/datasets/original/New_Version/Cam_2-Az_90/Tracking_Data.csv'
#     picture_path = '/home/rangya/Pycharm/Pycharm_Projects/YOLO_V8/datasets/original/Cam_2-Az_90'
#     dataset_result = "/home/rangya/Pycharm/Pycharm_Projects/YOLO_V8/datasets/YOLO_form/Cam2_yolo"
#     generator = YOLODatasetGenerator(data_path, dataset_result, picture_path, mode, image_type, overlap, cutting_size, desired_ratio, picture_base_name, focal_length=35, train_part=0.7, valid_part=0.2, test_part=0.1,width_default = 4418,height_default = 4418,width_mm_default = 14.14,height_mm_default = 14.14,H_number = 14.1376)
#     generator.process()
#     print("Success!")