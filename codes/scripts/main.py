from YOLODatasetGenerator import YOLODatasetGenerator
from YOLOv8Trainer import YOLOv8Trainer, ConfigLoader
from TrackingDataTransformer import TrackingDataTransformer
import time
import os                               # export PYTHONPATH="${PYTHONPATH}:/home/space/Code/Space-YOLO"
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '../..')
root_dir = os.path.abspath(root_dir)
sys.path.append(root_dir)
os.chdir(root_dir)
ultralytics_path = os.path.join(root_dir, 'codes')
sys.path.insert(0, ultralytics_path)

import numpy as np
import pandas as pd
from ObjectTracker import ObjectTracker, Args
from ObjectDetectionEvaluator import ObjectDetectionEvaluator
from TrackingEvaluator import TrackingEvaluator
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics import settings

if __name__ == "__main__":
    # Path for all the configuration parameters (The path need to be changed according to your environment)
    config = ConfigLoader('./codes/ultralytics/cfg/default.yaml')
    # Load all default parameters
    focal_length = config.get('focal_length');num_classes = config.get('num_classes')
    train_part = config.get('train_part'); valid_part = config.get('valid_part'); test_part = config.get('test_part')  
    width_default = config.get('width_default'); height_default = config.get('height_default')
    width_mm_default = config.get('width_mm_default'); height_mm_default = config.get('height_mm_default')  
    cutting_size = config.get('cutting_size'); overlap = config.get('overlap');desired_ratio = config.get('desired_ratio') 
    distance_threshold = config.get('distance_threshold');angle_threshold = config.get('angle_threshold')
    Iou_threshold = config.get('Iou_threshold');conf_threshold=config.get('conf_threshold');track_low_thresh=config.get('track_low_thresh')
    epochs = config.get('epochs');imgsz = config.get('imgsz');device = config.get('device');batch = config.get('batch');workers = config.get('workers')
    elevation_deg = config.get('elevation_deg'); azimuth_deg = config.get('azimuth_deg');optimizer = config.get('optimizer') 
    lr0 = config.get('lr0'); lrf = config.get('lrf');warmup_epochs = config.get('warmup_epochs')
    image_type = config.get('image_type');picture_base_name = config.get('picture_base_name')
    # camera_names = config.get('camera_names')
    mode = config.get('mode');result_name = config.get('result_name');resume = config.get('resume');dataset = config.get('dataset')
    # path parameters
    pretrained_model = config.get('pretrained_model')
    # data_folder = config.get('data_folder');picture_path_all = config.get('picture_path_all');dataset_result = config.get('dataset_result')
    model_path = config.get('model_path');config_path = config.get('config_path');runs_dir = config.get('runs_dir')
    
    # # path need to be changed according to your env (For A6000)
    # runs_dir = '/mnt/hdd/space_dataset/fourcamera/ya/yolo_results/runs/NeurIPS'
    # settings.update({'runs_dir': runs_dir, 'tensorboard': True})
    # data_folder = '/mnt/hdd/space_dataset/fourcamera'  # Used to store csv files
    # picture_path_all = '/mnt/hdd/space_dataset/fourcamera'    # 100images
    # dataset_result = "/mnt/hdd/space_dataset/fourcamera/ya/dataset_results/Cam2_5000_test"  #Cam2_100_NeurIPS_test Cam2_100_NeurIPS
    # evaluating_result = dataset_result + "/" + mode + "/evaluating_result_v8m_time111111.txt"
    # training_time_result = dataset_result + "/" + mode + "/training_time_result_v8m_time1111111.txt"
    # # model_path = "/home/space/space/yolo/codes/best_cam2.pt"   # now is based on cam2 5000 images
    # model_path = "/mnt/hdd/space_dataset/fourcamera/ya/yolo_results/runs/NeurIPS/detect/Cam2_100_v8n/weights/best.pt"  # now is based on cam2 5000 images
    # config_path = "/home/space/space/yolo/codes/botsort.yaml"
    
    # # path need to be changed according to your env (For 4090)
    # runs_dir = '/media/space/T9/ssa/dataset/fourcamera/ya/yolo_results/runs'
    # settings.update({'runs_dir': runs_dir, 'tensorboard': True})
    # data_folder = '/media/space/T9/ssa/dataset/fourcamera'  # Used to store csv files
    # picture_path_all = '/media/space/T9/ssa/dataset/fourcamera/ya'
    # dataset_result = "/media/space/T9/ssa/dataset/fourcamera/ya/Cam2_GEM_trying"  # Cam2_GEM_trying
    # evaluating_result = dataset_result + "/" + mode + "/evaluating_result.txt"
    # model_path = "/home/space/Code/Space-YOLO/best_cam2.pt"  # now is based on cam2 5000 images
    # config_path = "/home/space/Code/Space-YOLO/codes/botsort.yaml"


    data_folder = os.path.join('./data', dataset)
    picture_path_all = data_folder

    dataset_result = os.path.join('./results', result_name)
    evaluating_result = os.path.join(dataset_result, mode, config.get('evaluating_result'))
    training_time_result = os.path.join(dataset_result, mode, config.get('training_time_result'))
    if dataset=='SpaceNet-100' or dataset=='SpaceNet-5000':
        camera_names = ['images']
    elif dataset=='SpaceNet-full':
        data_folder = os.path.join(data_folder, 'images')
        picture_path_all = data_folder
        camera_names = ['camera1', 'camera2', 'camera3', 'camera4']
    else: raise ValueError("Invalid dataset selected. Please choose from 'SpaceNet-100', 'SpaceNet-5000', or 'SpaceNet-full'.")

    settings.update({'runs_dir': runs_dir, 'tensorboard': True}) # Ture on tensorboard for training
    start_time = time.time() # Recording the time of training
    generator = YOLODatasetGenerator(camera_names, data_folder, dataset_result, picture_path_all, mode, image_type, overlap, cutting_size,
                                     desired_ratio, picture_base_name,focal_length=35, train_part=0.7, valid_part=0.2,
                                     test_part=0.1, width_default=4418, height_default=4418, width_mm_default=14.14,
                                     height_mm_default=14.14, H_number=14.1376)
    segments_info = generator.process()
    print("Dataset generate successfully!")
    end_time_1 = time.time() # Recording the time of training
    print(f"Prediction time: {end_time_1 - start_time} seconds")

    trainer = YOLOv8Trainer(dataset_result, runs_dir, picture_base_name, data_folder, mode, distance_threshold, angle_threshold, Iou_threshold,conf_threshold,track_low_thresh,
                            config_path, pretrained_model, model_path, epochs, imgsz, device, focal_length*10**-6, width_default, height_default, width_mm_default,
                            height_mm_default, cutting_size, overlap)
    
    total_confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int) # Used the contain the confusion_matrix for whole datasets
    total_results = {'total_true_labels': 0,'total_predictions': 0,'true_positives': 0,'false_positives': 0,'false_negatives': 0,}

    # Execute different modes
    if trainer.mode == 'train':
        # val_paths = trainer.segment_root_path + '/valid/images'
        # trainer.create_yaml_file(trainer.mode, val_paths)
        # metrics = trainer.model.val(data=trainer.dataset_path,imgsz=640,conf=0.001,iou=0.5)
        val_paths = trainer.segment_root_path + '/valid/images'
        trainer.create_yaml_file(trainer.mode, val_paths)
        all_detection_results = [[] for _ in range(len(camera_names))]
        start_time_2 = time.time()
        trainer.model.train(data=trainer.dataset_path, epochs=trainer.epochs, batch=batch,imgsz=trainer.imgsz, device=trainer.device,workers=workers,name=result_name,optimizer=optimizer,resume=resume,lr0=lr0,lrf=lrf,warmup_epochs=warmup_epochs)
        end_time_2 = time.time()
        training_time = end_time_2 - start_time_2
        for i, segments_info_item in enumerate(segments_info):
            camera_name = camera_names[i]
            results_dir = os.path.join(trainer.original_image_path, 'test_'+camera_name, 'predict')
            label_folder = os.path.join(trainer.original_image_path, 'test_'+camera_name, 'labels')
            for image_path in segments_info_item[0]:
                merged_boxes = trainer.predict(image_path,results_dir)
        
            evaluator = ObjectDetectionEvaluator(results_dir, camera_name, label_folder, evaluating_result, num_classes, trainer.Iou_threshold)
            all_detection_results[i] = evaluator.evaluate_detection()
            confusion_matrix, precision, recall, f1_score = evaluator.evaluate_detection_with_class()
            total_confusion_matrix += confusion_matrix
        overall_result = evaluator.evaluate_multiple_datasets(all_detection_results,total_confusion_matrix)
        all_detection_results.append(overall_result)
        
        with open(training_time_result, 'w') as file:
            file.write(f"Training time for this model: {training_time} seconds")
        print(f"Training time for this model: {training_time} seconds")
        
    if trainer.mode == 'validate':
        results_dirs = [os.path.join(trainer.original_image_path, f'predict_{camera_name}') for camera_name in camera_names]
        picture_path = [os.path.join(picture_path_all, camera_name) for camera_name in camera_names]
        tracking_data_file = [os.path.join(data_folder, camera_name, 'Tracking_Data_predict.csv') for camera_name in camera_names]
        actual_tracking_file = [os.path.join(data_folder, camera_name, 'Tracking_Data.csv') for camera_name in camera_names]
        trackers = [ObjectTracker(model_path, picture_path[i], results_dirs[i], trainer.config_path, trainer.width) for i in range(len(camera_names))]
        label_folder = [os.path.join(trainer.original_image_path, f'labels_{camera_name}') for camera_name in camera_names]
        segments_dict = [{} for _ in range(len(camera_names))]
        tracking_data_all = [[] for _ in range(len(camera_names))]
        all_detection_results = [[] for _ in range(len(camera_names))]
        all_tracking_results = [[] for _ in range(len(camera_names))]
        # images_rgb_list = [[] for _ in range(len(camera_names))] #Used to store detection and tracking result in png format
        
        for i, segments_info_item in enumerate(segments_info):
            for image_path in segments_info_item[0]:
                time_info = image_path.split('/')[-1].split('Raw_Observation')[-1].split('.')[0]
                segments_dict[i][time_info] = image_path
        min_time = min(min(segment_dict) for segment_dict in segments_dict if segment_dict)
        max_time = max(max(segment_dict) for segment_dict in segments_dict if segment_dict)

        args = Args(trackers[0].load_yaml_config())
        if args.tracker_type == 'botsort':
            bot_tracker = [BOTSORT(args, frame_rate=1) for _ in range(len(camera_names))]
        else: bot_tracker = [BYTETracker(args, frame_rate=1) for _ in range(len(camera_names))]
        frame_id = 0

        det_times = []
        tracking_times = []
        for time_now in range(int(min_time), int(max_time) + 1):
            frame_id = frame_id + 1
            time_str = f"{time_now:04d}"
            tracked_objects = [[] for _ in range(len(camera_names))]
            tracking_data = [[] for _ in range(len(camera_names))]
            for i, segment_dict in enumerate(segments_dict):
                camera_name = camera_names[i]
                results_dir = os.path.join(trainer.original_image_path, 'predict_' + camera_name)
                image_path = segment_dict.get(time_str, None)
                # file_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]  #Used to store detection and tracking result in png format
                # label_path = os.path.join(results_dir, file_name_without_extension + ".txt")
                # image_results_dir = os.path.join(trainer.original_image_path, 'image_' + camera_name)
                # if not os.path.exists(image_results_dir):
                #     os.makedirs(image_results_dir)
                if image_path:
                    start_time_det = time.time()
                    merged_boxes = trainer.predict(image_path,results_dir)
                    end_time_det = time.time()
                    det_time = end_time_det - start_time_det
                    det_times.append(det_time)
                    if merged_boxes == 1:
                        continue
                    start_time_tracking = time.time()
                    tracked_objects[i] = trackers[i].track_objects(merged_boxes, image_path, frame_id, bot_tracker[i])
                    tracking_data[i] = trainer.tracking(tracked_objects[i])
                    tracking_data_all[i].extend(tracking_data[i])
                    end_time_tracking = time.time()
                    tracking_time = end_time_tracking - start_time_tracking
                    tracking_times.append(tracking_time)
                    # image_rgb = trainer.annotate_image(image_path, tracked_objects[i])      #Used to store detection and tracking result in png format
                    # new_image_name = os.path.join(image_results_dir,file_name_without_extension + ".png")
                    # # cv2.imwrite(new_image_name, image_rgb)
                    # images_rgb_list[i] = image_rgb
                    print(f"Tracking time: {end_time_tracking - start_time_tracking} seconds")
                print(f"Progress: {(time_now-int(min_time))*len(camera_names)+i+1}/{(int(max_time) + 1 - int(min_time))*len(camera_names)}")
        tracking_df_all = [pd.DataFrame(tracking_data, columns=['Timestamp', 'Object', 'Angle1', 'Angle2']) for tracking_data in tracking_data_all]
        avg_det_time = sum(det_times) / len(det_times)
        avg_tracking_time = sum(tracking_times) / len(tracking_times)

        print(f"Average Detection Time: {avg_det_time} seconds")
        print(f"Average Tracking Time: {avg_tracking_time} seconds")
        with open(training_time_result, 'w') as file:
            file.write(f"Average Detection Time: {avg_det_time} seconds\n")
            file.write(f"Average Tracking Time: {avg_tracking_time} seconds")
        

        # Save DataFrame to CSV file
        for i in range(len(camera_names)):
            tracking_df_all[i].to_csv(tracking_data_file[i], index=False)
            Detection_evaluator = ObjectDetectionEvaluator(results_dirs[i], camera_names[i], label_folder[i], evaluating_result,num_classes,trainer.Iou_threshold)
            all_detection_results[i] = Detection_evaluator.evaluate_detection()
            confusion_matrix, precision, recall, f1_score = Detection_evaluator.evaluate_detection_with_class()
            total_confusion_matrix += confusion_matrix

            Tracking_evaluator = TrackingEvaluator(tracking_data_file[i], actual_tracking_file[i], camera_names[i], evaluating_result, trainer.angle_threshold)
            Tracking_evaluator.process_tracking_data()
            tracking_results = Tracking_evaluator.print_results()
            # Tracking_evaluator.print_detailed_results()
            all_tracking_results[i] = tracking_results
        overall_result = Detection_evaluator.evaluate_multiple_datasets(all_detection_results,total_confusion_matrix)
        Tracking_evaluator.evaluate_multiple_tracking_datasets(all_tracking_results)
        all_detection_results.append(overall_result)
            
    if trainer.mode == 'predict':
        results_dirs = [os.path.join(trainer.original_image_path, f'predict_{camera_name}') for camera_name in camera_names]
        picture_path = [os.path.join(picture_path_all, camera_name) for camera_name in camera_names]
        tracking_data_file = [os.path.join(data_folder, camera_name, 'Tracking_Data_predict.csv') for camera_name in camera_names]
        trackers = [ObjectTracker(model_path, picture_path[i], results_dirs[i], trainer.config_path, trainer.width) for i in range(len(camera_names))]
        segments_dict = [{} for _ in range(len(camera_names))]
        tracking_data_all = [[] for _ in range(len(camera_names))]
        # images_rgb_list = [[] for _ in range(len(camera_names))]    #Used to store detection and tracking result in png format
        for i, segments_info_item in enumerate(segments_info):
            for image_path in segments_info_item[0]:
                time_info = image_path.split('/')[-1].split('Raw_Observation')[-1].split('.')[0]
                segments_dict[i][time_info] = image_path
        
        # min_time = min(min(segments_dict[0]), min(segments_dict[1]), min(segments_dict[2]), min(segments_dict[3]))
        # max_time = max(max(segments_dict[0]), max(segments_dict[1]), max(segments_dict[2]), max(segments_dict[3]))
        
        min_values = [];max_values = []
        # Iterate over the range of camera_names length
        for i in range(len(camera_names)):
            if segments_dict[i]:  # Ensure the list is not empty
                min_values.append(min(segments_dict[i]))
                max_values.append(max(segments_dict[i]))
        min_time = min(min_values);max_time = max(max_values)

        args = Args(trackers[0].load_yaml_config())
        if args.tracker_type == 'botsort':
            bot_tracker = [BOTSORT(args, frame_rate=1) for _ in range(len(camera_names))]
        else: bot_tracker = [BYTETracker(args, frame_rate=1) for _ in range(len(camera_names))]

        frame_id = 0
        for time_now in range(int(min_time), int(max_time) + 1):
            frame_id += 1
            time_str = f"{time_now:04d}"
            tracked_objects = [[] for _ in range(len(camera_names))]
            tracking_data = [[] for _ in range(len(camera_names))]
            for i, segment_dict in enumerate(segments_dict):
                camera_name = camera_names[i]
                results_dir = os.path.join(trainer.original_image_path, 'predict_' + camera_name)
                image_path = segment_dict.get(time_str, None)
                # file_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]    #Used to store detection and tracking result in png format
                # label_path = os.path.join(results_dir, file_name_without_extension + ".txt")
                # image_results_dir = os.path.join(trainer.original_image_path, 'image_' + camera_name)
                # if not os.path.exists(image_results_dir):
                #     os.makedirs(image_results_dir)
                if image_path:
                    merged_boxes = trainer.predict(image_path,results_dir)
                    if merged_boxes == 1:
                        continue
                    tracked_objects[i] = trackers[i].track_objects(merged_boxes, image_path, frame_id,bot_tracker[i])
                    tracking_data[i] = trainer.tracking(tracked_objects[i])
                    tracking_data_all[i].extend(tracking_data[i])
                    # image_rgb = trainer.annotate_image(image_path, tracked_objects[i])     #Used to store detection and tracking result in png format
                    # new_image_name = os.path.join(image_results_dir,file_name_without_extension + ".png")
                    # cv2.imwrite(new_image_name, image_rgb)
                    # images_rgb_list[i] = image_rgb
                print(f"Progress: {(time_now-int(min_time))*len(camera_names)+i+1}/{(int(max_time) + 1 - int(min_time))*len(camera_names)}")
        tracking_df_all = [pd.DataFrame(tracking_data, columns=['Timestamp', 'Object', 'Angle1', 'Angle2']) for tracking_data in tracking_data_all]
        
        # Save DataFrame to CSV file
        for i in range(len(camera_names)):
            tracking_df_all[i].to_csv(tracking_data_file[i], index=False)

    end_time = time.time()
    print(f"Prediction time: {end_time - start_time} seconds")
    # if mode == 'validate' or mode == 'predict':
    #     for i in range(len(camera_names)):
    #         transformer = TrackingDataTransformer(data_folder, camera_names[i], elevation_deg, azimuth_deg)
    #         transformer.executive_transform()
    print("Success!")




