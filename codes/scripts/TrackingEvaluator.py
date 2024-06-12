import pandas as pd


class TrackingEvaluator:
    def __init__(self, predicted_csv_path, actual_csv_path, camera_name, evaluating_result, angle_threshold=0.2):
        self.angle_threshold = angle_threshold
        self.predicted_df = pd.read_csv(predicted_csv_path)
        self.actual_df = pd.read_csv(actual_csv_path)
        self.camera_name = camera_name
        self.total_frames = self.actual_df['Timestamp'].nunique()
        self.matches = 0
        self.misses = 0
        self.false_positives = 0
        self.id_switches = 0
        self.accumulated_iou = 0
        self.total_target_in_all_frames = 0
        self.predict_target_in_all_frames = len(self.predicted_df)
        self.id_mappings = {}
        self.id_match_count = {}
        self.actual_id_count = {}
        self.evaluating_result = evaluating_result

    def calculate_angle_difference(self, angle1_pred, angle2_pred, angle1_actual, angle2_actual):
        diff1 = abs(angle1_pred - angle1_actual)
        diff2 = abs(angle2_pred - angle2_actual)
        return diff1, diff2

    def process_tracking_data(self):
        the_time = self.predicted_df['Timestamp'].unique()
        max_time = max(the_time); min_time = min(the_time)
        min_time = pd.to_datetime(min_time); max_time = pd.to_datetime(max_time)
        time_range = pd.date_range(start=min_time, end=max_time, freq='s')
        for timestamp in time_range:
            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            actual_data = self.actual_df[self.actual_df['Timestamp'] == timestamp]
            predicted_data = self.predicted_df[self.predicted_df['Timestamp'] == timestamp]
            self.total_target_in_all_frames += len(actual_data)
            current_matches = {}
            matched_predicted_ids = set()
            for _, actual_row in actual_data.iterrows():
                actual_id = actual_row['Object']
                self.actual_id_count[actual_id] = self.actual_id_count.get(actual_id, 0) + 1

                best_match = None
                best_diff = float('inf')

                for _, pred_row in predicted_data.iterrows():
                    pred_id = pred_row['Object']
                    if pred_id in matched_predicted_ids:
                        continue
                    diff1, diff2 = self.calculate_angle_difference(pred_row['Angle1'], pred_row['Angle2'],actual_row['Angle1'], actual_row['Angle2'])
                    if diff1 < self.angle_threshold and diff2 < self.angle_threshold:
                        total_diff = diff1 + diff2
                        if total_diff < best_diff:
                            best_diff = total_diff
                            best_match = pred_row['Object']

                if best_match is not None:
                    self.matches += 1
                    current_matches[actual_row['Object']] = best_match
                    matched_predicted_ids.add(best_match)
                    id_match_key = (actual_id, best_match)
                    self.id_match_count[id_match_key] = self.id_match_count.get(id_match_key, 0) + 1
                    if actual_row['Object'] in self.id_mappings and self.id_mappings[actual_row['Object']] != best_match:
                        self.id_switches += 1
                else:
                    self.misses += 1

            self.id_mappings.update(current_matches)
            self.false_positives += len(predicted_data) - len(current_matches)

    def calculate_mota(self):
        # real_obj_num = self.actual_df['Object'].nunique()
        # predict_obj_num = self.predicted_df['Object'].nunique()
        mota = 1 - (self.misses + self.false_positives + self.id_switches) / self.total_target_in_all_frames
        return mota

    def append_to_file(self, text):
        with open(self.evaluating_result, 'a') as file:
            file.write(text + '\n')

    def print_results(self):
        mota = self.calculate_mota()
        unique_predicted_timestamps = self.predicted_df['Timestamp'].unique()
        filtered_actual_df = self.actual_df[self.actual_df['Timestamp'].isin(unique_predicted_timestamps)]
        unique_actual_objects_in_predicted_range = filtered_actual_df['Object'].nunique()
        separator = "\n"
        title = "Tracking Results of " + self.camera_name + "\n"
        self.append_to_file(separator.strip())  # 移除首尾的换行符
        self.append_to_file(title.strip())
        results = []
        results.append(f'MOTA: {mota}')
        results.append(f"Total Targets in All Frames: {self.total_target_in_all_frames}")
        results.append(f"Predict Targets in All Frames: {self.predict_target_in_all_frames}")
        results.append(f'Total Matches: {self.matches}')
        results.append(f'Misses: {self.misses}')
        results.append(f'False Positives: {self.false_positives}')
        results.append(f'ID Switches: {self.id_switches}')
        results.append(f'Total Object Number(real): {unique_actual_objects_in_predicted_range}')
        results.append(f'Total Object Number(predict): {self.predicted_df["Object"].nunique()}')
        print(separator)
        print(title)
        for result in results:
            print(result)
            self.append_to_file(result)

        return {
            'MOTA': mota,
            'Total Targets in All Frames': self.total_target_in_all_frames,
            'Predict Targets in All Frames': self.predict_target_in_all_frames,
            'Total Matches': self.matches,
            'Misses': self.misses,
            'False Positives': self.false_positives,
            'ID Switches': self.id_switches,
            'Total Object Number(real)': unique_actual_objects_in_predicted_range,
            'Total Object Number(predict)': self.predicted_df["Object"].nunique()
        }

    def print_detailed_results(self):
        separator = "\n" + "\n"
        self.append_to_file(separator.strip()) 
        results = []
        results.append("Match count between actual and predicted IDs:")
        for key, count in self.id_match_count.items():
            results.append(f"Actual ID {key[0]} to Predicted ID {key[1]}: {count} times")
        results.append("\nTotal occurrences of each actual ID:")
        for key, count in self.actual_id_count.items():
            results.append(f"Actual ID {key}: {count} times")
        results.append("\nID Mappings:")
        for actual_id, predicted_id in self.id_mappings.items():
            results.append(f"Actual ID {actual_id} mapped to Predicted ID {predicted_id}")
        for result in results:
            # print(result)
            self.append_to_file(result)

    def evaluate_multiple_tracking_datasets(self, all_tracking_results):
        total_matches = sum(result['Total Matches'] for result in all_tracking_results)
        total_misses = sum(result['Misses'] for result in all_tracking_results)
        total_false_positives = sum(result['False Positives'] for result in all_tracking_results)
        total_id_switches = sum(result['ID Switches'] for result in all_tracking_results)
        total_targets = sum(result['Total Targets in All Frames'] for result in all_tracking_results)
        total_predictions = sum(result['Predict Targets in All Frames'] for result in all_tracking_results)
        total_object_real = sum(result['Total Object Number(real)'] for result in all_tracking_results)
        total_object_predict = sum(result['Total Object Number(predict)'] for result in all_tracking_results)
        mota = 1 - (total_misses + total_false_positives + total_id_switches) / total_targets
        
        # Store and print results
        results = []
        results.append(f"Overall MOTA: {mota}")
        results.append(f"Overall Total Targets in All Frames: {total_targets}")
        results.append(f"Overall Predict Targets in All Frames: {total_predictions}")
        results.append(f"Overall Total Matches: {total_matches}")
        results.append(f"Overall Misses: {total_misses}")
        results.append(f"Overall False Positives: {total_false_positives}")
        results.append(f"Overall ID Switches: {total_id_switches}")
        results.append(f'Overall Object Number(real): {total_object_real}')
        results.append(f'Overall Object Number(predict): {total_object_predict}')
        
        separator = "\n" 
        title = "Tracking Results of all cameras:\n"
        self.append_to_file(separator.strip())  
        self.append_to_file(title.strip())
        print(separator)
        print(title)
        for result in results:
            print(result)
            self.append_to_file(result)

# # Using examples
# if __name__ == "__main__":
#     angle_threshold = 0.2
#     predicted_csv_path = '/home/rangya/Pycharm/Pycharm_Projects/YOLO_V8/datasets/original/New_Version/Cam_2-Az_90/Tracking_Data_predict.csv'
#     actual_csv_path = '/home/rangya/Pycharm/Pycharm_Projects/YOLO_V8/datasets/original/New_Version/Cam_2-Az_90/Tracking_Data.csv'
#     evaluator = TrackingEvaluator(predicted_csv_path, actual_csv_path, angle_threshold)
#     evaluator.process_tracking_data()
#     evaluator.print_results()
#     evaluator.print_detailed_results()