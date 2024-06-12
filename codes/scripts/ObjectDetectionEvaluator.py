import os
import numpy as np


class ObjectDetectionEvaluator:
    def __init__(self, predict_folder, camera_name,label_folder, evaluating_result, num_classes, iou_threshold=0.5):
        self.predict_folder = predict_folder
        self.label_folder = label_folder
        self.camera_name = camera_name
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes + 1
        self.evaluating_result = evaluating_result

    @staticmethod
    def read_bboxes(file_path):
        with open(file_path) as f:
            bboxes = [list(map(float, line.split())) for line in f]
        return bboxes

    @staticmethod
    def bbox_to_coords(bbox):
        cx = bbox[1]; cy = bbox[2];w = bbox[3];h = bbox[4]
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        return [x_min, y_min, x_max, y_max]

    def compute_iou(self,bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = self.bbox_to_coords(bbox1)
        x2_min, y2_min, x2_max, y2_max = self.bbox_to_coords(bbox2)
        intersect_x_min = max(x1_min, x2_min)
        intersect_y_min = max(y1_min, y2_min)
        intersect_x_max = min(x1_max, x2_max)
        intersect_y_max = min(y1_max, y2_max)

        intersect_area = max(0, intersect_x_max - intersect_x_min) * max(0, intersect_y_max - intersect_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

        iou = intersect_area / float(bbox1_area + bbox2_area - intersect_area)

        return iou

    def append_to_file(self, text):
        with open(self.evaluating_result, 'a') as file:
            file.write(text + '\n')

    def evaluate_detection(self):
        if not os.path.exists(self.predict_folder):
            print(f"The predict folder does not exist. Creating: {self.predict_folder}")
            os.makedirs(self.predict_folder)
        predict_files = os.listdir(self.predict_folder)
        true_positives, false_positives, false_negatives = 0, 0, 0
        total_target = 0; total_predict = 0
        for filename in predict_files:
            pred_bboxes = self.read_bboxes(os.path.join(self.predict_folder, filename))
            true_bboxes = self.read_bboxes(os.path.join(self.label_folder, filename))

            matched = []

            for pred_bbox in pred_bboxes:
                for true_bbox in true_bboxes:
                    if self.compute_iou(pred_bbox, true_bbox) > self.iou_threshold and true_bbox not in matched:
                        true_positives += 1
                        matched.append(true_bbox)
                        break
                else:
                    false_positives += 1
            false_negatives += len(true_bboxes) - len(matched)
            total_target += len(true_bboxes); total_predict += len(pred_bboxes)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results = []
        separator = "\n" + "-" * 50 + "\n"
        title = "Detection Results of " + self.camera_name + "\n"
        self.append_to_file(separator.strip())  # 移除首尾的换行符
        self.append_to_file(title.strip())
        results.append(f"Total true labels: {total_target}, Total predictions: {total_predict}")
        results.append(f"True positives: {true_positives}, False positives: {false_positives}, False negatives: {false_negatives}")
        results.append(f"Precision: {precision}, Recall: {recall}, F1 score: {f1_score}")
        print(separator)
        print(title)
        for result in results:
            print(result)
            self.append_to_file(result)

        return {
            'total_true_labels': total_target,
            'total_predictions': total_predict,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def evaluate_detection_with_class(self):
        predict_files = os.listdir(self.predict_folder)
        self.num_classes = int(self.num_classes)
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

        for filename in predict_files:
            pred_bboxes = self.read_bboxes(os.path.join(self.predict_folder, filename))
            true_bboxes = self.read_bboxes(os.path.join(self.label_folder, filename))

            matched = []

            for pred_bbox in pred_bboxes:
                best_iou = 0
                best_true_bbox = None
                for true_bbox in true_bboxes:
                    iou = self.compute_iou(pred_bbox, true_bbox)
                    if iou > best_iou and true_bbox not in matched:
                        best_iou = iou
                        best_true_bbox = true_bbox

                if best_iou > self.iou_threshold:
                    matched.append(best_true_bbox)
                    true_class = int(best_true_bbox[0])
                    pred_class = int(pred_bbox[0])
                    confusion_matrix[pred_class, true_class] += 1
                else:
                    pred_class = int(pred_bbox[0])
                    # Mistakenly detected as a class (not background)
                    confusion_matrix[pred_class, -1] += 1

            for true_bbox in true_bboxes:
                if true_bbox not in matched:
                    true_class = int(true_bbox[0])
                    # Missed detection counted as detected as background
                    confusion_matrix[-1, true_class] += 1

        # precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0, where=~np.isnan(confusion_matrix))
        # recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1, where=~np.isnan(confusion_matrix))
        precision_denominator = np.sum(confusion_matrix, axis=0)
        precision = np.divide(np.diag(confusion_matrix), precision_denominator, where=precision_denominator!=0)
        precision = np.nan_to_num(precision)
        recall_denominator = np.sum(confusion_matrix, axis=1)
        recall = np.divide(np.diag(confusion_matrix), recall_denominator, where=recall_denominator!=0)
        recall = np.nan_to_num(recall)

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid division by zero

        self.append_to_file("Confusion Matrix:\n" + str(confusion_matrix))
        self.append_to_file(f"Precision per class: {precision}")
        self.append_to_file(f"Recall per class: {recall}")
        self.append_to_file(f"F1 Score per class: {f1_score}")

        return confusion_matrix, precision, recall, f1_score

    def evaluate_multiple_datasets(self,all_detection_results,total_confusion_matrix):
        total_results = {'total_true_labels': 0, 'total_predictions': 0, 'true_positives': 0, 'false_positives': 0, 'false_negatives': 0, }
        for i in range(len(all_detection_results)):
            for key in total_results.keys():
                total_results[key] += all_detection_results[i][key]

        # overall_precision = total_results['true_positives'] / (total_results['true_positives'] + total_results['false_positives'])
        # overall_recall = total_results['true_positives'] / (total_results['true_positives'] + total_results['false_negatives'])
        if (total_results['true_positives'] + total_results['false_positives']) == 0:
            overall_precision = 0; overall_recall = 0; overall_f1_score = 0
        else:
            overall_precision = total_results['true_positives'] / (total_results['true_positives'] + total_results['false_positives'])
            overall_recall = total_results['true_positives'] / (total_results['true_positives'] + total_results['false_negatives'])
            overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-6)  # Avoid division by zero

        total_confusion_matrix_safe = np.nan_to_num(total_confusion_matrix, nan=0.0, posinf=np.finfo(np.float64).max)
        class_precision = np.diag(total_confusion_matrix_safe) / (np.sum(total_confusion_matrix_safe, axis=0) + 1e-10)

        # class_precision = np.diag(total_confusion_matrix) / np.sum(total_confusion_matrix, axis=0, where=~np.isnan(total_confusion_matrix))
        class_totals = np.sum(total_confusion_matrix, axis=1, where=~np.isnan(total_confusion_matrix))
        class_recall = np.where(class_totals > 0, np.diag(total_confusion_matrix) / class_totals, np.nan)
        # class_recall = np.diag(total_confusion_matrix) / np.sum(total_confusion_matrix, axis=1, where=~np.isnan(total_confusion_matrix))
        class_f1_score = 2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-6)  # Avoid division by zero

        results = []
        separator = "\n" + "-" * 50 + "\n"
        title = "Overall Evaluation Results:\n"
        self.append_to_file(separator.strip())  # 移除首尾的换行符
        self.append_to_file(title.strip())
        results.append(f"Total true labels: {total_results['total_true_labels']}, Total predictions: {total_results['total_predictions']}")
        results.append(f"True positives: {total_results['true_positives']}, False positives: {total_results['false_positives']}, False negatives: {total_results['false_negatives']}")
        results.append(f"Precision: {overall_precision}, Recall: {overall_recall}, F1 score: {overall_f1_score}")
        print(separator)
        print(title)
        for result in results:
            print(result)
            self.append_to_file(result)
        self.append_to_file("Confusion Matrix:\n" + str(total_confusion_matrix))
        self.append_to_file(f"Precision per class: {class_precision}")
        self.append_to_file(f"Recall per class: {class_recall}")
        self.append_to_file(f"F1 Score per class: {class_f1_score}")

        return {
            'total_true_labels': total_results['total_true_labels'],
            'total_predictions': total_results['total_predictions'],
            'true_positives': total_results['true_positives'],
            'false_positives': total_results['false_positives'],
            'false_negatives': total_results['false_negatives'],
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1_score
        }
    # 使用示例
    # evaluator = ObjectDetectionEvaluator('path_to_predict_folder', 'path_to_labels_folder')
    # result = evaluator.evaluate_detection()
    # print(result)

















