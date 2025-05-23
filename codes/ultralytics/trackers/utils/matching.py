# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import bbox_ioa

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lapx>=0.5.2")  # update to lap package from https://github.com/rathaROG/lapx
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True) -> tuple:
    """
    Perform linear assignment using scipy or lap.lapjv.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments.
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool, optional): Whether to use lap.lapjv. Defaults to True.

    Returns:
        Tuple with:
            - matched indices
            - unmatched indices from 'a'
            - unmatched indices from 'b'
    """

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # Use lap.lapjv
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Use scipy.optimize.linear_sum_assignment
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(set(np.arange(cost_matrix.shape[0])) - set(matches[:, 0]))
            unmatched_b = list(set(np.arange(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU.
    """

    if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        ious = bbox_ioa(
            np.ascontiguousarray(atlbrs, dtype=np.float32), np.ascontiguousarray(btlbrs, dtype=np.float32), iou=True
        )
    return 1 - ious  # cost matrix

def euclidean_distance_normalized(atracks, btracks, scale=1.0,max_distance=40):
    """
    Compute cost based on normalized Euclidean distance between the centers of bounding boxes using Sigmoid function.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.
        scale (float): Scale factor to adjust the sensitivity of the sigmoid function.

    Returns:
        (np.ndarray): Cost matrix computed based on normalized Euclidean distance.
    """

    # Extract bounding boxes if necessary
    def calculate_center(box):
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        return np.array([x_center, y_center])

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    # Initialize the cost matrix
    costs = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    for i, atlbr in enumerate(atlbrs):
        for j, btlbr in enumerate(btlbrs):
            center_a = calculate_center(atlbr)
            center_b = calculate_center(btlbr)
            distance = np.linalg.norm(center_a - center_b)
            if distance > max_distance:
                costs[i, j] = 1.0
            else:
                costs[i, j] = 1 - sigmoid(scale * (max_distance - distance))  # 距离越远，成本越接近 1
    return costs


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """
    Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[STrack]): List of tracks.
        detections (list[BaseTrack]): List of detections.
        metric (str, optional): Metric for distance computation. Defaults to 'cosine'.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings.
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    if det_features.shape[1] != 1: det_features_reshaped = det_features.reshape(det_features.shape[0], 1, det_features.shape[1])
    else: det_features_reshaped = det_features
    if track_features.shape[1] != 1: track_features_reshaped = track_features.reshape(track_features.shape[0], 1, track_features.shape[1])
    else: track_features_reshaped = track_features
    
    det_features_squeezed = np.squeeze(det_features_reshaped, axis=1)
    track_features_squeezed = np.squeeze(track_features_reshaped, axis=1)

    cost_matrix = np.maximum(0.0, cdist(track_features_squeezed, det_features_squeezed, metric))  # Normalized features
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """
    Fuses cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments.
        detections (list[BaseTrack]): List of detections with scores.

    Returns:
        (np.ndarray): Fused similarity matrix.
    """

    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
