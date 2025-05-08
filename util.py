import time
from dataclasses import dataclass
from typing import List
import cv2
import numpy as np
import torch
from onemetric.cv.utils.iou import box_iou_batch
from supervision import Detections


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
        detections: Detections,
        tracks: List
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def convert_bbox_resized_to_original(resized_size, bbox_resized, original_size):
    """
    Chuyển đổi bounding box từ ảnh đã resize sang ảnh gốc.

    Tham số:
    - bbox_resized: Tuple (x_min, y_min, x_max, y_max) biểu thị bounding box trên ảnh đã resize.
    - original_size: Tuple (width, height) kích thước của ảnh gốc.
    - resized_size: Tuple (width, height) kích thước của ảnh đã resize.

    Trả về:
    - Tuple (x_min_original, y_min_original, x_max_original, y_max_original) biểu thị bounding box trên ảnh gốc.
    """
    x_min_resized, y_min_resized, x_max_resized, y_max_resized = bbox_resized
    original_width, original_height = original_size
    resized_width, resized_height = resized_size

    # Tính tỷ lệ thay đổi kích thước
    width_scale = original_width / resized_width
    height_scale = original_height / resized_height

    # Áp dụng tỷ lệ để tính toán bounding box trên ảnh gốc
    x_min_original = x_min_resized * width_scale
    y_min_original = y_min_resized * height_scale
    x_max_original = x_max_resized * width_scale
    y_max_original = y_max_resized * height_scale

    return (x_min_original, y_min_original, x_max_original, y_max_original)


def convert_bbox_list_resized_to_original(resized_size, bbox_list_resized, original_size, ):
    """
    Chuyển đổi danh sách bounding boxes từ ảnh đã resize sang ảnh gốc.

    Tham số:
    - bbox_list_resized: Danh sách các bounding boxes, mỗi bounding box là một tuple (x_min, y_min, x_max, y_max).
    - original_size: Tuple (width, height) kích thước của ảnh gốc.
    - resized_size: Tuple (width, height) kích thước của ảnh đã resize.

    Trả về:
    - Danh sách các bounding boxes trên ảnh gốc.
    """
    original_height, original_width, _ = original_size
    resized_height, resized_width, _ = resized_size

    # Tính tỷ lệ thay đổi kích thước
    width_scale = original_width / resized_width
    height_scale = original_height / resized_height

    # Sử dụng list comprehension để chuyển đổi từng bounding box trong danh sách
    bbox_list_original = [(x_min_resized * width_scale, y_min_resized * height_scale, x_max_resized * width_scale,
                           y_max_resized * height_scale) for x_min_resized, y_min_resized, x_max_resized, y_max_resized
                          in bbox_list_resized]

    return bbox_list_original


def add_fps(frame, start_time):
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    print("FPS: ", fps)
    return frame
