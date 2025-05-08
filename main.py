import time
import os
import sys
import cv2
import numpy as np
from supervision import Detections

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

from util import (BYTETrackerArgs, detections2boxes,
                  match_detections_with_tracks, convert_bbox_list_resized_to_original,
                  add_fps)

from configs import ENV
from infer import FaceAlignment, RetinaFaceInfer
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class RecogitionAI:
    def __init__(self, config) -> None:
        self.config = config
        self.facemodel = RetinaFaceInfer(ENV.FACEMODEL_PATH)
        self.facealign = FaceAlignment()
        if self.config['liveness']:
            print('load liveness')
            self.livenessmodel = YOLO(ENV.LIVENESS_PATH)

    def start(self, show=True):
        class_ids = [0]
        byte_tracker = BYTETracker(BYTETrackerArgs())
        font_color = (0, 255, 0)  
        font = cv2.FONT_HERSHEY_SIMPLEX

        cap = cv2.VideoCapture(0)
        st_time_0 = time.time()

        while cap.isOpened():
            current_time = datetime.now()
            current_time.strftime("%Y-%m-%d %H:%M:%S")
            print("=" * 100)

            start_time = time.time()
            ret, org_frame = cap.read()

            if not ret:
                break
            h0, w0 = org_frame.shape[:2]
            x1p, y1p, x2p, y2p = [0, 0, w0, h0]

            frame = org_frame[y1p:y2p, x1p:x2p]
            frame_resize = cv2.resize(frame, None, fx=0.5, fy=0.5)
            dets, _ = self.facemodel.detect(frame_resize)
            if len(dets) == 0:
                print("No Face")
                frame = add_fps(frame, start_time)
                org_frame[y1p:y2p, x1p:x2p] = frame
                org_frame = cv2.rectangle(org_frame, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
                if show:
                    cv2.imshow("Face Detection", org_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            print("Have Face")
            dets[:, :4] = convert_bbox_list_resized_to_original(
                frame_resize.shape, dets[:, :4], frame.shape)

            xyxy = dets[:, :4]
            scale_box = ENV.RATIO_TRACKING_BBOX
            xyxy_scale = np.column_stack((xyxy[:, 0] + (xyxy[:, 2] - xyxy[:, 0]) * scale_box,
                                          xyxy[:, 1] + (xyxy[:, 3] - xyxy[:, 1]) * scale_box,
                                          xyxy[:, 2] - (xyxy[:, 2] - xyxy[:, 0]) * scale_box,
                                          xyxy[:, 3] - (xyxy[:, 3] - xyxy[:, 1]) * scale_box))

            confidence = dets[:, 4]
            class_id = np.array([0] * len(dets))
            detections = Detections(
                xyxy=xyxy_scale,
                confidence=confidence,
                class_id=class_id
            )
            mask = np.array(
                [class_id in class_ids for class_id in detections.class_id], dtype=bool)
            detections = detections[mask]
            xyxy = xyxy[mask]
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(
                detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            mask = np.array(
                [tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections = detections[mask]
            xyxy = xyxy[mask]

            for _, det in zip(tracker_id, dets):
                try:
                    x1, y1, x2, y2 = np.array(det[:4]).astype(int)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    name_put = "Face"
                    cv2.putText(frame, name_put, (x1, y2+30), font,
                                1, font_color, 2, cv2.LINE_AA)
                    image_with_text = np.array(Image.fromarray(frame))
                    img_pil = Image.fromarray(image_with_text)
                    frame = np.array(img_pil)
                    x1, y1, x2, y2 = np.array(det[:4]).astype(int)
                    face_org = frame[y1:y2, x1:x2, :].copy()
                    face = cv2.resize(face_org, (112, 112))
                except Exception as e:
                    print("Exception: ", e)
                    continue
                straight = self.facealign.straight(det)
                w = x2 - x1
                h = y2 - y1
                face_org = frame[y1:y2, x1:x2, :].copy()
                aligned_face, _ = self.facealign.align(face_org, det)
                face = cv2.resize(aligned_face, (112, 112))
                
                if (1 - ENV.RATIO_THRESHOLD_STRAIGHT < straight < 1 + ENV.RATIO_THRESHOLD_STRAIGHT and
                        w > 112 * ENV.RATIO_THRESHOLD_FACE_SIZE and h > 112 * ENV.RATIO_THRESHOLD_FACE_SIZE and
                        0.8 < h / w < 1.8):
                    if self.config['liveness']:
                        cls = self.livenessmodel(face, verbose=False)[0].probs
                        conf = np.round(cls.top1conf.numpy(), 2)
                        real = int(cls.top1)
                        if real == 1 and conf > 0.99:
                            bbox_color = (255, 0, 0)
                            print("Live")
                            cv2.putText(frame, f"Live_{conf}", (x2 - 50, y1),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        
                        else:
                            bbox_color = (0, 0, 255)
                            print("Spoof")
                            cv2.putText(frame, f"Spoof_{conf}", (x2 - 50, y1),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                
            frame = add_fps(frame, start_time)
            
            # out.write(frame)

            if show:
                cv2.imshow("Face Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        print('Time process: ', time.time() - st_time_0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = {
        "liveness": True,
    }

    camerasys = RecogitionAI(config=config)
    camerasys.start(show=True)
