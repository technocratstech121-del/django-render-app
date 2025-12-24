"""
models_integration.py
YOLOv8m-based detectors for Railway Safety Monitoring
- Object detection (people, phone, bags, bottles)
- Pose detection (gestures: phone usage, drinking, signal callout, cross talk)
- ROI-based fusion for higher accuracy
- Event prioritization: only the most critical event per frame is logged
"""

import cv2
import numpy as np
from ultralytics import YOLO


class MultiModelDetector:
    def __init__(self, use_yolo=True, use_pose=True,
                 obj_model="yolov8m.pt", pose_model="yolov8m-pose.pt"):
        self.use_yolo = use_yolo
        self.use_pose = use_pose

        # --- YOLOv8 Object Detector ---
        if self.use_yolo:
            self.obj_detector = YOLO(obj_model)
            self.yolo_classes = {
                0: "person",
                67: "cell phone",
                24: "backpack",
                41: "cup",
                46: "bottle"
            }

        # --- YOLOv8 Pose Detector ---
        if self.use_pose:
            self.pose_detector = YOLO(pose_model)

    # -------------------- Object Detection --------------------
    def detect_objects(self, frame):
        results = []
        if not self.use_yolo:
            return results
        dets = self.obj_detector(frame, verbose=False)
        for box in dets[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if conf > 0.5 and cls in self.yolo_classes:
                results.append({"label": self.yolo_classes[cls], "conf": conf,
                                "bbox": box.xyxy[0].tolist()})
        return results

    # -------------------- Pose Detection --------------------
    def detect_pose(self, frame):
        if not self.use_pose:
            return None
        results = self.pose_detector(frame, verbose=False)
        return results[0] if results else None

    # -------------------- Event Mapping --------------------
    def detect_events(self, frame, roi=None):
        """
        Runs YOLOv8 object + pose models on ROI (if provided) and maps
        detections to railway safety events.
        """
        events = []

        # --- Crop ROI (e.g., motorman chair) ---
        if roi:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            frame_roi = frame[y:y + h, x:x + w]
        else:
            frame_roi = frame

        # --- Resize ROI to 640x640 for YOLO ---
        roi_resized = cv2.resize(frame_roi, (640, 640))

        # --- Object detection ---
        objects = self.detect_objects(roi_resized)
        people = [o for o in objects if o["label"] == "person"]
        phones = [o for o in objects if o["label"] == "cell phone"]
        bags = [o for o in objects if o["label"] == "backpack"]
        drinks = [o for o in objects if o["label"] in ["bottle", "cup"]]

        if len(people) > 2:
            events.append(("Excess_People_in_Cab", 0.8))
        if len(people) > 1:
            events.append(("Multiple_at_Motorman_Chair", 0.8))
        if bags:
            events.append(("Packing_Belongings", bags[0]["conf"]))

        # --- Pose-based events ---
        pose_result = self.detect_pose(roi_resized)
        if pose_result and pose_result.keypoints is not None:
            keypoints = pose_result.keypoints.xy[0].cpu().numpy()

            if len(keypoints) > 10:
                head = keypoints[0]       # nose
                right_wrist = keypoints[10]
                left_wrist = keypoints[9]

                def near_head(hand, head, thresh=80):
                    return np.linalg.norm(hand - head) < thresh

                # ✅ Signal Call Out: wrist above head
                if (right_wrist[1] < head[1]) or (left_wrist[1] < head[1]):
                    events.append(("Signal_Call_Out", 0.7))

                # ✅ Fused Mobile Usage
                if phones:
                    if near_head(right_wrist, head) or near_head(left_wrist, head):
                        events.append(("Mobile_Usage", phones[0]["conf"]))

                # ✅ Fused Food/Beverage (drinking gesture)
                if drinks:
                    if near_head(right_wrist, head) or near_head(left_wrist, head):
                        events.append(("Food_Beverage", drinks[0]["conf"]))

                # ✅ Cross Talk Detection
                if len(people) > 1:
                    # Approximation: multiple heads & wrist variance = talking
                    mouth_region = keypoints[9:11]  # around mouth/wrist
                    if np.std(mouth_region[:, 1]) > 10:  # vertical movement
                        events.append(("Cross_Talk", 0.7))

        # ---------------- Event Prioritization ----------------
        if events:
            priority_map = {
                "Mobile_Usage": 5,
                "Cross_Talk": 4,
                "Excess_People_in_Cab": 4,
                "Multiple_at_Motorman_Chair": 3,
                "Food_Beverage": 2,
                "Packing_Belongings": 1,
                "Signal_Call_Out": -1  # positive action
            }

            # Sort by priority → confidence
            events.sort(key=lambda e: (priority_map.get(e[0], 0), e[1]), reverse=True)

            # Keep only highest-priority event
            events = [events[0]]

        return events
