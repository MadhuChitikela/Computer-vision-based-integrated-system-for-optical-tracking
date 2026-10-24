import sys
import os
import cv2
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List, Dict
from hybrid_detector import HybridDetector
from tracker import RocketTracker
from control_system import GimbalSimulator
from domain_validator import DomainValidator

@dataclass
class TrackingTelemetry:
    frame_idx: int
    timestamp: float
    detected: bool
    bbox: Optional[Tuple]
    confidence: float
    source: str
    tracker_state: str
    zoom: float
    target_zoom: float
    center_offset_x: float
    center_offset_y: float
    process_time_ms: float

class RocketTrackingSystem:
    """
    End-to-end optical rocket tracking system integrating hybrid detection,
    aerospace domain validation, 6-state Kalman tracking, and simulated motorized
    gimbal control with live heads-up display (HUD) telemetry.
    """
    def __init__(self, model_path="rocket_yolo.pt", source_w=720, source_h=576, preset="tracking_telephoto", **kwargs):
        self.source_w, self.source_h = source_w, source_h
        self.detector = HybridDetector(model_path, source_w, source_h)
        self.tracker = RocketTracker(source_w=source_w, source_h=source_h)
        self.gimbal = GimbalSimulator(source_w, source_h, source_w, source_h, preset)
        self.validator = DomainValidator(source_w, source_h)
        self.frame_idx = 0
        self.telemetry_history = []
        self.tracked_count = 0

    def _draw_tracking_overlay(self, frame, target_coords, tracker_state, ox, oy, confidence=0.0, source=""):
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # === TOP-LEFT HUD ===
        panel_w, panel_h = 280, 140
        px, py = 15, 15
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (30, 30, 30), -1)
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 255, 0), 1)

        lines = [
            (f"INITIAL FOCAL: {self.gimbal.base_focal_length:.1f}mm", (150, 150, 150), 0.55, 1),
            (f"CURRENT FOCAL: {self.gimbal.get_focal_length():.1f}mm", (0, 255, 0), 0.80, 2),
            (f"STATE: {tracker_state}", (255, 255, 255), 0.55, 1),
            (f"X Offset: {ox:+.1f} px", (0, 255, 255), 0.50, 1),
            (f"Y Offset: {oy:+.1f} px", (0, 255, 255), 0.50, 1),
            (f"Confidence: {confidence:.2f}", (0, 255, 150), 0.50, 1)
        ]
        y_offset = py + 28
        for text, color, scale, thickness in lines:
            cv2.putText(overlay, text, (px + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            y_offset += 22

        # === TRACKED TARGET CROSSHAIR + BOX ===
        if target_coords is not None and tracker_state in ['LOCKED', 'OCCLUDED_FLOW', 'OCCLUDED_KALMAN']:
            cx, cy, w_box, h_box = target_coords

            if 'yolo_only_1' in source or 'hybrid_yolo_1' in source:
                label, label_color = f"Rocket Body {confidence:.2f}", (0, 255, 0)
            elif 'hybrid_yolo_0' in source or 'flame_only_unmatched' in source:
                label, label_color = f"Engine Flames {confidence:.2f}", (0, 0, 255)
            else:
                label, label_color = f"Target {confidence:.2f}", (255, 255, 0)

            # Target Box
            rx, ry = int(cx - w_box/2), int(cy - h_box/2)
            cv2.rectangle(overlay, (rx, ry), (rx + w_box, ry + h_box), label_color, 2)

            # Crosshair and center dot
            cv2.drawMarker(overlay, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 46, 6)
            cv2.drawMarker(overlay, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
            cv2.circle(overlay, (cx, cy), 10, (0, 0, 0), -1)
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)

            # Dynamic Label Positioning
            font, f_scale, f_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            (tw, th), bl = cv2.getTextSize(label, font, f_scale, f_thick)
            ly1, ly2 = ry - th - 8, ry
            if ly1 < 0:
                ly1, ly2 = ry + h_box, ry + h_box + th + 8
            cv2.rectangle(overlay, (rx, ly1), (rx + tw + 6, ly2), (15, 15, 15), -1)
            cv2.putText(overlay, label, (rx + 3, ly2 - 4), font, f_scale, label_color, f_thick, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        return frame

    def _process_single_frame(self, frame):
        t_start = time.time()
        self.frame_idx += 1

        # Detection & Validation
        candidates = self.detector.detect(frame)
        validated = []
        for c in candidates:
            is_v, score = self.validator.validate_candidate(c, self.tracker.last_state_dict)
            if is_v:
                c['validation_score'] = score
                validated.append(c)

        # Tracker Step
        target = self.tracker.step(frame, validated)
        self.gimbal.update(target)

        # Apply Digital Zoom (Clean frame without candidates)
        zoomed_frame = self.gimbal.apply_zoom_to_frame(frame.copy())
        ox, oy = self.gimbal.get_steering_offsets()

        # Process target for HUD overlay
        v_target = None
        best_conf = 0.0
        best_src = ""

        if target:
            self.tracked_count += 1
            vx, vy = self.gimbal.map_source_to_viewport(target[0], target[1])
            current_zoom_factor = self.gimbal.focal_length / self.gimbal.base_focal_length
            v_target = (vx, vy, int(target[2]*current_zoom_factor), int(target[3]*current_zoom_factor))

            # Match tracker target back to candidates for metadata
            for c in validated:
                rx, ry, rw, rh = c['rocket_bbox']
                if abs((rx + rw/2) - target[0]) < 10 and abs((ry + rh/2) - target[1]) < 10:
                    best_conf = float(c['confidence'])
                    best_src = c['source']
                    break

        final_frame = self._draw_tracking_overlay(zoomed_frame, v_target, self.tracker.state, ox, oy, confidence=best_conf, source=best_src)

        process_time = (time.time() - t_start) * 1000
        tele = TrackingTelemetry(
            self.frame_idx,
            time.time(),
            target is not None,
            target,
            best_conf,
            best_src,
            self.tracker.state,
            self.gimbal.focal_length / self.gimbal.base_focal_length,
            self.gimbal.target_focal_length / self.gimbal.base_focal_length,
            ox,
            oy,
            process_time
        )
        self.telemetry_history.append(tele)
        return final_frame, target, tele

    def process_video(self, video_path, output_name=None):
        """
        Process an input video file and render tracking visualization video with HUD.
        """
        cap = cv2.VideoCapture(video_path)
        if not output_name:
            base = os.path.splitext(os.path.basename(video_path))[0]
            output_name = f"zoomed_{base}.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 25.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_name, fourcc, fps, (self.source_w, self.source_h))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != self.source_w or frame.shape[0] != self.source_h:
                frame = cv2.resize(frame, (self.source_w, self.source_h))
            res, _, _ = self._process_single_frame(frame)
            writer.write(res)
            if self.frame_idx % 50 == 0:
                print(f"Processing frame {self.frame_idx}...", end='\r', flush=True)
        cap.release()
        writer.release()
        print(f"\nVideo processing complete: {output_name}")
        return output_name

    def export_telemetry(self, output_json_path):
        """
        Export all frame telemetry records to a JSON file.
        """
        records = [asdict(t) for t in self.telemetry_history]
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        print(f"✅ Telemetry exported to {output_json_path}")

    def get_summary(self):
        last = self.telemetry_history[-1] if self.telemetry_history else None
        return {
            'total_frames': self.frame_idx,
            'tracked_frames': self.tracked_count,
            'tracking_rate': self.tracked_count / max(1, self.frame_idx),
            'lost_frames': self.frame_idx - self.tracked_count,
            'final_tracker_state': self.tracker.state,
            'final_residual_azimuth': last.center_offset_x if last else 0.0,
            'final_residual_elevation': last.center_offset_y if last else 0.0,
            'final_azimuth': last.center_offset_x if last else 0.0,
            'final_elevation': last.center_offset_y if last else 0.0,
            'final_focal_length': self.gimbal.focal_length if last else 0.0
        }
