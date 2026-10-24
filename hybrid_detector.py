import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction

class HybridDetector:
    """
    Dual-mode hybrid detector combining deep learning (YOLOv8 + optional SAHI sliced inference)
    with classical computer vision (HSV flame color filtering + morphological core detection).
    """
    def __init__(self, model_path="rocket_yolo.pt", source_w=720, source_h=576):
        self.source_w = source_w
        self.source_h = source_h
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if model_path == "rocket_yolo.pt" and not os.path.exists(model_path):
            print("⚠️ rocket_yolo.pt not found! Falling back to pre-trained yolov8n.pt")
            model_path = "yolov8n.pt"

        self.model = YOLO(model_path)
        self.sahi_model = UltralyticsDetectionModel(
            model_path=model_path,
            device=self.device,
            confidence_threshold=0.35
        )

        if "rocket_yolo" in model_path:
            self.target_classes = [0, 1]
            print("🎯 Custom Rocket Model loaded! Target classes: [0(Flames), 1(Body)]")
        else:
            self.target_classes = [4, 14, 33]
            print("🎯 Pre-trained COCO Model loaded! Target classes: [4(Airplane), 14(Bird), 33(Kite)]")

    def detect(self, frame, last_target_bbox=None, use_sahi=False, strict_mode=False):
        """
        Run hybrid detection on a single frame.

        Args:
            frame: input BGR image (numpy array)
            last_target_bbox: optional previous bbox
            use_sahi: whether to run SAHI sliced inference
            strict_mode: applies higher contrast & solidity thresholds

        Returns:
            final_candidates: list of candidate dicts
        """
        plume_candidates = self._detect_plume(frame, strict_mode=strict_mode)
        yolo_boxes = []

        if use_sahi:
            try:
                result = get_sliced_prediction(
                    frame, self.sahi_model,
                    slice_height=256, slice_width=256,
                    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                    verbose=0
                )
                for obj in result.object_prediction_list:
                    bbox = obj.bbox.to_xyxy()
                    cls_id = obj.category.id
                    conf = obj.score.value
                    if cls_id == 1 and conf < 0.45:
                        continue
                    yolo_boxes.append((int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1]), cls_id, conf))
            except Exception as e:
                yolo_boxes = self._run_standard_yolo(frame)
        else:
            yolo_boxes = self._run_standard_yolo(frame)
            # Additional filter for YOLO-only detections
            yolo_boxes = [box for box in yolo_boxes if not (box[4] == 1 and box[5] < 0.40)]

        hybrid_candidates = self._merge_detections(plume_candidates, yolo_boxes, frame.shape, strict_mode=strict_mode)
        hybrid_candidates = self._nms(hybrid_candidates)

        # Final filter: keep only the best detection per class
        best_body = None
        best_flame = None
        final_candidates = []

        for c in hybrid_candidates:
            source = c.get('source', '')
            if 'yolo_only_1' in source or 'hybrid_yolo_1' in source:
                if best_body is None or c['confidence'] > best_body['confidence']:
                    best_body = c
            else:
                if best_flame is None or c['confidence'] > best_flame['confidence']:
                    best_flame = c

        if best_body:
            final_candidates.append(best_body)
        if best_flame:
            final_candidates.append(best_flame)

        return final_candidates

    def _detect_plume(self, frame, strict_mode=False):
        """
        Segment high-intensity exhaust core and orange/red flame regions.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if strict_mode:
            percentile_thresh = np.percentile(gray, 98.5)
            core_thresh = max(150, percentile_thresh) if percentile_thresh > 200 else 150
            core_thresh = min(core_thresh, 245)
        else:
            core_thresh = 150

        _, mask_core = cv2.threshold(gray, core_thresh, 255, cv2.THRESH_BINARY)

        lower_fire1 = np.array([0, 40, 120])
        upper_fire1 = np.array([38, 255, 255])
        mask_fire1 = cv2.inRange(hsv, lower_fire1, upper_fire1)

        lower_fire2 = np.array([170, 40, 120])
        upper_fire2 = np.array([180, 255, 255])
        mask_fire2 = cv2.inRange(hsv, lower_fire2, upper_fire2)

        mask_orange = cv2.bitwise_or(mask_fire1, mask_fire2)
        mask_plume = cv2.bitwise_or(mask_core, mask_orange)

        # Mask broadcast logos at top corners
        logo_h = int(h * 0.28)
        logo_w = int(w * 0.22)
        mask_plume[0:logo_h, 0:logo_w] = 0
        mask_plume[0:logo_h, w - logo_w:w] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_plume = cv2.morphologyEx(mask_plume, cv2.MORPH_OPEN, kernel)

        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 80)) # Larger kernel to merge nearby plume segments
        mask_plume = cv2.morphologyEx(mask_plume, cv2.MORPH_CLOSE, merge_kernel)

        mask_plume[0:logo_h, 0:logo_w] = 0
        mask_plume[0:logo_h, w - logo_w:w] = 0

        contours, _ = cv2.findContours(mask_plume, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = max(30, int(frame_area * 0.00005))
        max_area = frame_area * (0.20 if strict_mode else 0.65)

        raw_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, w_box, h_box = cv2.boundingRect(contour)

            aspect_ratio_plume = h_box / float(max(1, w_box))
            if w_box < (0.05 * self.source_w) and aspect_ratio_plume > 6.0: # Filter out very thin vertical noise
                continue

            solidity = 0.0
            if strict_mode:
                bbox_area = max(1, w_box * h_box)
                solidity = area / float(bbox_area)
                if solidity < 0.18: # Filter out sparse contours
                    continue

            roi_orange = mask_orange[y:y+h_box, x:x+w_box]
            roi_core = mask_core[y:y+h_box, x:x+w_box]
            blob_pixels = w_box * h_box
            orange_ratio = cv2.countNonZero(roi_orange) / float(blob_pixels) if blob_pixels > 0 else 0.0
            core_ratio = cv2.countNonZero(roi_core) / float(blob_pixels) if blob_pixels > 0 else 0.0

            is_fire_color = (orange_ratio > 0.35 and core_ratio > 0.10) or \
                            (core_ratio > 0.20 and orange_ratio > 0.15)

            raw_candidates.append({
                'bbox': (x, y, w_box, h_box),
                'area': area,
                'centroid': (x + w_box//2, y + h_box//2),
                'is_fire_color': is_fire_color,
                'solidity': solidity
            })

        fire_candidates = [c for c in raw_candidates if c['is_fire_color']]
        candidates = []
        if fire_candidates:
            # Merge all fire candidates into one union bounding box
            min_x = min(c['bbox'][0] for c in fire_candidates)
            min_y = min(c['bbox'][1] for c in fire_candidates)
            max_x = max(c['bbox'][0] + c['bbox'][2] for c in fire_candidates)
            max_y = max(c['bbox'][1] + c['bbox'][3] for c in fire_candidates)
            union_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            union_area = (max_x - min_x) * (max_y - min_y)
            candidates.append({
                'bbox': union_bbox,
                'area': union_area,
                'centroid': (min_x + (max_x - min_x)//2, min_y + (max_y - min_y)//2),
                'is_fire_color': True,
                'solidity': union_area / max(1, (max_x - min_x) * (max_y - min_y))
            })
        return candidates

    def _run_standard_yolo(self, frame):
        results = self.model(frame, verbose=False, device=self.device)
        boxes = []
        for r in results:
            # Handle standard detection boxes
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Stricter confidence thresholds
                    if cls_id == 1:  # Body detection
                        if conf < 0.40:
                            continue
                    elif cls_id == 0:  # Flame detection
                        if conf < 0.30:
                            continue
                    else:
                        if conf < 0.35:
                            continue

                    xyxy = box.xyxy[0].tolist()
                    bx, by, bw, bh = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])
                    boxes.append((bx, by, bw, bh, cls_id, conf))

            # Handle Oriented Bounding Box (OBB) model outputs
            elif hasattr(r, 'obb') and r.obb is not None and len(r.obb) > 0:
                for obb in r.obb:
                    cls_id = int(obb.cls[0])
                    conf = float(obb.conf[0])

                    if cls_id == 1:
                        if conf < 0.40:
                            continue
                    elif cls_id == 0:
                        if conf < 0.30:
                            continue
                    else:
                        if conf < 0.35:
                            continue

                    xyxy = obb.xyxy[0].tolist()
                    bx, by, bw, bh = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])
                    boxes.append((bx, by, bw, bh, cls_id, conf))

        return boxes


    def _nms(self, candidates, iou_thresh=0.4):
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        With confidence weighting and smaller box overlap handling.
        """
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda c: c['confidence'], reverse=True)
        keep = []

        for i, c in enumerate(candidates):
            rx, ry, rw, rh = c['rocket_bbox']
            overlap = False

            for j, k in enumerate(keep):
                kx, ky, kw, kh = k['rocket_bbox']

                ix = max(rx, kx)
                iy = max(ry, ky)
                iw = min(rx + rw, kx + kw) - ix
                ih = min(ry + rh, ky + kh) - iy

                if iw > 0 and ih > 0:
                    inter = iw * ih
                    area_c = rw * rh
                    area_k = kw * kh
                    min_area = min(area_c, area_k)

                    if min_area > 0 and inter / min_area > iou_thresh:
                        overlap = True
                        if k['confidence'] < c['confidence']:
                            keep[j] = c
                        break

            if not overlap:
                keep.append(c)

        # Merge near-identical detections
        merged = []
        for c in keep:
            if not merged:
                merged.append(c)
                continue

            duplicate = False
            for m in merged:
                rx, ry, rw, rh = c['rocket_bbox']
                mx, my, mw, mh = m['rocket_bbox']

                cx1, cy1 = rx + rw/2, ry + rh/2
                cx2, cy2 = mx + mw/2, my + mh/2
                center_dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

                if center_dist < 20:
                    duplicate = True
                    if c['confidence'] > m['confidence']:
                        merged[-1] = c
                    break

            if not duplicate:
                merged.append(c)

        return merged

    def _merge_detections(self, plume_candidates, yolo_boxes, frame_shape=None, strict_mode=False):
        merged = []
        used_yolo_indices = set()

        if frame_shape is not None:
            h, w = frame_shape[:2]
        else:
            h, w = self.source_h, self.source_w

        logo_h = int(h * 0.28)
        logo_w = int(w * 0.22)

        def _in_logo_zone(cx, cy):
            in_top_left = (cx < logo_w) and (cy < logo_h)
            in_top_right = (cx > (w - logo_w)) and (cy < logo_h)
            return in_top_left or in_top_right

        # Find the best body detection
        best_body_bbox = None
        best_body_conf = 0.0
        best_body_idx = -1

        for idx, ybox in enumerate(yolo_boxes):
            yx, yy, yw, yh, cls_id, conf = ybox
            if cls_id == 1 and conf > best_body_conf and conf >= 0.35:
                best_body_bbox = (yx, yy, yw, yh)
                best_body_conf = conf
                best_body_idx = idx

        # Process each plume candidate
        for p in plume_candidates:
            px, py, pw, ph = p['bbox']
            p_cy = py + ph / 2.0
            p_cx = px + pw / 2.0

            if _in_logo_zone(p_cx, p_cy):
                continue

            # Find the best overlapping YOLO detection for this plume
            best_yolo_candidate = None
            best_overlap_area = 0
            best_idx = -1

            for idx, ybox in enumerate(yolo_boxes):
                if idx in used_yolo_indices:
                    continue
                yx, yy, yw, yh, cls_id, conf = ybox

                ix = max(px, yx)
                iy = max(py, yy)
                iw = min(px + pw, yx + yw) - ix
                ih = min(py + ph, yy + yh) - iy

                if iw > 0 and ih > 0:
                    overlap = iw * ih
                    if overlap > best_overlap_area:
                        best_overlap_area = overlap
                        best_yolo_candidate = ybox
                        best_idx = idx

            if best_yolo_candidate is None:
                if p.get('is_fire_color', False):
                    if best_body_bbox is not None:
                        bx, by, bw, bh = best_body_bbox
                        if abs(p_cx - (bx + bw/2)) > w * 0.15:
                            continue
                    merged.append({
                        'rocket_bbox': (px, py, pw, ph),
                        'plume_bbox': (px, py, pw, ph),
                        'confidence': 0.35,
                        'source': 'flame_only_unmatched'
                    })
                continue

            yx, yy, yw, yh, cls_id, conf = best_yolo_candidate

            if cls_id == 0 and not p.get('is_fire_color', False):
                continue

            if cls_id == 0 and best_body_bbox is not None:
                bx, by, bw, bh = best_body_bbox
                merged.append({
                    'rocket_bbox': (bx, by, bw, bh),
                    'plume_bbox': (px, py, pw, ph),
                    'confidence': best_body_conf,
                    'source': 'hybrid_yolo_1'
                })
                used_yolo_indices.add(best_body_idx)
                continue

            if cls_id == 0:
                rx, ry, rw, rh = px, py, pw, ph
                if best_body_bbox is not None:
                    bx, by, bw, bh = best_body_bbox
                    rocket_body_bottom_y = by + bh
                    if ry < rocket_body_bottom_y and not (rx + rw < bx or rx > bx + bw):
                        new_ry = rocket_body_bottom_y
                        new_rh = rh - (new_ry - ry)
                        if new_rh > 5:
                            ry, rh = new_ry, new_rh
                        else:
                            continue
            else:
                rx = min(px, yx)
                ry = min(py, yy)
                rw = max(px + pw, yx + yw) - rx
                rh = max(py + ph, yy + yh) - ry
                if rh > yh * 4.0 or rw > yw * 4.0:
                    rx, ry, rw, rh = yx, yy, yw, yh

            duplicate = False
            for existing in merged:
                ex, ey, ew, eh = existing['rocket_bbox']
                cx1, cy1 = rx + rw/2, ry + rh/2
                cx2, cy2 = ex + ew/2, ey + eh/2
                if ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5 < 15:
                    duplicate = True
                    if conf > existing['confidence']:
                        existing.update({
                            'rocket_bbox': (rx, ry, rw, rh),
                            'confidence': conf,
                            'source': f'hybrid_yolo_{cls_id}'
                        })
                    break

            if not duplicate:
                merged.append({
                    'rocket_bbox': (rx, ry, rw, rh),
                    'plume_bbox': (px, py, pw, ph),
                    'confidence': conf,
                    'source': f'hybrid_yolo_{cls_id}'
                })
                used_yolo_indices.add(best_idx)

        # Add the best YOLO-only body detection if not already included
        if best_body_bbox is not None:
            bx, by, bw, bh = best_body_bbox
            already_added = False
            for m in merged:
                mx, my, mw, mh = m['rocket_bbox']
                if abs((bx + bw/2) - (mx + mw/2)) < 10 and abs((by + bh/2) - (my + mh/2)) < 10:
                    already_added = True
                    break
            if not already_added:
                merged.append({
                    'rocket_bbox': (bx, by, bw, bh),
                    'plume_bbox': (bx + bw//2, by + bh//2, max(5, bw//4), max(5, bh//4)),
                    'confidence': best_body_conf,
                    'source': 'yolo_only_1'
                })

        return merged

    def detect_semantic_objects(self, frame, frame_idx, target_coords=None):
        """
        Generate semantic annotations for visualization.
        """
        annotations = []
        if target_coords is not None:
            cx, cy, w, h = target_coords
            rx, ry = int(cx - w/2), int(cy - h/2)
            annotations.append({
                'label': 'ROCKET BODY (Active Tracking)',
                'bbox': (rx, ry, w, h),
                'color': (0, 255, 0)
            })
            pw = w
            ph = int(h * 0.35)
            px = rx
            py = ry + h - ph
            annotations.append({
                'label': 'ENGINE PLUME / EXHAUST',
                'bbox': (px, py, pw, ph),
                'color': (0, 0, 255)
            })
        return annotations
