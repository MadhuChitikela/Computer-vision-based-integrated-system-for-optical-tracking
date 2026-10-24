import cv2
import numpy as np

class RocketTracker:
    """
    Robust optical rocket tracker with a 6-state Kalman filter, Lucas-Kanade optical flow
    feature tracking during occlusions, shot-cut scene change detection, and body-only target gating.
    """
    def __init__(self, max_occlusion_frames=60, source_w=720, source_h=576):
        self.source_w = source_w
        self.source_h = source_h
        self.max_occlusion_frames = max_occlusion_frames
        self.state = 'LOST'
        self.occlusion_counter = 0
        self.kf = None
        self.last_state_dict = None
        self.prev_gray = None
        self.flow_features = None
        self.max_features = 40
        self.prev_hist = None
        self.shot_cut_threshold = 0.55

    def reset(self):
        self.state = 'LOST'
        self.occlusion_counter = 0
        self.kf = None
        self.last_state_dict = None
        self.prev_gray = None
        self.flow_features = None

    def _init_kalman(self, cx, cy, w, h):
        """
        State: [cx, cy, vx, vy, w, h]^T
        Measurement: [cx, cy, w, h]^T
        """
        kf = cv2.KalmanFilter(6, 4)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        kf.processNoiseCov[2, 2] = 0.5
        kf.processNoiseCov[3, 3] = 0.5
        kf.processNoiseCov[4, 4] = 1e-1
        kf.processNoiseCov[5, 5] = 1e-1
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        kf.errorCovPost = np.eye(6, dtype=np.float32) * 10.0
        kf.statePost = np.array([[cx], [cy], [0.0], [-2.0], [w], [h]], dtype=np.float32)
        self.kf = kf

    def detect_shot_cut(self, frame):
        """
        Detect abrupt camera switches or scene shot-cuts via HSV histogram correlation.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        is_cut = False
        if self.prev_hist is not None:
            corr = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
            distance = 1.0 - corr
            if distance > self.shot_cut_threshold:
                is_cut = True
        self.prev_hist = hist
        return is_cut

    def step(self, frame, validated_candidates):
        """
        Process a single frame and update tracker state.

        Args:
            frame: current BGR frame
            validated_candidates: list of candidate detections from DomainValidator

        Returns:
            (cx, cy, w, h) of target if tracked, else None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detect_shot_cut(frame):
            self.reset()
            self.prev_gray = gray
            return None

        pred_bbox = None
        if self.kf is not None:
            pred = self.kf.predict()
            pred_cx, pred_cy, pred_vx, pred_vy, pred_w, pred_h = pred.flatten()
            pred_bbox = (int(pred_cx - pred_w / 2.0), int(pred_cy - pred_h / 2.0), int(pred_w), int(pred_h))
            self.last_state_dict = {
                'cx': pred_cx, 'cy': pred_cy,
                'vx': pred_vx, 'vy': pred_vy,
                'w': pred_w, 'h': pred_h
            }

        # ONLY the rocket BODY may become/remain the tracked target.
        # Flame/plume candidates are still detected + drawn for visualization,
        # but they must never be selected as the tracked centroid.
        def _is_body(c):
            source = c.get('source', '')
            return 'yolo_only_1' in source or 'hybrid_yolo_1' in source

        best_candidate = None
        if len(validated_candidates) > 0:
            body_candidates = [c for c in validated_candidates if _is_body(c)]

            if self.state == 'LOST':
                # Only allow a body detection to initiate a lock. Never lock onto a flame.
                if body_candidates:
                    best_candidate = max(body_candidates, key=lambda c: c['validation_score'])
            else:
                pred_cx = self.last_state_dict['cx']
                pred_cy = self.last_state_dict['cy']
                best_dist = float('inf')
                # Scale max distance search region based on frame width (40% of width)
                max_dist = max(400.0, self.source_w * 0.4)
                for c in body_candidates:
                    rx, ry, rw, rh = c['rocket_bbox']
                    cx, cy = rx + rw / 2.0, ry + rh / 2.0
                    dist = np.sqrt((cx - pred_cx)**2 + (cy - pred_cy)**2)
                    if dist < best_dist and dist < max_dist:
                        best_dist = dist
                        best_candidate = c

                # Fallback recovery: if no BODY candidate was within max_dist, grab the
                # highest-confidence body detection available
                if best_candidate is None and body_candidates:
                    best_candidate = max(body_candidates, key=lambda c: c['confidence'])

        if best_candidate is not None:
            rx, ry, rw, rh = best_candidate['rocket_bbox']
            cx, cy = rx + rw / 2.0, ry + rh / 2.0
            if self.kf is None:
                self._init_kalman(cx, cy, rw, rh)
            else:
                measurement = np.array([[cx], [cy], [rw], [rh]], dtype=np.float32)
                self.kf.correct(measurement)
            self.state = 'LOCKED'
            self.occlusion_counter = 0
            self._refresh_flow_features(gray, rx, ry, rw, rh)
        else:
            if self.state in ['LOCKED', 'OCCLUDED_FLOW', 'OCCLUDED_KALMAN']:
                self.occlusion_counter += 1
                if self.occlusion_counter > self.max_occlusion_frames:
                    self.reset()
                else:
                    flow_success = False
                    if self.prev_gray is not None and self.flow_features is not None and len(self.flow_features) > 0:
                        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                            self.prev_gray, gray, self.flow_features, None,
                            winSize=(15, 15), maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                        )
                        good_prev = self.flow_features[status == 1]
                        good_next = next_pts[status == 1]
                        if len(good_next) >= 4:
                            displacements = good_next - good_prev
                            dx = np.median(displacements[:, 0])
                            dy = np.median(displacements[:, 1])
                            if self.kf is not None:
                                state = self.kf.statePost
                                state[0, 0] += dx
                                state[1, 0] += dy
                                measurement = np.array([[state[0, 0]], [state[1, 0]], [state[4, 0]], [state[5, 0]]], dtype=np.float32)
                                self.kf.correct(measurement)
                            self.state = 'OCCLUDED_FLOW'
                            self.flow_features = good_next.reshape(-1, 1, 2)
                            flow_success = True
                    if not flow_success:
                        self.state = 'OCCLUDED_KALMAN'
                        if self.kf is not None:
                            self.kf.statePost = self.kf.statePre
                        self.flow_features = None
            else:
                self.state = 'LOST'

        self.prev_gray = gray

        if self.kf is not None:
            state = self.kf.statePost
            cx, cy, vx, vy, w, h = state.flatten()
            self.last_state_dict = {'cx': cx, 'cy': cy, 'vx': vx, 'vy': vy, 'w': w, 'h': h}
            return (int(cx), int(cy), int(w), int(h))
        else:
            return None

    def _refresh_flow_features(self, gray_frame, rx, ry, rw, rh):
        """
        Detect good features within the rocket body ROI for optical flow tracking.
        """
        roi_h = int(rh * 0.65)
        roi_y = ry
        roi_x = rx
        roi_w = rw
        roi_y = max(0, roi_y)
        roi_x = max(0, roi_x)
        roi_h = min(self.source_h - roi_y, roi_h)
        roi_w = min(self.source_w - roi_x, roi_w)
        if roi_h <= 5 or roi_w <= 5:
            self.flow_features = None
            return
        roi = gray_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        corners = cv2.goodFeaturesToTrack(roi, maxCorners=self.max_features, qualityLevel=0.05, minDistance=3)
        if corners is not None:
            corners[:, 0, 0] += roi_x
            corners[:, 0, 1] += roi_y
            self.flow_features = corners.astype(np.float32)
        else:
            self.flow_features = None

    def get_predicted_bbox(self):
        if self.last_state_dict is not None:
            cx = self.last_state_dict['cx']
            cy = self.last_state_dict['cy']
            w = self.last_state_dict['w']
            h = self.last_state_dict['h']
            rx = int(cx - w / 2.0)
            ry = int(cy - h / 2.0)
            return (rx, ry, int(w), int(h))
        return None
