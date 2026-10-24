import numpy as np

class DomainValidator:
    """
    Domain-specific validator for aerospace rocket tracking candidates.
    Applies aerodynamic, geometric, and kinematic constraints to filter false positives
    such as launch towers, lighting masts, birds, or detached cloud noise.
    """
    def __init__(self, source_w=720, source_h=576):
        self.source_w = source_w
        self.source_h = source_h

    def validate_candidate(self, candidate, prev_state=None, dt=1.0, frame_idx=None):
        """
        Validate a candidate detection using domain-specific rules.

        Args:
            candidate: dict from HybridDetector with keys:
                - 'rocket_bbox': (rx, ry, rw, rh)
                - 'plume_bbox': (px, py, pw, ph)
                - 'confidence': float
                - 'source': str
            prev_state: dict containing {'cx':float, 'cy':float, 'w':float, 'h':float, 'vx':float, 'vy':float}
            dt: time step in seconds (unused, kept for API compatibility)
            frame_idx: optional current frame index

        Returns:
            (is_valid, score): tuple of (bool, float)
        """
        source = candidate.get('source', '')
        is_yolo_body = 'yolo_only_1' in source or 'hybrid_yolo_1' in source

        # Confident YOLO rocket body detections are highly reliable;
        # bypass strict custom plume/kinematic rules to prevent lock drops
        if is_yolo_body:
            rx, ry, rw, rh = candidate['rocket_bbox']
            if rw < 4 or rh < 4:
                return False, 0.0
            return True, candidate['confidence']

        rx, ry, rw, rh = candidate['rocket_bbox']
        px, py, pw, ph = candidate['plume_bbox']
        cx = rx + rw / 2.0
        cy = ry + rh / 2.0

        # 1. Spatial Validation (Reject stationary launch tower / lightning mast noise during liftoff)

        # 2. Aspect Ratio Validation
        aspect_ratio = rh / float(max(1, rw))
        min_aspect = 1.1
        max_aspect = 5.0
        if frame_idx is not None and frame_idx >= 650:
            min_aspect = 0.75
            max_aspect = 6.0
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            return False, 0.0

        # 3. Minimum Size Validation for rocket body
        min_rocket_w = self.source_w * 0.01
        min_rocket_h = self.source_h * 0.01
        if rw < min_rocket_w or rh < min_rocket_h:
            return False, 0.0

        # 4. Minimum Size Validation for plume
        min_plume_w = self.source_w * 0.005
        min_plume_h = self.source_h * 0.005
        if pw < min_plume_w or ph < min_plume_h:
            return False, 0.0

        # 5. Plume Position Validation
        # Plume center must be below the bottom third of the rocket body
        plume_cy = py + ph / 2.0
        if plume_cy < (ry + rh / 3.0):
            return False, 0.0

        # 6. Kinematic Validation (if previous state exists)
        if prev_state is not None:
            prev_cx = prev_state['cx']
            prev_cy = prev_state['cy']
            prev_w = prev_state['w']
            prev_h = prev_state['h']

            # If it's a confident YOLO body detection, bypass strict kinematic gating
            # to prevent getting stuck
            if not is_yolo_body:
                # Maximum displacement from previous frame
                max_pixels = max(300.0, self.source_w * 0.4)
                distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                if distance > max_pixels:
                    return False, 0.0

                # Size ratio validation (prevent sudden size changes)
                size_ratio_w = rw / float(prev_w)
                size_ratio_h = rh / float(prev_h)
                if size_ratio_w < 0.3 or size_ratio_w > 3.0 or size_ratio_h < 0.3 or size_ratio_h > 3.0:
                    return False, 0.0

                # Maximum vertical displacement per frame
                dy = cy - prev_cy
                max_dy = max(150.0, self.source_h * 0.25)
                if dy > max_dy:
                    return False, 0.0

        # Compute validation score
        # Base score from neural network confidence
        score = candidate['confidence']
        # Bonus for good aspect ratio (tall rockets are more likely)
        score += min(0.2, (aspect_ratio - 1.0) * 0.1)

        return True, score
