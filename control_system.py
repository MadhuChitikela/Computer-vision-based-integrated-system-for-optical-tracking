import cv2
import numpy as np

ZOOM_PRESETS = {
    "launch_pad_wide": {
        "target_height_viewport": 150.0,
        "k_zoom": 0.08,
        "zoom_min": 1.0,
        "zoom_max": 4.0,
        "base_focal_length_mm": 50.0
    },
    "tracking_telephoto": {
        "target_height_viewport": 240.0,
        "k_zoom": 0.10,
        "zoom_min": 1.0,
        "zoom_max": 8.0,
        "base_focal_length_mm": 50.0
    },
}

class GimbalSimulator:
    """
    Simulates a 2-axis motorized optical tracking mount with servo steering and
    dynamic focal-length zoom control to keep the target centered in the viewport.
    """
    def __init__(self, source_w=720, source_h=576, viewport_w=1280, viewport_h=720, preset="tracking_telephoto"):
        self.source_w = max(1, source_w)
        self.source_h = max(1, source_h)
        self.viewport_w = max(1, viewport_w)
        self.viewport_h = max(1, viewport_h)
        self.aspect_ratio = self.viewport_w / self.viewport_h

        self.current_preset = ZOOM_PRESETS.get(preset, ZOOM_PRESETS["tracking_telephoto"])

        self.x_g = float(self.source_w / 2)
        self.y_g = float(self.source_h / 2)
        self.base_focal_length = self.current_preset["base_focal_length_mm"]
        self.focal_length = self.base_focal_length
        self.target_focal_length = self.base_focal_length
        self.target_x = self.x_g
        self.target_y = self.y_g

        self.k_steer = 0.15
        self.k_zoom = self.current_preset["k_zoom"]
        self.target_height_viewport = self.current_preset["target_height_viewport"]
        # Convert zoom factors to focal lengths
        self.focal_min = self.current_preset["zoom_min"] * self.base_focal_length
        self.focal_max = self.current_preset["zoom_max"] * self.base_focal_length

    def update(self, target_coords=None):
        """
        Update gimbal position and focal length based on current target coordinates (cx, cy, w, h).
        """
        if target_coords is not None:
            cx, cy, w, h = target_coords
            self.target_x, self.target_y = cx, cy
            if h > 0:
                # Target focal length based on filling the viewport height
                # target_height_viewport / h gives a zoom factor. Multiply by base_focal_length to get target focal_length
                self.target_focal_length = np.clip(
                    (self.target_height_viewport / h) * self.base_focal_length,
                    self.focal_min,
                    self.focal_max
                )

        # Smooth steering towards target
        self.x_g += self.k_steer * (self.target_x - self.x_g)
        self.y_g += self.k_steer * (self.target_y - self.y_g)
        # Smooth zoom transition
        self.focal_length += self.k_zoom * (self.target_focal_length - self.focal_length)
        self.focal_length = np.clip(self.focal_length, self.focal_min, self.focal_max)

    def get_steering_offsets(self):
        """
        Return (offset_x, offset_y) from optical center.
        """
        return self.x_g - self.source_w / 2.0, -(self.y_g - self.source_h / 2.0)

    def get_crop_coords(self):
        """
        Determine crop bounding box (x1, y1, x2, y2) in source coordinates.
        """
        current_zoom_factor = self.focal_length / self.base_focal_length
        w_crop = self.source_w / current_zoom_factor
        h_crop = w_crop / self.aspect_ratio

        # Handle cases where height crop exceeds source height
        if h_crop > self.source_h:
            h_crop = self.source_h
            w_crop = h_crop * self.aspect_ratio

        # Center the crop on gimbal coordinates, then clamp to source frame boundaries
        x1 = int(np.clip(self.x_g - w_crop / 2.0, 0, self.source_w - w_crop))
        y1 = int(np.clip(self.y_g - h_crop / 2.0, 0, self.source_h - h_crop))
        x2 = int(x1 + w_crop)
        y2 = int(y1 + h_crop)

        return x1, y1, x2, y2

    def map_source_to_viewport(self, px, py):
        """
        Map a point (px, py) from source frame space into viewport window coordinates.
        """
        x1, y1, x2, y2 = self.get_crop_coords()
        w_crop = max(1, x2 - x1)
        h_crop = max(1, y2 - y1)
        vx = int(((px - x1) / w_crop) * self.viewport_w)
        vy = int(((py - y1) / h_crop) * self.viewport_h)
        return vx, vy

    def apply_zoom_to_frame(self, frame):
        """
        Crop and resize the frame according to current gimbal steer and focal length.
        """
        x1, y1, x2, y2 = self.get_crop_coords()
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return cv2.resize(frame, (self.viewport_w, self.viewport_h))
        return cv2.resize(cropped, (self.viewport_w, self.viewport_h), interpolation=cv2.INTER_LINEAR)

    def get_focal_length(self):
        return self.focal_length

    def get_target_focal_length(self):
        return self.target_focal_length
