"""
Optical Rocket Tracking System using Kalman Filter, YOLO, and SAHI

Authors:
    - Madhu Chitikela
    - Prasanthi Reddy
    - Pudari Sathwik
    - Sanjana Reddy
"""

__author__ = "Madhu Chitikela, Prasanthi Reddy, Pudari Sathwik, Sanjana Reddy"
__version__ = "1.0.0"



from domain_validator import DomainValidator
from control_system import GimbalSimulator, ZOOM_PRESETS
from hybrid_detector import HybridDetector
from tracker import RocketTracker
from rocket_tracking import RocketTrackingSystem, TrackingTelemetry
from detect import process_image, process_video, _enhance_frame_for_image_detection

__all__ = [
    "DomainValidator",
    "GimbalSimulator",
    "ZOOM_PRESETS",
    "HybridDetector",
    "RocketTracker",
    "RocketTrackingSystem",
    "TrackingTelemetry",
    "process_image",
    "process_video",
    "_enhance_frame_for_image_detection",
]
