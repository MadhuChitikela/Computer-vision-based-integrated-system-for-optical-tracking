import os
import sys
import argparse
import cv2

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rocket_tracking import RocketTrackingSystem


def main():
    parser = argparse.ArgumentParser(description="Run complete optical rocket tracking on a video")
    parser.add_argument("--video", "-v", type=str, help="Path to input video file")
    parser.add_argument("--model", "-m", type=str, default="rocket_yolo.pt", help="Path to YOLO weights")
    parser.add_argument("--preset", "-p", type=str, default="tracking_telephoto", choices=["launch_pad_wide", "tracking_telephoto"], help="Gimbal zoom preset")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output video file path")
    parser.add_argument("--telemetry", "-t", type=str, default=None, help="Output path for telemetry JSON export")
    args = parser.parse_args()

    video_file = args.video
    if not video_file:
        candidates = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm')) and not f.startswith(('tracked_', 'zoomed_', 'compare_'))]
        if candidates:
            video_file = candidates[0]
            print(f"ℹ️ Auto-selected video: '{video_file}'")
        else:
            print("❌ Error: No video file specified and none found in workspace.")
            print("Usage: python run_video_tracking.py --video <video_path> [--model rocket_yolo.pt]")
            sys.exit(1)

    if not os.path.exists(video_file):
        print(f"❌ Error: Video '{video_file}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(video_file)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if w == 0 or h == 0:
        print(f"❌ Error: Could not determine video dimensions. File may be corrupt.")
        sys.exit(1)

    print(f"🚀 Initializing RocketTrackingSystem: {w}x{h} source, {fps:.2f} FPS, {total_frames} frames")
    tracker = RocketTrackingSystem(
        model_path=args.model,
        source_w=w,
        source_h=h,
        preset=args.preset
    )

    output = tracker.process_video(video_file, output_name=args.output)

    if output and os.path.exists(output):
        print(f"\n✅ Tracking complete! Output saved to: {output}")
        summary = tracker.get_summary()
        print("\n📊 Performance Telemetry Summary:")
        print(f"   Total Frames:       {summary['total_frames']}")
        print(f"   Tracked Frames:     {summary['tracked_frames']}")
        print(f"   Lost Frames:        {summary['lost_frames']}")
        print(f"   Tracking Rate:      {summary['tracking_rate']*100:.1f}%")
        print(f"   Final Tracker State: {summary['final_tracker_state']}")
        print(f"   Final Focal Length:  {summary['final_focal_length']:.1f} mm")
        print(f"   Final Azimuth Offset:   {summary['final_residual_azimuth']:+.2f} px")
        print(f"   Final Elevation Offset: {summary['final_residual_elevation']:+.2f} px")

        if args.telemetry:
            tracker.export_telemetry(args.telemetry)
        else:
            default_telemetry_name = f"telemetry_{os.path.splitext(os.path.basename(video_file))[0]}.json"
            tracker.export_telemetry(default_telemetry_name)

if __name__ == "__main__":
    main()
