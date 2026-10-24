import os
import sys
import argparse
import time
import cv2
import matplotlib.pyplot as plt

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from detect import process_image
from rocket_tracking import RocketTrackingSystem


DEFAULT_MODELS = {
    "rocket_yolo.pt": "YOLOv8_Original",
    "rocket_yolo26.pt": "YOLO26",
    "rocket_yolo_v8n.pt": "YOLOv8n_Custom",
    "rocket_yolo_obb.pt": "YOLOv8_OBB",
    "yolov8n.pt": "YOLOv8n_Pretrained"
}

def compare_on_image(image_path, model_dict=None, output_fig="model_comparison_image.png", show=False):
    """
    Compare multiple YOLO models on a single image.
    Outputs side-by-side detection visual with bounding boxes, confidence, and latency.
    """
    if model_dict is None:
        model_dict = DEFAULT_MODELS

    available_models = {p: n for p, n in model_dict.items() if os.path.exists(p)}
    missing = [n for p, n in model_dict.items() if not os.path.exists(p)]

    if not available_models:
        print("⚠️ None of the custom model files were found locally. Using default 'yolov8n.pt'.")
        available_models = {"yolov8n.pt": "YOLOv8n_Pretrained"}
    else:
        if missing:
            print(f"ℹ️ Skipping absent model weights: {missing}")

    results = []
    print(f"\n🔍 Comparing {len(available_models)} model(s) on: {image_path}")

    for model_path, model_name in available_models.items():
        print(f"\n--- Running detection with {model_name} ({model_path}) ---")
        out_name, time_ms = process_image(image_path, model_path_override=model_path)
        if out_name and os.path.exists(out_name):
            annotated = cv2.imread(out_name)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            results.append((model_name, annotated_rgb, out_name, time_ms))

    if not results:
        print("❌ No detection results obtained.")
        return

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for i, (name, img_rgb, path, ms) in enumerate(results):
        axes[i].imshow(img_rgb)
        axes[i].axis('off')
        axes[i].set_title(f"{name}\n{os.path.basename(path)}\nLatency: {ms:.1f} ms", fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_fig, dpi=200, bbox_inches='tight')
    print(f"\n✅ Model image comparison saved to: {output_fig}")

    if show:
        plt.show()

def compare_on_video(video_path, model_dict=None):
    """
    Benchmark multiple tracking models on an input video.
    Measures tracking rate (%), final gimbal offsets, and generates comparison table.
    """
    if model_dict is None:
        model_dict = DEFAULT_MODELS

    available_models = {p: n for p, n in model_dict.items() if os.path.exists(p)}
    if not available_models:
        print("⚠️ No custom weights found; using 'yolov8n.pt'")
        available_models = {"yolov8n.pt": "YOLOv8n_Pretrained"}

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if w == 0 or h == 0:
        print(f"❌ Error: Unable to read video properties from '{video_path}'.")
        return

    benchmark_records = []

    for model_path, model_name in available_models.items():
        print("\n" + "=" * 60)
        print(f"🚀 Benchmarking Video Tracker with: {model_name} ({model_path})")
        print("=" * 60)

        tracker = RocketTrackingSystem(
            model_path=model_path,
            source_w=w,
            source_h=h,
            preset="tracking_telephoto"
        )
        out_name = f"tracked_compare_{model_name}.mp4"
        result_file = tracker.process_video(video_path, out_name)
        summary = tracker.get_summary()

        benchmark_records.append({
            'name': model_name,
            'file': result_file,
            'total_frames': summary['total_frames'],
            'tracked_rate': summary['tracking_rate'],
            'final_state': summary['final_tracker_state'],
            'final_azimuth': summary['final_azimuth'],
            'final_elevation': summary['final_elevation'],
            'final_focal': summary['final_focal_length']
        })

    print("\n" + "=" * 75)
    print("📊 BENCHMARK COMPARISON SUMMARY")
    print("=" * 75)
    header = f"{'Model':<22} {'Tracked %':>10} {'Final State':<16} {'Azi Offset':>12} {'Ele Offset':>12}"
    print(header)
    print("-" * 75)
    for r in benchmark_records:
        line = f"{r['name']:<22} {r['tracked_rate']*100:>9.1f}% {r['final_state']:<16} {r['final_azimuth']:>+11.1f}px {r['final_elevation']:>+11.1f}px"
        print(line)
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="Multi-model benchmark for image detection and video tracking")
    parser.add_argument("--image", "-i", type=str, help="Path to input image for model comparison")
    parser.add_argument("--video", "-v", type=str, help="Path to input video for model tracking comparison")
    parser.add_argument("--output", "-o", type=str, default="model_comparison_image.png", help="Output comparison figure path")
    parser.add_argument("--show", action="store_true", help="Display visual comparison plot")
    args = parser.parse_args()

    if not args.image and not args.video:
        print("❌ Error: Please specify either --image or --video to compare models.")
        print("Usage:")
        print("  python compare_models.py --image rocket.jpg")
        print("  python compare_models.py --video launch.mp4")
        return

    if args.image:
        compare_on_image(args.image, output_fig=args.output, show=args.show)
    elif args.video:
        compare_on_video(args.video)

if __name__ == "__main__":
    main()
