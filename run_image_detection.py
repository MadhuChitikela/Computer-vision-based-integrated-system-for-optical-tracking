import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from detect import process_image


def main():
    parser = argparse.ArgumentParser(description="Run single-image rocket detection using HybridDetector")
    parser.add_argument("--image", "-i", type=str, help="Path to input image file")
    parser.add_argument("--model", "-m", type=str, default="rocket_yolo.pt", help="Path to YOLO weights")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image file path")
    parser.add_argument("--show", action="store_true", help="Display the annotated image using Matplotlib")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        # Check current directory for common image names
        candidates = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not f.startswith('detected_')]
        if candidates:
            image_path = candidates[0]
            print(f"ℹ️ No image specified; automatically using found image: '{image_path}'")
        else:
            print("❌ Error: No input image specified and no images found in current directory.")
            print("Usage: python run_image_detection.py --image <path_to_image> [--model rocket_yolo.pt] [--show]")
            sys.exit(1)

    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found!")
        sys.exit(1)

    output_path, process_time_ms = process_image(image_path, model_path_override=args.model, output_name=args.output)

    if output_path and os.path.exists(output_path):
        print(f"✅ Detection successful!")
        print(f"   Output file: {output_path}")
        print(f"   Inference time: {process_time_ms:.2f} ms")

        if args.show:
            annotated = cv2.imread(output_path)
            if annotated is not None:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10, 8))
                plt.imshow(annotated_rgb)
                plt.axis('off')
                plt.title(f"Detection Result: {output_path} ({process_time_ms:.1f} ms)")
                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    main()
