import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from detect import _enhance_frame_for_image_detection


def run_enhancement_demo(image_path=None, output_path="enhancement_demo_comparison.png", show=False):
    """
    Demonstrates the CLAHE + Denoise + Sharpen + Gamma enhancement pipeline.
    Plots and saves side-by-side comparison between raw frame and enhanced frame.
    """
    if not image_path:
        candidates = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not f.startswith(('detected_', 'enhancement_'))]
        if candidates:
            image_path = candidates[0]
            print(f"ℹ️ Auto-selected image for enhancement demo: '{image_path}'")
        else:
            print("❌ No input image specified and none found in directory.")
            print("Usage: python demo_enhancement.py --image <image_path>")
            return

    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' does not exist.")
        return

    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"❌ Error: Failed to decode image '{image_path}'.")
        return

    print("⚡ Applying aerospace enhancement pipeline:")
    print("   1. LAB Color Space Conversion")
    print("   2. CLAHE (Adaptive Histogram Equalization) on L-channel")
    print("   3. Mild Denoising (fastNlMeansDenoisingColored)")
    print("   4. 2D Convolutional Sharpening Kernel")
    print("   5. Gamma Correction LUT (0.85) for haze/smoke penetration")

    enhanced_img = _enhance_frame_for_image_detection(original_img)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original Image
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original Image\n{os.path.basename(image_path)} ({original_img.shape[1]}x{original_img.shape[0]})', fontsize=12)
    axes[0].axis('off')

    # Enhanced Image
    axes[1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Enhanced Image (CLAHE + Denoise + Sharpen + Gamma 0.85)', fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Comparison figure saved to: {output_path}")

    if show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Demonstrate frame enhancement for smoke and haze penetration")
    parser.add_argument("--image", "-i", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, default="enhancement_demo_comparison.png", help="Path to save comparison plot")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    run_enhancement_demo(args.image, args.output, args.show)

if __name__ == "__main__":
    main()
