import os
import sys
import time
import argparse
from collections import deque
import numpy as np
import cv2
import matplotlib.pyplot as plt

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def plot_kalman_vs_deepsort(kalman_data=None, deepsort_data=None, output_path='kalman_vs_deepsort_FIXED_COLORS.png', show=False):
    """
    Plot Delta X and Delta Y frame-to-frame displacement comparison:
    Kalman Filter (Bright Green) vs DeepSORT (Bright Red).
    Computes standard deviation to quantify tracking jitter.
    """
    if kalman_data is not None and deepsort_data is not None:
        try:
            k_frames = kalman_data["frames"][1:]
            k_dx = kalman_data["dx"][1:]
            k_dy = kalman_data["dy"][1:]
            ds_frames = deepsort_data["frames"][1:]
            ds_dx = deepsort_data["dx"][1:]
            ds_dy = deepsort_data["dy"][1:]
            print("Using provided tracking telemetry data.")
        except Exception:
            kalman_data = None

    if kalman_data is None:
        print("Using simulated rocket trajectory data for displacement analysis.")
        np.random.seed(42)
        frames = np.arange(1, 200)

        # Kalman: smooth, continuous, small deltas
        k_dx = np.random.normal(0, 1.5, len(frames))
        k_dy = np.random.normal(-2.5, 2.0, len(frames))

        # DeepSORT: more jitter, tracking identity switches and spikes
        ds_dx = np.random.normal(0, 8.0, len(frames))
        ds_dy = np.random.normal(-2.5, 12.0, len(frames))
        ds_dx[20] = -120
        ds_dx[80] = 90
        ds_dy[30] = 250
        ds_dy[90] = -180

        k_frames = frames
        ds_frames = frames

    KALMAN_COLOR = '#00AA00'      # Bright GREEN
    KALMAN_FILL = '#90EE90'       # Light green fill
    DEEPSORT_COLOR = '#CC0000'    # Bright RED
    DEEPSORT_FILL = '#FFB6C1'     # Light red fill

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 1: DELTA X ---
    ax1.plot(ds_frames, ds_dx, color=DEEPSORT_COLOR, linewidth=1.5,
             label='DeepSORT', alpha=0.8, zorder=2)
    ax1.plot(k_frames, k_dx, color=KALMAN_COLOR, linewidth=2.5,
             label='Kalman Filter', alpha=1.0, zorder=3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)

    ax1.fill_between(ds_frames, ds_dx, alpha=0.15, color=DEEPSORT_FILL, zorder=1)
    ax1.fill_between(k_frames, k_dx, alpha=0.3, color=KALMAN_FILL, zorder=2)

    ax1.set_title('Delta X: Horizontal Movement', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Delta X (pixels)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    k_dx_std = float(np.std(k_dx))
    ds_dx_std = float(np.std(ds_dx))
    ax1.text(0.02, 0.98,
             f'Kalman Std: {k_dx_std:.2f}px\nDeepSORT Std: {ds_dx_std:.2f}px',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # --- PLOT 2: DELTA Y ---
    ax2.plot(ds_frames, ds_dy, color=DEEPSORT_COLOR, linewidth=1.5,
             label='DeepSORT', alpha=0.8, zorder=2)
    ax2.plot(k_frames, k_dy, color=KALMAN_COLOR, linewidth=2.5,
             label='Kalman Filter', alpha=1.0, zorder=3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)

    ax2.fill_between(ds_frames, ds_dy, alpha=0.15, color=DEEPSORT_FILL, zorder=1)
    ax2.fill_between(k_frames, k_dy, alpha=0.3, color=KALMAN_FILL, zorder=2)

    ax2.set_title('Delta Y: Vertical Movement', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('Delta Y (pixels)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    k_dy_std = float(np.std(k_dy))
    ds_dy_std = float(np.std(ds_dy))
    ax2.text(0.02, 0.98,
             f'Kalman Std: {k_dy_std:.2f}px\nDeepSORT Std: {ds_dy_std:.2f}px',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    plt.suptitle('Kalman Filter (GREEN) vs DeepSORT (RED): Frame-to-Frame Displacement',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Displacement comparison figure saved to: {output_path}")

    print("\n" + "="*60)
    print("COLOR SCHEME & ANALYSIS:")
    print("="*60)
    print("GREEN  = Kalman Filter (smooth, stable, low jitter)")
    print("RED    = DeepSORT (jittery, displacement spikes during fast ascent)")
    print(f"Kalman Std Dev:   X = {k_dx_std:.2f} px, Y = {k_dy_std:.2f} px")
    print(f"DeepSORT Std Dev: X = {ds_dx_std:.2f} px, Y = {ds_dy_std:.2f} px")
    print("="*60)

    if show:
        plt.show()

def compare_filters_on_video(video_path, model_path="rocket_yolo.pt"):
    """
    Run 3-way tracking comparison on video:
      Method A: Kalman Filter (Full state + covariance)
      Method B: Moving Average (Sliding window deque)
      Method C: Alpha-Beta Filter (Fixed-gain predictor)
    """
    from hybrid_detector import HybridDetector
    from domain_validator import DomainValidator
    from rocket_tracking import RocketTrackingSystem

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\n📊 Input Video: {w}x{h}, {fps:.2f} FPS, {total_frames} frames")

    # ============================================================
    # METHOD A: KALMAN FILTER
    # ============================================================
    print("\n" + "="*60)
    print("🎬 A) KALMAN FILTER — Full RocketTrackingSystem")
    print("="*60)
    kalman_tracker = RocketTrackingSystem(
        model_path=model_path,
        source_w=w,
        source_h=h,
        preset="tracking_telephoto"
    )
    out_kalman = "compare_kalman.mp4"
    kalman_tracker.process_video(video_path, out_kalman)
    print(f"✅ Kalman saved: {out_kalman}")

    # ============================================================
    # METHOD B: MOVING AVERAGE
    # ============================================================
    print("\n" + "="*60)
    print("🎬 B) MOVING AVERAGE — Simple smoothing, no prediction")
    print("="*60)
    detector = HybridDetector(model_path=model_path, source_w=w, source_h=h)
    validator = DomainValidator(source_w=w, source_h=h)
    out_ma = "compare_moving_avg.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_ma = cv2.VideoWriter(out_ma, fourcc, fps, (w, h))

    buffer_size = 5
    ma_buffer = deque(maxlen=buffer_size)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        display_frame = frame.copy()

        candidates = detector.detect(frame, use_sahi=False, strict_mode=False)
        if len(candidates) == 0:
            candidates = detector.detect(frame, use_sahi=True, strict_mode=False)

        validated = []
        for c in candidates:
            is_v, score = validator.validate_candidate(c)
            if is_v:
                c['validation_score'] = score
                validated.append(c)

        best = None
        for c in validated:
            src = c.get('source', '')
            if 'yolo_only_1' in src or 'hybrid_yolo_1' in src:
                if best is None or c['confidence'] > best['confidence']:
                    best = c

        if best:
            rx, ry, rw, rh = best['rocket_bbox']
            cx, cy = rx + rw/2, ry + rh/2
            ma_buffer.append((cx, cy, rw, rh))

        if len(ma_buffer) > 0:
            avg_cx = np.mean([b[0] for b in ma_buffer])
            avg_cy = np.mean([b[1] for b in ma_buffer])
            avg_w = np.mean([b[2] for b in ma_buffer])
            avg_h = np.mean([b[3] for b in ma_buffer])

            rx = int(avg_cx - avg_w/2)
            ry = int(avg_cy - avg_h/2)

            cv2.rectangle(display_frame, (rx, ry), (int(rx+avg_w), int(ry+avg_h)), (255, 165, 0), 3)
            cv2.putText(display_frame, f"Moving Avg ({len(ma_buffer)}/{buffer_size})",
                        (rx, max(25, ry-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        else:
            cv2.putText(display_frame, "LOST — No buffer", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(display_frame, (10, 10), (320, 50), (0, 0, 0), -1)
        cv2.putText(display_frame, "⚠️ MOVING AVERAGE FILTER", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        writer_ma.write(display_frame)
        if frame_idx % 50 == 0:
            print(f"Moving Average frame {frame_idx}/{total_frames}...", end='\r', flush=True)

    cap.release()
    writer_ma.release()
    print(f"\n✅ Moving Average saved: {out_ma}")

    # ============================================================
    # METHOD C: ALPHA-BETA FILTER
    # ============================================================
    print("\n" + "="*60)
    print("🎬 C) ALPHA-BETA FILTER — Fixed gains, no covariance")
    print("="*60)
    out_ab = "compare_alpha_beta.mp4"
    writer_ab = cv2.VideoWriter(out_ab, fourcc, fps, (w, h))

    ab_x, ab_y = float(w / 2), float(h / 2)
    ab_vx, ab_vy = 0.0, 0.0
    ab_w, ab_h = 50.0, 100.0
    alpha, beta = 0.4, 0.1

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        display_frame = frame.copy()

        candidates = detector.detect(frame, use_sahi=False, strict_mode=False)
        if len(candidates) == 0:
            candidates = detector.detect(frame, use_sahi=True, strict_mode=False)

        validated = []
        for c in candidates:
            is_v, score = validator.validate_candidate(c)
            if is_v:
                c['validation_score'] = score
                validated.append(c)

        best = None
        for c in validated:
            src = c.get('source', '')
            if 'yolo_only_1' in src or 'hybrid_yolo_1' in src:
                if best is None or c['confidence'] > best['confidence']:
                    best = c

        # Predict step
        ab_x += ab_vx
        ab_y += ab_vy

        if best:
            rx, ry, rw, rh = best['rocket_bbox']
            meas_x = rx + rw/2.0
            meas_y = ry + rh/2.0

            # Update step
            residual_x = meas_x - ab_x
            residual_y = meas_y - ab_y

            ab_x += alpha * residual_x
            ab_y += alpha * residual_y
            ab_vx += beta * residual_x
            ab_vy += beta * residual_y
            ab_w = ab_w + alpha * (rw - ab_w)
            ab_h = ab_h + alpha * (rh - ab_h)

            status = "TRACKED"
            color = (0, 255, 255)
        else:
            status = "PREDICTING (No detection)"
            color = (0, 165, 255)

        rx = int(ab_x - ab_w/2.0)
        ry = int(ab_y - ab_h/2.0)
        cv2.rectangle(display_frame, (rx, ry), (int(rx+ab_w), int(ry+ab_h)), color, 3)
        cv2.putText(display_frame, f"Alpha-Beta: {status}",
                    (rx, max(25, ry-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.rectangle(display_frame, (10, 10), (320, 50), (0, 0, 0), -1)
        cv2.putText(display_frame, "⚠️ ALPHA-BETA FILTER", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        writer_ab.write(display_frame)
        if frame_idx % 50 == 0:
            print(f"Alpha-Beta frame {frame_idx}/{total_frames}...", end='\r', flush=True)

    cap.release()
    writer_ab.release()
    print(f"\n✅ Alpha-Beta saved: {out_ab}")

    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY TABLE")
    print("="*70)
    print(f"{'Feature':<28} {'Kalman Filter':<15} {'Moving Avg':<15} {'Alpha-Beta':<15}")
    print("-" * 70)
    print(f"{'Predicts during occlusion':<28} {'✅ Yes':<15} {'❌ No':<15} {'✅ Yes':<15}")
    print(f"{'Auto-tunes to noise':<28} {'✅ Yes':<15} {'❌ No':<15} {'❌ Fixed gains':<15}")
    print(f"{'Handles fast acceleration':<28} {'✅ Yes':<15} {'⚠️ Lags':<15} {'⚠️ Slow response':<15}")
    print(f"{'Tracks velocity':<28} {'✅ Yes':<15} {'❌ No':<15} {'✅ Yes':<15}")
    print(f"{'Complexity':<28} {'Medium':<15} {'Simple':<15} {'Simple':<15}")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Tracking filter benchmarking (Kalman vs DeepSORT vs MA vs Alpha-Beta)")
    parser.add_argument("--video", "-v", type=str, help="Video path to compare Kalman vs Moving Average vs Alpha-Beta")
    parser.add_argument("--model", "-m", type=str, default="rocket_yolo.pt", help="Path to YOLO model")
    parser.add_argument("--plot", action="store_true", help="Generate Kalman vs DeepSORT displacement stability plot")
    parser.add_argument("--output", "-o", type=str, default="kalman_vs_deepsort_FIXED_COLORS.png", help="Path for displacement plot")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    if args.video:
        compare_filters_on_video(args.video, model_path=args.model)

    if args.plot or not args.video:
        plot_kalman_vs_deepsort(output_path=args.output, show=args.show)

if __name__ == "__main__":
    main()
