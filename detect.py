import cv2
import os
import sys
import shutil
import subprocess
import threading
import time
import traceback
import argparse
import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from hybrid_detector import HybridDetector
from domain_validator import DomainValidator


def _enhance_frame_for_image_detection(frame: np.ndarray) -> np.ndarray:
    """
    Enhance frame clarity before detection.
    Helps YOLO detect rocket body through smoke, haze, and low light.
    Applies CLAHE on the L-channel, denoising, sharpening, and gamma correction.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE on L channel (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Mild denoising to reduce grain from enhancement
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)

    # Slight sharpening
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Gamma correction for haze/smoke penetration
    gamma = 0.85  # < 1 = brighten shadows
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)

    return enhanced

def process_image(image_path, model_path_override=None, output_name=None):
    """
    Run single-image rocket detection using enhancement, HybridDetector, and DomainValidator.
    Returns (output_path, process_time_ms).
    """
    t_start = time.time()
    if output_name is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if model_path_override:
            model_suffix = os.path.splitext(os.path.basename(model_path_override))[0]
            output_name = f"detected_{base_name}_{model_suffix}.png"
        else:
            output_name = f"detected_{base_name}.png"
    else:
        root, ext = os.path.splitext(output_name)
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            output_name = root + ".png"

    model_to_use = model_path_override if model_path_override else "rocket_yolo.pt"
    print(f"📸 Image mode: Detecting rocket in '{image_path}' using '{model_to_use}'...")
    detector = HybridDetector(model_path=model_to_use)
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image '{image_path}'!")
        return None, 0.0

    # Apply image enhancement
    enhanced_img = _enhance_frame_for_image_detection(img)

    # Initialize DomainValidator for single image processing
    validator = DomainValidator(source_w=img.shape[1], source_h=img.shape[0])

    candidates = detector.detect(enhanced_img, use_sahi=False, strict_mode=True)

    # Validate and score candidates
    validated_candidates = []
    for cand in candidates:
        is_valid, score = validator.validate_candidate(cand)
        if is_valid:
            cand['validation_score'] = score
            validated_candidates.append(cand)

    img_h, img_w = img.shape[:2]
    min_box_h = img_h * 0.01
    validated_candidates = [c for c in validated_candidates if c['rocket_bbox'][3] >= min_box_h]

    # Sort candidates: prioritize by validation_score, then rocket body, then confidence
    def sort_key(c):
        is_rocket_body = 'yolo_only_1' in c['source'] or 'hybrid_yolo_1' in c['source']
        return (c.get('validation_score', c['confidence']), is_rocket_body, c['confidence'])

    validated_candidates = sorted(validated_candidates, key=sort_key, reverse=True)[:5]

    annotated_img = img.copy()
    print(f"Found {len(validated_candidates)} targets.")
    best_overall_rocket_body_confidence = 0.0

    for c in validated_candidates:
        rx, ry, rw, rh = c['rocket_bbox']
        conf = c['confidence']
        source = c['source']

        if ('yolo_only_1' in source or 'hybrid_yolo_1' in source) and conf > best_overall_rocket_body_confidence:
            best_overall_rocket_body_confidence = conf

        if 'yolo_only_1' in source or 'hybrid_yolo_1' in source:
            box_color = (0, 255, 0)
            box_label = "Rocket Body"
        elif 'hybrid_yolo_0' in source or 'flame_only_unmatched' in source:
            box_color = (0, 0, 255)
            box_label = "Engine Flames"
        else:
            box_color = (255, 255, 0)
            box_label = "Unknown Target"
        cv2.rectangle(annotated_img, (rx, ry), (rx+rw, ry+rh), box_color, 2)
        cv2.putText(annotated_img, f"{box_label} ({conf:.2f})", (rx, max(25, ry - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA)
        print(f"📍 Detected: {box_label} BBox {(rx, ry, rw, rh)}, Source: {source}, Conf: {conf:.2f}")

    if validated_candidates:
        text = f"Overall Confidence (Rocket Body): {best_overall_rocket_body_confidence:.2f}"
        fixed_font_scale = 0.8
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fixed_font_scale, 2)
        text_x = img_w - text_size[0] - 20
        text_y = text_size[1] + 20
        cv2.putText(annotated_img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fixed_font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_name, annotated_img)
    t_end = time.time()
    process_time_ms = (t_end - t_start) * 1000
    print(f"🎉 Detection complete with {os.path.basename(model_to_use)}! Annotated image saved as: {output_name}. Processing time: {process_time_ms:.2f} ms")
    return output_name, process_time_ms

def _drain_stream(stream, sink):
    try:
        for line in iter(stream.readline, b""):
            sink.append(line.decode(errors="replace"))
    except Exception:
        pass

def process_video(video_path, model_path_override=None, output_name=None):
    """
    Run video detection and output tracked/detected video.
    Uses ffmpeg pipeline if ffmpeg is available, otherwise falls back to OpenCV VideoWriter.
    """
    if output_name is None:
        output_name = f"tracked_{os.path.splitext(os.path.basename(video_path))[0]}.webm"
    model_to_use = model_path_override if model_path_override else "rocket_yolo.pt"
    print(f"🎥 Video mode: Processing tracking on '{video_path}' using '{model_to_use}'...")

    has_ffmpeg = shutil.which("ffmpeg") is not None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if not fps_video or fps_video != fps_video or fps_video <= 0:
        print(f"⚠️ Video reported invalid FPS ({fps_video}), defaulting to 25.")
        fps_video = 25.0
    if total_frames <= 0:
        print("❌ Error: Could not read video frames!")
        cap.release()
        return None

    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame to determine video dimensions!")
        cap.release()
        return None

    frame_h, frame_w, _ = frame.shape
    print(f"ℹ️ Video properties -- W: {frame_w}, H: {frame_h}, FPS: {fps_video:.2f}, Total Frames: {total_frames}")
    detector = HybridDetector(model_path=model_to_use, source_w=frame_w, source_h=frame_h)
    out_w, out_h = frame_w, frame_h

    # If ffmpeg is not available, switch extension to mp4 and use cv2.VideoWriter
    if not has_ffmpeg and output_name.endswith('.webm'):
        output_name = os.path.splitext(output_name)[0] + ".mp4"
        print("ℹ️ ffmpeg not found in PATH; falling back to OpenCV VideoWriter (.mp4).")

    ffmpeg_process = None
    cv_writer = None
    stderr_thread = None
    ffmpeg_stderr_lines = []

    if has_ffmpeg and output_name.endswith('.webm'):
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{out_w}x{out_h}',
            '-r', str(int(fps_video)), '-i', '-',
            '-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0',
            '-row-mt', '1', '-deadline', 'good', '-cpu-used', '5',
            '-pix_fmt', 'yuv420p', output_name
        ]
        try:
            ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            stderr_thread = threading.Thread(target=_drain_stream, args=(ffmpeg_process.stderr, ffmpeg_stderr_lines), daemon=True)
            stderr_thread.start()
        except Exception as e:
            print(f"⚠️ ffmpeg start failed ({e}), falling back to OpenCV VideoWriter.")
            ffmpeg_process = None

    if ffmpeg_process is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cv_writer = cv2.VideoWriter(output_name, fourcc, fps_video, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    error_count = 0
    max_errors_to_print = 5
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))
            display_frame = frame.copy()
            try:
                candidates = detector.detect(frame, use_sahi=False, strict_mode=False)
                if len(candidates) == 0:
                    candidates = detector.detect(frame, use_sahi=True, strict_mode=False)
                for c in candidates:
                    rx, ry, rw, rh = c['rocket_bbox']
                    conf = c['confidence']
                    source = c['source']
                    if 'yolo_only_1' in source or 'hybrid_yolo_1' in source:
                        box_color = (0, 255, 0)
                        box_label = "Rocket Body"
                    elif 'hybrid_yolo_0' in source or 'flame_only_unmatched' in source:
                        box_color = (0, 0, 255)
                        box_label = "Engine Flames"
                    else:
                        box_color = (255, 255, 0)
                        box_label = "Unknown Target"
                    cv2.rectangle(display_frame, (rx, ry), (rx+rw, ry+rh), box_color, 2)
                    cv2.putText(display_frame, f"{box_label} ({conf:.2f})", (rx, max(25, ry - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)
            except Exception as e:
                error_count += 1
                if error_count <= max_errors_to_print:
                    print(f"⚠️ Detection failed on frame {frame_idx}/{total_frames}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                elif error_count == max_errors_to_print + 1:
                    print(f"⚠️ Suppressing further per-frame error prints...")

            if ffmpeg_process is not None:
                try:
                    ffmpeg_process.stdin.write(display_frame.tobytes())
                except Exception as e:
                    print(f"❌ Fatal: ffmpeg stdin write failed at frame {frame_idx}: {e}")
                    break
            elif cv_writer is not None:
                cv_writer.write(display_frame)

            if frame_idx % 25 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_time
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                print(f"Frame {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%) -- {proc_fps:.1f} fps, {elapsed:.0f}s elapsed with {os.path.basename(model_to_use)}", flush=True)
    finally:
        cap.release()
        if ffmpeg_process is not None:
            print(f"ℹ️ All {frame_idx} frames sent to ffmpeg stdin. ({error_count} frames had errors.) Encoding...")
            try:
                ffmpeg_process.stdin.close()
            except Exception:
                pass
            ffmpeg_process.wait()
            if stderr_thread:
                stderr_thread.join(timeout=5)
            if ffmpeg_process.returncode != 0:
                print(f"❌ Error: ffmpeg exited with code {ffmpeg_process.returncode}")
                if ffmpeg_stderr_lines:
                    print("---- ffmpeg error output (tail) ----")
                    print("".join(ffmpeg_stderr_lines[-40:]))
                return None
        elif cv_writer is not None:
            cv_writer.release()

    if os.path.exists(output_name) and os.path.getsize(output_name) > 0:
        size_kb = os.path.getsize(output_name) / 1024
        print(f"✅ Successfully created '{output_name}' ({size_kb:.1f} KB)")
    else:
        print(f"❌ Error: Process reported success, but '{output_name}' is missing or empty.")
        return None

    print(f"🎉 Processing complete with {os.path.basename(model_to_use)}! Output: {output_name}")
    return output_name

def main():
    parser = argparse.ArgumentParser(description="Rocket Detection on Images and Videos")
    parser.add_argument("input", nargs="?", help="Path to input image or video")
    parser.add_argument("--model", "-m", default="rocket_yolo.pt", help="Path to YOLO model weights")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()

    if not args.input:
        print("❌ Error: No input file specified!")
        print("Usage: python detect.py <image_or_video_path> [--model <model_path>] [--output <output_path>]")
        return

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ Error: File '{input_path}' not found!")
        return

    ext = os.path.splitext(input_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    video_exts = ['.mp4', '.avi', '.webm', '.mov', '.mkv']
    if ext in image_exts:
        process_image(input_path, model_path_override=args.model, output_name=args.output)
    elif ext in video_exts:
        process_video(input_path, model_path_override=args.model, output_name=args.output)
    else:
        print(f"❌ Error: Unsupported file format '{ext}'!")

if __name__ == "__main__":
    main()
