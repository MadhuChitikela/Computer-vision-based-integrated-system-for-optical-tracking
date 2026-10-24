import os
import sys
import glob
import argparse

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PATTERNS = [

    'detected_*.png',
    'tracked_*.webm',
    'tracked_*.mp4',
    'compare_*.mp4',
    'zoomed_*.mp4',
    'telemetry_*.json',
    'tracking_telemetry_*.json',
    'tracking_telemetry_plot.png',
    'enhancement_demo_comparison.png',
    'kalman_vs_deepsort_*.png',
    'model_comparison_*.png',
    'Screenshot*.png'
]

def cleanup(directory=".", dry_run=False):
    """
    Remove temporary output artifacts, generated videos, plots, and logs.
    """
    removed_count = 0
    matched_files = []

    for pattern in PATTERNS:
        search_path = os.path.join(directory, pattern)
        for file_path in glob.glob(search_path):
            matched_files.append(file_path)

    if not matched_files:
        print("✨ Workspace is clean. No temporary artifacts found.")
        return

    print(f"Found {len(matched_files)} temporary artifact(s):")
    for fp in matched_files:
        if dry_run:
            print(f" [DRY-RUN] Would remove: {os.path.basename(fp)}")
        else:
            try:
                os.remove(fp)
                print(f" 🗑️ Removed: {os.path.basename(fp)}")
                removed_count += 1
            except OSError as e:
                print(f" ❌ Error removing {fp}: {e}")

    if dry_run:
        print(f"\n🔍 Dry-run complete. {len(matched_files)} files would be deleted.")
    else:
        print(f"\n✅ Total {removed_count} temporary files removed.")

def main():
    parser = argparse.ArgumentParser(description="Clean up temporary tracking videos, images, and telemetry files")
    parser.add_argument("--dir", "-d", default=".", help="Directory to clean (default: current directory)")
    parser.add_argument("--dry-run", action="store_true", help="List files to be deleted without removing them")
    args = parser.parse_args()

    cleanup(directory=args.dir, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
