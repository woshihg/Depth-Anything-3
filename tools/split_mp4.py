import cv2
import os
import argparse
import numpy as np

def extract_frames(video_path, output_dir, num_frames=256):
    """
    Extract a fixed number of frames from a video file at equal intervals.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, which is less than requested {num_frames}.")
        num_frames = total_frames

    # Calculate intervals
    # We use np.linspace to get indices to ensure we cover the whole video
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    count = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame at index {idx}")
            continue
        
        output_path = os.path.join(output_dir, f"{count:04d}.jpg")
        cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        count += 1
        
        if count % 50 == 0:
            print(f"Extracted {count}/{num_frames} frames...")

    cap.release()
    print(f"Done! Extracted {count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MP4 at equal intervals")
    parser.add_argument("--input", required=True, help="Path to input mp4 file")
    parser.add_argument("--output", required=True, help="Output directory for images")
    parser.add_argument("--num", type=int, default=256, help="Number of frames to extract")

    args = parser.parse_args()
    extract_frames(args.input, args.output, args.num)
