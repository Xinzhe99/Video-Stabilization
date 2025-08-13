import cv2
import numpy as np
from video_stabilizer import VideoStabilizer
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Video Stabilization with Kalman Filter")
    parser.add_argument('--feature_type', type=str, default='gftt', choices=['gftt', 'orb', 'sift'], help='Feature detection algorithm')
    parser.add_argument('--input_path', type=str, default=r'C:\Users\xxz\Desktop\A001_08131429_C013.mov', help='Input video file path')
    parser.add_argument('--output_path', type=str, default='stabilized_output_kalman.avi', help='Output stabilized video file path')
    parser.add_argument('--show', type=bool,default=True, help='Show comparison window during processing')
    parser.add_argument('--use_camera', action='store_true', help='Use camera instead of video file (default: use video file)')
    parser.add_argument('--max_corners', type=int, default=200, help='Max corners for GFTT/ORB/SIFT')
    parser.add_argument('--quality_level', type=float, default=0.01, help='Quality level for GFTT')
    parser.add_argument('--min_distance', type=int, default=30, help='Min distance for GFTT')
    parser.add_argument('--center_crop_ratio', type=float, default=1.3, help='Center crop ratio for removing black borders (1.0 = no crop, >1.0 = zoom in)')
    parser.add_argument('--downsample_ratio', type=float, default=0.2, help='Downsample ratio for feature detection (0.5 = half resolution, 1.0 = full resolution)')
    args = parser.parse_args()

    feature_type = args.feature_type
    input_path = args.input_path
    output_path = args.output_path
    show = args.show
    use_camera = args.use_camera
    max_corners = args.max_corners
    quality_level = args.quality_level
    min_distance = args.min_distance
    center_crop_ratio = args.center_crop_ratio
    downsample_ratio = args.downsample_ratio

    # Create stabilizer with feature_type, center_crop_ratio and downsample_ratio
    stabilizer = VideoStabilizer(feature_type=feature_type, center_crop_ratio=center_crop_ratio, downsample_ratio=downsample_ratio)
    stabilizer.test = show
    # Pass max_corners, quality_level, min_distance to stabilizer if needed
    if hasattr(stabilizer, 'max_corners'):
        stabilizer.max_corners = max_corners
    if hasattr(stabilizer, 'quality_level'):
        stabilizer.quality_level = quality_level
    if hasattr(stabilizer, 'min_distance'):
        stabilizer.min_distance = min_distance
    
    # Choose video source based on argument
    if use_camera:
        cap = cv2.VideoCapture(0)
        print("Using camera as video source")
    else:
        cap = cv2.VideoCapture(input_path)
        print(f"Using video file as source: {input_path}")
    
    if not cap.isOpened():
        if use_camera:
            print("Cannot open camera")
        else:
            print(f"Cannot open video: {input_path}")
        return
    
    ret, frame_1 = cap.read()
    if not ret:
        print("Cannot read the first frame")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print("Start video stabilization... Press 'q' to quit")
    total_time = 0.0
    frame_count = 0
    while True:
        try:
            ret, frame_2 = cap.read()
            if not ret:
                break
            start_time = time.time()
            stabilized_frame = stabilizer.stabilize(frame_1, frame_2)
            end_time = time.time()
            total_time += (end_time - start_time)
            frame_count += 1
            out.write(stabilized_frame)
            # Show stabilized video if requested
            # if show:
            #     cv2.imshow("Stabled", stabilized_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            frame_1 = frame_2.copy()
        except Exception as e:
            print(f"Error: {e}")
            ret, frame_1 = cap.read()
            if not ret:
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video stabilization finished!")
    if frame_count > 0:
        print(f"Average time per frame: {total_time / frame_count:.4f} s, total frames: {frame_count}")

if __name__ == "__main__":
    main()