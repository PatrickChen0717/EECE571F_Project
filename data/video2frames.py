import os
import cv2
import glob

paths = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\left_video.mp4", recursive=True)

all_videos = paths
print("num videos:", len(all_videos))

for video_path in all_videos:
    video_dir = os.path.dirname(video_path)

    # decide output folder name
    if "left_video.mp4" in video_path:
        output_dir = os.path.join(video_dir, "left_frames")
        
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out_path = os.path.join(output_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(out_path, frame)
            frame_idx += 1

        cap.release()

        print(f"{video_path} -> Saved {frame_idx} frames to {output_dir}")
        
    elif "right_video.mp4" in video_path:
        output_dir = os.path.join(video_dir, "right_frames")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out_path = os.path.join(output_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(out_path, frame)
            frame_idx += 1

        cap.release()

        print(f"{video_path} -> Saved {frame_idx} frames to {output_dir}")
        