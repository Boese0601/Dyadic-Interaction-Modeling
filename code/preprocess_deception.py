import os
import cv2

def find_merge_mp4_files(input_path):
    merge_mp4_files = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".mp4") and "merge" in root:
                merge_mp4_files.append(os.path.join(root, file))
    return merge_mp4_files

def split_videos(merge_mp4_files):
    for merge_mp4_file in merge_mp4_files:
        # Load the video
        cap = cv2.VideoCapture(merge_mp4_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create VideoWriter objects for left and right videos
        left_video_writer = cv2.VideoWriter(merge_mp4_file.replace(".mp4", "_left.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width // 2, height))
        right_video_writer = cv2.VideoWriter(merge_mp4_file.replace(".mp4", "_right.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width // 2, height))
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Split the frame into left and right halves
            left_half = frame[:, :width // 2]
            right_half = frame[:, width // 2:]
            
            # Write the left and right halves to the respective videos
            left_video_writer.write(left_half)
            right_video_writer.write(right_half)
            
        # Release the video capture and video writers
        cap.release()
        left_video_writer.release()
        right_video_writer.release()

# Example usage
input_path = "/path/to/input"
merge_mp4_files = find_merge_mp4_files(input_path)
split_videos(merge_mp4_files)
