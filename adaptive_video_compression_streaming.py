import os
import subprocess
import time
import re
from PIL import Image
import numpy as np
from collections import defaultdict
import cv2
import random

def create_hls_stream(input_file, output_dir):
    renditions = [('640x480', '900k'), ('1280x720', '3000k'), ('1920x1080', '6000k'), ('2560x1440', '9000k'), ('3840x2160', '18000k')]
    master_playlist = []

    for resolution, bitrate in renditions:
        output_subdir = os.path.join(output_dir, f'{bitrate}')
        os.makedirs(output_subdir, exist_ok=True)

        command = [
            'ffmpeg',
            '-i', input_file,
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-s', resolution,
            '-start_number', '0',
            '-hls_time', '10',
            '-hls_list_size', '0',
            '-b:v', bitrate,
            '-f', 'hls',
            os.path.join(output_subdir, 'output.m3u8')
        ]
        subprocess.run(command, check=True)

        master_playlist.append(f'#EXT-X-STREAM-INF:BANDWIDTH={bitrate.replace("k", "000")},RESOLUTION={resolution}\n{bitrate}/output.m3u8')

    # Write master playlist
    with open(os.path.join(output_dir, 'master.m3u8'), 'w') as f:
        f.write('#EXTM3U\n' + '\n'.join(master_playlist))

def process_all_videos(input_dir, output_dir):
    for class_folder in os.listdir(input_dir):
        # Ignore .DS_Store files and text files
        if class_folder == '.DS_Store':
            continue

        for file in os.listdir(os.path.join(input_dir, class_folder)):
            input_file_path = os.path.join(input_dir, class_folder, file)
            if input_file_path.lower().endswith(('.mp4', '.avi')):  # Add or remove file extensions as needed
                create_hls_stream(input_file_path, output_dir)

def calculate_mse(original_frames_directory, compressed_frames_directory):
    """
    Calculate the Mean Squared Error (MSE) between the original and compressed video frames.
    
    Parameters:
    - original_frames_directory: Path to the directory containing original video frames.
    - compressed_frames_directory: Path to the directory containing compressed video frames.
    """
    mse_values = defaultdict(list)

    # Iterate over all frames in the original directory
    for frame_file in os.listdir(original_frames_directory):

        match = re.match(r"([a-z_]+)([0-9]+)", os.path.splitext(frame_file)[0], re.I)

        if match:
            action, number = match.groups()

        # Ensure the file is an image file
        if frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the original and compressed frames
            original_frame = Image.open(os.path.join(original_frames_directory, frame_file))
            compressed_frame = Image.open(os.path.join(compressed_frames_directory, frame_file))

            # Convert the images to numpy arrays
            original_array = np.array(original_frame) / 255.0
            compressed_array = np.array(compressed_frame) / 255.0

            # Calculate the MSE for this frame and add it to the dict
            mse = np.sum((original_array - compressed_array) ** 2)
            mse_values[str(action) + str(number)].append(mse)

def extract_bytes_from_video(output_videos):
    '''
    Extract bytes from the compressed video files and save them in a text file.
    '''
    output_directory = 'compressed_video_bytes_random_drop'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Open the MP4 video file
    for file in os.listdir(output_videos):
        if os.path.isfile(os.path.join(output_videos, file)) and file.lower().endswith('.mp4'):
            video_path = os.path.join(output_videos, file)

            cap = cv2.VideoCapture(video_path)

            # Check if the video file is opened successfully
            if not cap.isOpened():
                print("Error: Could not open the video file.")
                exit()
            total_bytes = 0
            output_filepath = os.path.join(output_directory, f"{os.path.splitext(file)[0]}.txt")
            with open(output_filepath, 'wb') as f:
                frame_number = 0
                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret: break

                    # Get the encoded information of the frame
                    encoded_info = cv2.imencode('.jpg', frame)[1].tobytes()
                    # 'encoded_info' now contains the encoded information of the frame
                    f.write(encoded_info)

                    # NOTE: Drop 10% of the bytes randomly (this might break lol)
                    # encoded_info = drop_data(encoded_info)
                    # nparr = np.frombuffer(encoded_info, np.uint8)
                    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # cv2.imshow('image', img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
                    # Process the frame or save the encoded information as needed
                    total_bytes += len(encoded_info) # NOTE: encoded_info is a bytes object, so calling len() returns the number of bytes (not the # of elements)!
                    frame_number += 1 # Increment frame number

            avg_bytes_per_vid = total_bytes / frame_number if frame_number else 0
            # if file in ['diving7.mp4', 'golf_front7.mp4', 'golf_front8.mp4', 'kick_front9.mp4', 'kick_front10.mp4', 'lifting5.mp4', 'lifting6.mp4', 'riding_horse11.mp4', 'riding_horse12.mp4', 'running11.mp4', 'running12.mp4', 'running13.mp4', 'skating11.mp4', 'skating12.mp4', 'swing_bench18.mp4', 'swing_bench19.mp4', 'swing_bench20.mp4']:
            if file in ['diving7.mp4', 'diving8.mp4', 'golf_front7.mp4', 'golf_front8.mp4', 'kick_front9.mp4', 'kick_front10.mp4', 'lifting5.mp4', 'lifting6.mp4', 'riding_horse8.mp4', 'riding_horse9.mp4', 'running7.mp4', 'running8.mp4', 'running9.mp4', 'skating8.mp4', 'skating9.mp4', 'swing_bench7.mp4', 'swing_bench8.mp4', 'swing_bench9.mp4']:
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                print(f"Total VIDEO (not all frames) bytes = {len(video_bytes)} for {file}.")
                print(f"{file}: Total bytes of all extracted frames = {total_bytes}, average = {avg_bytes_per_vid} bytes per frame")

            # Release the video capture object and close the video file
            cap.release()

# Driver
input_dir = 'video_data'
output_dir = 'hls_data'
# process_all_videos(input_dir, output_dir)
# print("The reconstruction MSE is ", calculate_mse('new_video_frames_dataset', output_dir))
# extract_bytes_from_video(output_dir)