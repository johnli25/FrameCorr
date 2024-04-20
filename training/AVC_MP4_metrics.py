import os
import socket
import struct
import os
import subprocess
import time
import re
from PIL import Image
import numpy as np
from collections import defaultdict
import random
import ffmpeg
#import h264_avc

def create_decoded_output_frames(output_videos, output_frames_directory):
    """
    (Decoding/Decompression) Extract frames from all .mp4 files in the output_videos and save them in output_frames_directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_frames_directory, exist_ok=True)

    # List all files in the input directory
    for file in os.listdir(output_videos):
        if os.path.isfile(os.path.join(output_videos, file)) and file.lower().endswith('.mp4'):
            match = re.match(r"([a-z_]+)([0-9]+)", os.path.splitext(file)[0], re.I)

            if match:
                action, number = match.groups()

                # Define the FFmpeg command for extracting frames
                command = [
                    'ffmpeg', '-i', os.path.join(output_videos, file), 
                    os.path.join(output_frames_directory, f"{action}_{number}_%03d.jpg")
                ]

                # Run the FFmpeg command
                subprocess.run(command, check=True)
                
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



original_frames_folder = "/home/deepakn3/Progressive-Neural-Compression/compressed_videos_frames_output_dataset"
metrics_input_folder_vids = "/home/deepakn3/Progressive-Neural-Compression/send_vids"
metrics_input_folder_frames= "/home/deepakn3/Progressive-Neural-Compression/send_frames"
create_decoded_output_frames(metrics_input_folder_vids,metrics_input_folder_frames)
calculate_mse(original_frames_folder,metrics_input_folder_frames) 


# USE metrics library
    