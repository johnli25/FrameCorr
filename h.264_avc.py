import os
import subprocess
import time
import re
from PIL import Image
import numpy as np
from collections import defaultdict
import cv2
import random

'''
Helper function to get the number of frames in a video file using FFprobe.
'''
def get_number_of_frames(video_file_path):
    command = [
        'ffprobe', 
        '-v', 'error', 
        '-select_streams', 'v:0', 
        '-show_entries', 'stream=nb_frames', 
        '-of', 'default=nokey=1:noprint_wrappers=1', 
        video_file_path
    ]
    output = subprocess.check_output(command).decode('utf-8')
    return int(output)


def compress_videos(input_directory, output_directory, codec='libx264', crf=23, preset='medium'):
    """
    (Encoding) Compress all .avi files in the input_directory using H.264 AVC codec with FFmpeg.
    
    Parameters:
    - input_directory: Path to the directory containing .avi video files.
    - output_directory: Path to the directory where compressed videos will be saved.
    - codec: Video codec to use for compression (default: 'libx264').
    - crf: Constant Rate Factor for quality level (default: 23).
    - preset: Encoding speed to compression ratio (default: 'medium').
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # List all files in the input directory
    for class_folder in os.listdir(input_directory):
        # Construct the full file path
        if class_folder == '.DS_Store':
            continue
        for file in os.listdir(os.path.join(input_directory, class_folder)):
            input_file_name = file
            input_file_path = os.path.join(input_directory, class_folder, input_file_name)

            # Check if the file is an .avi file
            if os.path.isfile(input_file_path) and input_file_path.lower().endswith('.avi'):
                # Define the output file name (change extension to .mp4)
                output_file_name = f"{os.path.splitext(input_file_name)[0]}.mp4"
                output_file_path = os.path.join(output_directory, output_file_name)
                
                # Construct the FFmpeg command for compression
                command = [
                    'ffmpeg',
                    '-i', input_file_path,     # Input file
                    '-c:v', codec,             # Video codec
                    '-crf', str(crf),          # Constant Rate Factor
                    '-preset', preset,         # Preset for compression speed/quality
                    '-c:a', 'aac',             # Audio codec
                    '-b:a', '128k',            # Audio bitrate
                    output_file_path           # Output file
                ]

                num_frames = get_number_of_frames(input_file_path)
                print(f"Number of frames in {input_file_name}: {num_frames}")
                
                # Execute the command and time it
                start_time = time.time()
                try:
                    subprocess.run(command, check=True)
                    duration = time.time() - start_time
                    print(f"Compressed: {output_file_name} in {duration:.2f} seconds.")
                    num_frames = get_number_of_frames(output_file_path)
                    print(f"(Output) Number of frames in {output_file_name}: {num_frames}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to compress {output_file_name}: {e}")

                
# NOTE: didn't end up using this dict
action_to_index = {"running": 0, "skating": 1, "kick_front": 2, "riding_horse": 3, "golf_front": 4, "swing_bench": 5,"diving": 6,"lifting": 7}

def create_new_input_frames(input_directory, output_directory): 
    """
    Extract frames from all .avi files in the input_directory and save them in the output_directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Open the output file in write mode
    # List all files in the input directory
    for class_folder in os.listdir(input_directory):
        # Ignore .DS_Store files and text files
        if class_folder == '.DS_Store':
            continue

        for file in os.listdir(os.path.join(input_directory, class_folder)):
            input_file_path = os.path.join(input_directory, class_folder, file)

            # Check if the file is an .avi file
            if os.path.isfile(input_file_path) and input_file_path.lower().endswith('.avi'):
                # Use regex to split the base name into the action and the number
                match = re.match(r"([a-z_]+)([0-9]+)", os.path.splitext(file)[0], re.I)

                if match:
                    action, number = match.groups()

                    # Define the FFmpeg command for extracting frames
                    command = [
                        'ffmpeg', '-i', input_file_path, 
                        os.path.join(output_directory, f"{action}_{number}_%03d.jpg")
                    ]

                    # Run the FFmpeg command
                    subprocess.run(command, check=True)


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


def create_new_labels_txt(directory='new_video_frames_dataset'):
    """
    Create a new .txt file containing the file names and their corresponding action labels.
    """
    output_file = 'new_video_numidx_labels.txt'
    with open(output_file, 'w') as f:

        # Write the file name and the action to the output file
        for filename in sorted(os.listdir(directory)):
            print(filename)
            if "running" in filename:
                number = 0
            elif "skating" in filename:
                number = 1
            elif "kick_front" in filename:
                number = 2
            elif "riding_horse" in filename:
                number = 3
            elif "golf_front" in filename:
                number = 4
            elif "swing_bench" in filename:
                number = 5
            elif "diving" in filename:
                number = 6
            elif "lifting" in filename:
                number = 7
            else: # throw error
                raise Exception(f"Error: {filename} does not match any action.")

            f.write(f"{filename} {number}\n")

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

    # Calculate the average MSE over all image frames per video
    for key in mse_values:
        mse_values[key] = np.mean(mse_values[key])

    print("mse_values dict length: ", len(mse_values))
    print(mse_values['diving_7'])
    print(mse_values['diving_8'])
    print(mse_values['golf_front_7'])
    print(mse_values['golf_front_8'])
    print(mse_values['kick_front_9'])
    print(mse_values['kick_front_10'])
    print(mse_values['lifting_5'])
    print(mse_values['lifting_6'])
    print(mse_values['riding_horse_8'])
    print(mse_values['riding_horse_9'])
    print(mse_values['running_7'])
    print(mse_values['running_8'])
    print(mse_values['running_9'])
    print(mse_values['skating_8'])
    print(mse_values['skating_9'])
    print(mse_values['swing_bench_7'])
    print(mse_values['swing_bench_8'])
    print(mse_values['swing_bench_9'])
    return mse_values # return average mse_values per vid

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
                # print(f"Total Bytes = {len(video_bytes)} for {file}.")
                print(f"{file}: Total bytes of all extracted frames = {total_bytes}, average = {avg_bytes_per_vid} bytes per frame")

            # Release the video capture object and close the video file
            cap.release()

def drop_data(encoded_info):
    num_bytes_to_drop_out = int(len(encoded_info) * 0.1)
    indices_to_drop = random.sample(range(len(encoded_info)), num_bytes_to_drop_out)

    encoded_info_np = np.frombuffer(encoded_info, dtype=np.uint8)
    encoded_info_np = encoded_info_np.copy()

    for idx in indices_to_drop:
        encoded_info_np[idx] = 0
    
    encoded_info = encoded_info_np.tobytes()
    return encoded_info

original_input_dir = 'video_data'
output_dir = 'compressed_videos_output'
home_dir = os.getcwd()

start_time = time.time()
'''
uncomment the below function driver calls when necessary
'''
# create_new_input_frames(original_input_dir, 'new_video_frames_dataset')
# create_new_labels_txt('new_video_frames_dataset')
# compress_videos(original_input_dir, output_dir)
# create_decoded_output_frames(output_dir, 'compressed_video_frames_output_dataset')
# print("The reconstruction MSE is ", calculate_mse('new_video_frames_dataset', 'compressed_video_frames_output_dataset'))
extract_bytes_from_video(output_dir)
print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")

# NOTE: sanity checks
num_files = sum([len(files) for r, d, files in os.walk("video_data")])
print(f'There are {num_files} - 1 files in video_data directory (due to the .DS_store file, dont count it).')

num_files = len([f for f in os.listdir("compressed_videos_output") if os.path.isfile(os.path.join("compressed_videos_output", f))])
print(f'There are {num_files} files in video_data directory.')

num_files = len([f for f in os.listdir("new_video_frames_dataset") if os.path.isfile(os.path.join("new_video_frames_dataset", f))])
print(f'There are {num_files} files in new_video_frames_dataset directory.')

num_files = len([f for f in os.listdir("compressed_video_frames_output_dataset") if os.path.isfile(os.path.join("compressed_video_frames_output_dataset", f))])
print(f'There are {num_files} files in compressed_video_frames_output_dataset directory.')

num_files = len([f for f in os.listdir("compressed_video_bytes") if os.path.isfile(os.path.join("compressed_video_bytes", f))])
print(f'There are {num_files} files in compressed_video_bytes directory.')

filename = 'new_video_numidx_labels.txt'  # replace with your file
with open(filename, 'r') as file:
    lines = file.readlines()
print(f'The file new_video_numidx_labels.txt has {len(lines)} lines.')
