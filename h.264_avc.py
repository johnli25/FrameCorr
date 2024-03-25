import os
import subprocess
import time
import re

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
    Compress all .avi files in the input_directory using H.264 AVC codec with FFmpeg.
    
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
                    '-i', input_file_path,      # Input file
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
                
# NOTE: didn't end up using
action_to_index = {"running": 0, "skating": 1, "kick_front": 2, "riding_horse": 3, "golf_front": 4, "swing_bench": 5,"diving": 6,"lifting": 7}

def create_new_dataset(input_directory, output_directory): 
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
                # Create a new directory for the frames
                # frames_directory = os.path.join(output_directory, class_folder)
                frames_directory = output_directory
                os.makedirs(frames_directory, exist_ok=True)

                # Use regex to split the base name into the action and the number
                match = re.match(r"([a-z_]+)([0-9]+)", os.path.splitext(file)[0], re.I)

                if match:
                    action, number = match.groups()

                    # Define the FFmpeg command for extracting frames
                    command = [
                        'ffmpeg', '-i', input_file_path, 
                        os.path.join(frames_directory, f"{action}_{number}_%03d.jpg")
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
                action, number = "running", 0
            elif "skating" in filename:
                action, number = "skating", 1
            elif "kick_front" in filename:
                action, number = "kick_front", 2
            elif "riding_horse" in filename:
                action, number = "riding_horse", 3
            elif "golf_front" in filename:
                action, number = "golf_front", 4
            elif "swing_bench" in filename:
                action, number = "swing_bench", 5
            elif "diving" in filename:
                action, number = "diving", 6
            elif "lifting" in filename:
                action, number = "lifting", 7
            else: # throw error
                raise Exception(f"Error: {filename} does not match any action.")

            f.write(f"{filename} {number}\n")

# Example usage
original_input_dir = 'video_data_files'
output_dir = 'compressed_videos_output'
home_dir = os.getcwd()

start_time = time.time()
# uncomment these function driver calls when necessary
# create_new_dataset(original_input_dir, 'new_video_frames_dataset')
create_new_labels_txt('new_video_frames_dataset')
# compress_videos(original_input_dir, output_dir)
print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")
