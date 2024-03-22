import os
import subprocess
import time

def compress_videos(home_dir, input_directory, output_directory, codec='libx264', crf=23, preset='medium'):
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
            file_name = file
            input_file_path = os.path.join(input_directory, class_folder, file_name)

            # Check if the file is an .avi file
            if os.path.isfile(input_file_path) and input_file_path.lower().endswith('.avi'):
                # Define the output file name (change extension to .mp4)
                output_file_name = f"{os.path.splitext(file_name)[0]}.mp4"
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
                
                # Execute the command and time it
                start_time = time.time()
                try:
                    subprocess.run(command, check=True)
                    duration = time.time() - start_time
                    print(f"Compressed: {output_file_name} in {duration:.2f} seconds.")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to compress {file_name}: {e}")

# Example usage
input_dir = 'video_data_files'
output_dir = 'compressed_videos'
home_dir = os.getcwd()

start_time = time.time()
compress_videos(home_dir, input_dir, output_dir)
print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")
