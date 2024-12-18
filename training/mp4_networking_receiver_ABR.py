import os
import socket
import time

host = "172.22.153.20"
port = 50013

DELIMITER = b'\xFF\x00\xFF'
output_folder = "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/received_mp4_vids_ABR"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# reference folder for all the names of the compressed video files 
input_folder = "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output"

def measure_throughput(start_time, data_received):
    end_time = time.time()
    print(end_time - start_time)
    throughput = data_received / (end_time - start_time)  # bits per second
    return int(throughput)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
    s_sock.bind((host, port))
    s_sock.listen()
    conn, addr = s_sock.accept()
    total_start_time = time.time()
    video_files = sorted(os.listdir(input_folder))  # Dynamically get the list of video files
    for i in range(len(video_files)):
        video_file = video_files[i]
        output_path = os.path.join(output_folder, video_file)
        start_time = time.time()
        received_bytes = 0
        buffer = b''
        # with open(output_path, 'ab+') as f:
        #     print(output_path)                
        #     while True:
        #         chunk = conn.recv(4096)
        #         if DELIMITER in chunk:
        #             video_data, _, remaining = chunk.partition(DELIMITER)
        #             f.write(video_data)  # Write the last piece of video data before the delimiter
        #             total_data_received += len(video_data)
        #             break
        #         f.write(chunk)
        #         total_data_received += len(chunk)
        with open(output_path, 'ab+') as f:
            print(output_path)
            received_bytes = 0
            while buffer.find(DELIMITER) == -1:
                chunk = conn.recv(4096) 
                if not chunk: 
                    break 
                buffer += chunk
                received_bytes += len(chunk)
            print(received_bytes)
            mp4_bytes, _, buffer = buffer.partition(DELIMITER)
            f.write(mp4_bytes)

        # Determine appropriate bitrate based on throughput
        throughput = measure_throughput(start_time, received_bytes)
        print("throughput", str(throughput))
        
        # Inform sender of the next video file and desired bitrate
        if i + 1 < len(video_files):  # If there is a next video file
            next_video_file = video_files[i + 1]
            conn.send(f"{next_video_file},{throughput}".encode())
        else:  # If this is the last video file
            conn.send(b"END")

    total_end_time = time.time()
    total_time_elapsed = total_end_time - total_start_time
    print(f"Total time elapsed: {total_time_elapsed} seconds")
    conn.close()
