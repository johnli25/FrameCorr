import os
import socket
import time

host = "172.22.153.20"
port = 50013

DELIMITER = b'\xFF\x00\xFF'
output_folder = "/path/to/output"

# Specify the input folder where video files or their names are stored
input_folder = "/path/to/input_folder"

def measure_throughput(start_time, data_received):
    end_time = time.time()
    throughput = data_received * 8 / (end_time - start_time)  # bits per second
    return throughput

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
    s_sock.bind((host, port))
    s_sock.listen()
    conn, addr = s_sock.accept()
    video_files = sorted(os.listdir(input_folder))  # Dynamically get the list of video files

    for video_file in video_files:
        start_time = time.time()
        total_data_received = 0
        while True:
            chunk = conn.recv(4096)
            if DELIMITER in chunk:
                video_data, _, remaining = chunk.partition(DELIMITER)
                total_data_received += len(video_data)
                break
            total_data_received += len(chunk)
        
        # Determine appropriate bitrate based on throughput
        throughput = measure_throughput(start_time, total_data_received)
        if throughput < 1e6:  # less than 1 Mbps
            bitrate = 'low'
        elif throughput < 5e6:  # less than 5 Mbps
            bitrate = 'medium'
        else:
            bitrate = 'high'
        
        # Inform sender of the next video file and desired bitrate
        conn.send(f"{video_file},{bitrate}".encode())
    conn.send(b"END")
    conn.close()
    print("Reception and adjustment complete.")
