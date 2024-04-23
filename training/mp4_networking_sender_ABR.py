import os
import socket
from moviepy.editor import VideoFileClip

host = "172.22.153.20"
port = 50013

# Directories for different bitrates
def choose_bitrate(throughput):
    if throughput < 8000000: # low 
        print("low")
        return "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output_crf30"
    elif throughput < 12000000: # default-medium
        print("medium")
        return "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output"
    else: # high
        print("high")
        return "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output_crf18"

DELIMITER = b'\xFF\x00\xFF'

def send_video(file_path):
    with open(file_path, 'rb') as f:
        video_bytes = f.read()
        print(file_path, len(video_bytes))

        chunk_size = 4096
        for i in range(0, len(video_bytes), chunk_size):
            chunk = video_bytes[i:i + chunk_size]
            s_sock.sendall(chunk)
        s_sock.sendall(DELIMITER)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
    s_sock.connect((host, port))
    first_one_flag = True
    while True:
        if first_one_flag:
            # hardcode the first video sent 
            send_video("/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output/diving11.mp4")
            first_one_flag = False
            continue
        # Receive the next video file name and bitrate to use from the receiver
        message = s_sock.recv(1024).decode().strip()
        if "END" in message:
            break
        print(message)
        video_name, throughput = message.split(',')
        throughput = int(throughput)
        video_path = os.path.join(choose_bitrate(throughput), video_name)
        send_video(video_path)
    s_sock.close()
    print("socket_closed") 
