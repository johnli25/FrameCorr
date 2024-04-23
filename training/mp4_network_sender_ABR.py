import os
import socket

host = "172.22.153.20"
port = 50013

# Directories for different bitrates
bitrate_folders = {
    'low': "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output_crf30",
    'medium': "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output",
    'high': "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output_crf18"
}

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
        if message == "END":
            break
        print(message)
        video_name, bitrate = message.split(',')
        video_path = os.path.join(bitrate_folders[bitrate], video_name)
        send_video(video_path)
    s_sock.close()
    print("socket_closed") 
