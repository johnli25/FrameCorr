
import os
import socket
import struct
import sys
import time

host = "172.22.153.20"
port = 50013
video_folder = "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output"

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: 
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

host = "172.22.153.20"
port = 50013
MAX_PAYLOAD = 500
DELIMITER = b'\xFF\x00\xFF' 
video_folder = "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output"

def get_object_size(obj):
    return sys.getsizeof(obj)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
    s_sock.connect((host, port))
    for filename in sorted(os.listdir(video_folder)):
        video_path = os.path.join(video_folder, filename)
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
            num_bytes = get_object_size(video_bytes)
            print(video_path, len(video_bytes))

            chunk_size = 4096  # Adjust if needed 
            for i in range(0, len(video_bytes), chunk_size):
                chunk = video_bytes[i:i + chunk_size]
                s_sock.sendall(chunk)
            s_sock.sendall(DELIMITER)
    s_sock.close()
    print("socket_closed") 
