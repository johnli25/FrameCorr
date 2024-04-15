
import os
import socket
import struct
import sys

host = "172.22.154.247"
port = 50013
video_folder = "/home/deepakn3/Progressive-Neural-Compression/compressed_videos_output"

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: 
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

host = "172.22.154.247"
port = 50002
MAX_PAYLOAD = 500
DELIMITER = b'\xFF\x00\xFF' 
video_folder = "/home/deepakn3/Progressive-Neural-Compression/compressed_videos_output"

def get_object_size(obj):
    return sys.getsizeof(obj)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
    s_sock.connect((host, port))
    for filename in os.listdir(video_folder):
       # video_file = os.listdir(video_folder)[0]  # Adjust if you need specific selection
        video_path = os.path.join(video_folder, filename)
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
            num_bytes = get_object_size(video_bytes)
            print(video_path,len(video_bytes))
        
            # num_bytes_packed = struct.pack('!I', num_bytes)  
            # s_sock.sendall(num_bytes_packed)
            #s_sock.recv(20)  

            chunk_size = 4096  # Adjust if needed 
            for i in range(0, len(video_bytes), chunk_size):
                chunk = video_bytes[i:i + chunk_size]
                s_sock.sendall(chunk)
            s_sock.sendall(b"")
    s_sock.close()
    print("socket_closed") 
    
    
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
#     s_sock.connect((host, port))

#     for filename in os.listdir(video_folder):
#         video_path = os.path.join(video_folder, filename)

#         with open(video_path, 'rb') as f:
#             video_bytes = f.read()
#             num_bytes = len(video_bytes)
#             print(video_path, num_bytes) 

#             num_bytes_packed = struct.pack('!I', num_bytes)  
#             s_sock.send(num_bytes_packed) 
            
#             confirmation_bytes = s_sock.recv(8)
#             confirm_format= confirmation_bytes.decode('utf-8')
#             print(confirm_format)
#             if confirm_format == b'received':
#                 s_sock.sendall(video_bytes)

#     s_sock.close()
#     print("socket_closed") 

