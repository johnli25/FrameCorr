import os
import socket
import struct
from .. import h.264_avc.py
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        print(len(newbuf))
        if not newbuf: 
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

host = "172.22.154.247"
port = 50013
MAX_PAYLOAD = 500
DELIMITER = b'\xFF\x00\xFF' 

def next_filename(file_iter):
    try:
        return next(file_iter)
    except StopIteration:
        return None  # Signal that there are no more files


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
    s_sock.bind((host, port))
    s_sock.listen()
    sock_conn, sock_addr = s_sock.accept()
    video_folder = "/home/deepakn3/Progressive-Neural-Compression/compressed_videos_output"
        
    # Receive video data in chunks and write to file
    file_iter = iter(os.listdir(video_folder))
    
    # length_bytes = sock_conn.recv(4)
    # video_length = struct.unpack('!I', length_bytes)[0]
    #sock_conn.send(b'received') 
    output_folder = "/home/deepakn3/Progressive-Neural-Compression/send_vids"
    print(next(file_iter))
    output_filename = next(file_iter)
    output_path = os.path.join(output_folder, output_filename)
    buffer = b''
    while True:
        with open(output_path, 'ab+') as f:
            print(output_path)
            received_bytes = 0
            while buffer.find(DELIMITER) == -1:
                chunk = sock_conn.recv(4096) 
                if not chunk: 
                    break 
                buffer += chunk
                received_bytes += len(chunk)
                print(received_bytes)
            mp4_bytes, _, buffer = buffer.partition(DELIMITER)
            f.write(mp4_bytes)
            output_filename = next_filename(file_iter)
            if output_filename == None:
                break    
            output_path = os.path.join(output_folder, output_filename)
    sock_conn.close()
    
metrics_input_folder = "/home/deepakn3/Progressive-Neural-Compression/send_vids"
create_decoded_output_frames(metrics_input_folder, output_frames_directory):
    
# host = "172.22.154.247"
# port = 50013
# output_folder = "/home/deepakn3/Progressive-Neural-Compression/send_vids"
# video_folder = "/home/deepakn3/Progressive-Neural-Compression/compressed_videos_output"


# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
#     s_sock.bind((host, port))
#     s_sock.listen()
#     sock_conn, sock_addr = s_sock.accept()
#     for filename in os.listdir(video_folder):
#         while True:  
#             # Receive the length of the video data
#             length_bytes = sock_conn.recv(4)
#             video_length = struct.unpack('!I', length_bytes)[0]
#             print(video_length)
#             sock_conn.send(b'received')
            

#             output_path = os.path.join(output_folder, filename)

#             # Receive the entire video data in one go
#             with open(output_path, 'ab+') as f:
#                 #print(video_length)
#                 video_data = recvall(sock_conn, video_length-30)
#                 print(type(video_data))
#                 f.write(video_data)

#             print(f"Received video: {output_path}")

#     sock_conn.close() 

    