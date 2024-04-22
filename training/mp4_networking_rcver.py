import os
import socket
import struct
import time 
#from .. import h.264_avc.py
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

host = "172.22.153.20"
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
    video_folder = "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/compressed_videos_output"
        
    # Receive video data in chunks and write to file
    file_iter = iter(sorted(os.listdir(video_folder)))
    
    output_folder = "/home/johnwl2/FrameCorr/Progressive-Neural-Compression/received_mp4_vids"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = next(file_iter)
    output_path = os.path.join(output_folder, output_filename)
    buffer = b''
    start = time.perf_counter()
    total_bytes_received = 0
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
        total_bytes_received += received_bytes

    sock_conn.close()
    end = time.perf_counter()
    duration = end - start
    print(f"Received {total_bytes_received} bytes")
    print(f"Code execution time: {duration:.6f} seconds") 

    