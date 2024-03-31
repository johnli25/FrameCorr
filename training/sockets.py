import argparse
import socket
import numpy as np 

host = "172.22.154.247"
port = 50000
MAX_PAYLOAD = 1000
DELIMITER = b'\xFF\x00\xFF' 

def chunk_data(data, chunk_size):
  """Chunks data into smaller pieces."""
  for i in range(0, len(data), chunk_size):
    yield data[i:i + chunk_size]
    
def append_zeros(frame_data, frame_length,newarray):
    expected_bytes = frame_length
    received_bytes = len(frame_data)
    missing_bytes = expected_bytes - received_bytes 

    if missing_bytes > 0:
        zero_padding = np.zeros(missing_bytes, dtype=np.float32) 
        frame_data = np.concatenate((newarray, zero_padding))

    return newarray
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, help='0 for sending, 1 for receiving.')
    args = parser.parse_args()

    data = []
    for i in range(10):
        dummy_data = np.random.rand(1, 32, 32, 10)
        data.append(dummy_data)
        
        

    if args.mode == 0:
        # Sender
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_sock:  # UDP socket 
            for j in data:
                image_bytes = j.tobytes() + DELIMITER
                for chunk in chunk_data(image_bytes, MAX_PAYLOAD):
                    s_sock.sendto(chunk, (host, port))
                
                
    elif args.mode == 1:
        # Receiver
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_sock:
            s_sock.bind((host, port))
            buffer = b''
            while True:
                recv_data, address = s_sock.recvfrom(1024)
                buffer += recv_data
                if DELIMITER in buffer:
                    imagebytes, _, buffer = buffer.partition(DELIMITER)
                image_array= np.frombuffer(imagebytes, dtype=np.float32)
                image_array = image_array.reshape(1, 32, 32, 10)  
                print(image_array)  
                