import cv2
import numpy as np
import random
import os

# Function to simulate partial loss of encoded information within each frame
def simulate_partial_loss(frame_data, loss_probability=0.1):
    # Convert frame data to byte array
    frame_bytes = bytearray(frame_data)

    # Determine the number of bytes to drop based on loss probability
    num_bytes = len(frame_bytes)
    num_bytes_to_drop = int(loss_probability * num_bytes)

    # Randomly select bytes to drop
    bytes_to_drop = random.sample(range(num_bytes), num_bytes_to_drop)

    # Set the selected bytes to zero
    for byte_index in bytes_to_drop:
        frame_bytes[byte_index] = 0

    # Convert modified byte array back to bytes
    modified_frame_data = bytes(frame_bytes)

    return modified_frame_data

# Function to decode the video from the modified encoded information
def decode_video(encoded_frames):
    decoded_frames = []
    for frame_data in encoded_frames:
        # Decode the frame
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        decoded_frames.append(frame)
    return decoded_frames

# Load the H.264 encoded video
video_path = os.path.join('video_data', 'diving', 'diving7.avi')
# video_path = os.path.join('compressed_video_data', 'diving7.mp4')
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Read all frames and store the encoded data
encoded_frames = []
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Encode the frame using H.264 codec and store the encoded data
    _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    encoded_frames.append(encoded_frame.tobytes())

# Close the video capture object
cap.release()

# Simulate partial loss of encoded information within each frame
loss_probability = 0.1  # 10% loss probability
modified_encoded_frames = []
for frame_data in encoded_frames:
    modified_frame_data = simulate_partial_loss(frame_data, loss_probability)
    modified_encoded_frames.append(modified_frame_data)

# Decode the video from the modified encoded information
decoded_frames = decode_video(modified_encoded_frames)

# Display or process the decoded frames as needed
for frame in decoded_frames:
    cv2.imshow('Decoded Frame', frame)
    cv2.waitKey(30)  # Adjust frame rate as needed

cv2.destroyAllWindows()
