import cv2
import numpy as np
import random
import os

def find_nal_units(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()

    nal_units = []
    start_codes = [b'\x00\x00\x00\x01', b'\x00\x00\x01']
    start_idx = 0

    while True:
        next_start_idx = min((data.find(start_code, start_idx + 1) for start_code in start_codes if data.find(start_code, start_idx + 1) != -1), default = -1)
        if next_start_idx == -1:
            # "Last" NAL unit, so just add the remaining data
            nal_units.append(data[start_idx:])
            break
        else:
            # Add the next NAL unit AKA the data between the current start code index and the next start code index
            nal_units.append(data[start_idx:next_start_idx])
            start_idx = next_start_idx
        
    return nal_units


def reconstruct_video(nal_units, output_path):
    """
    Reconstructs the video stream from NAL units and saves it to a file.
    
    :param nal_units: A list of NAL units.
    :param output_path: Path where the reconstructed video will be saved.
    """
    with open(output_path, 'wb') as f:
        for unit in nal_units:
            f.write(unit)


def drop_nal_units(nal_units, drop_probability=0.1):
    """
    Simulates packet loss by dropping some NAL units.
    
    :param nal_units: A list of NAL units (each NAL unit is a bytearray).
    :param drop_probability: Probability of dropping a single NAL unit.
    :return: A list of NAL units after simulating packet loss.
    """
    result_units = []
    for unit in nal_units:
        if random.random() > drop_probability:
            result_units.append(unit)
        else:
            print("Dropping a NAL unit")
    return result_units

# file_path = os.path.join('video_data', 'diving', 'diving4.avi')
file_path = os.path.join('compressed_videos_output', 'diving4.mp4')
output_path = "reconstructed_video.h264"

# Parse NAL units
nal_units = find_nal_units(file_path)

# Simulate packet loss
dropped_units = drop_nal_units(nal_units, drop_probability=0.1)

# Reconstruct and save the video
reconstruct_video(dropped_units, output_path)



