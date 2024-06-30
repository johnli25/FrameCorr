import argparse
from pathlib import Path
import logging
import os, datetime
from rateless_ae_imagenet import prepare_labels, extract_info_of_dataset, prepare_data_AE, ae_model_loader
from utils.utils_preprocess import boolean_string
import tensorflow as tf
import base64
import socket
import time
import sys
import struct
import numpy as np
from collections import defaultdict
import sys
import struct
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import copy

np.set_printoptions(threshold=np.inf)

host = "172.22.153.20"
port = 50013
DELIMITER = b'\xFF\x00\xFF' 
END_FRAME_DELIMITER = b'\x11\xEE\x11' 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, help='0 for sending, 1 for receiving.')
    parser.add_argument('--model', type=str, default='PNC', help='Folder to save the model.')
    parser.add_argument('--input_size', type=int, default=224, help='Size of the input images.')
    parser.add_argument('--ae_path', type=str, default='saved_models/default', help='Folder to save the model.')
    parser.add_argument('--joint_path', type=str, default='saved_models/default', help='Folder to save the model.')
    parser.add_argument('--log_save_path', type=str, default='./', help='Path to save the logs.')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--restart_training', type=boolean_string, default=False, help='If start from scratch.')
    args = parser.parse_args()

    # Logging
    log_save_path = args.log_save_path
    Path(log_save_path).mkdir(parents=True, exist_ok=True)

    logging.basicConfig( level=logging.INFO, 
                format='[%(asctime)s]%(levelname)s|%(module)s|%(funcName)s: %(message)s',
                handlers=[
                    logging.FileHandler(filename=os.path.join(log_save_path, '_training_results_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.log'), mode='w'),
                    logging.StreamHandler()
                ]
    )
    logging.info(str(args))

    # # prepare labels
    image_to_label_map = prepare_labels()#[:50000]
    data_info, class_to_video = extract_info_of_dataset(image_to_label_map)
    print("data info", data_info)
    print("class to video", class_to_video)

    img_folder = "../new_video_frames_dataset"
    assert(os.path.exists(img_folder))

    # img_paths = sorted(glob.glob(img_folder+'/*'))[:50000]
    # logging.info("Number of imgs in the folder: {}".format(len(img_paths)))
    input_size = (args.input_size, args.input_size)

    ae_train_dataset, ae_val_dataset, ae_test_dataset = prepare_data_AE(img_folder, data_info, class_to_video, img_size=input_size, args=args)
    print("ae_train_dataset", ae_train_dataset)
    print("ae_val_dataset", ae_val_dataset)
    print("ae_test_dataset", ae_test_dataset)
    # cls_train_dataset, cls_val_dataset, cls_test_dataset = prepare_data_CLS(img_paths, gts, img_size=input_size)

    # Print the file paths from the test dataset
    # for file_path, input_image, target_image in ae_test_dataset:
    #     print(file_path.numpy().decode('utf-8'))

    ae_path = args.ae_path
    joint_path = args.joint_path
    ModelObject = ae_model_loader(args.model)
    model = ModelObject(out_size=10).asym_ae(tailDrop=False if args.mode != 19 else False)

    if (not args.restart_training) and os.path.exists(ae_path):
        logging.info("<<<<<<<<<<<<<<<<<< LOAD PREVIOUS MODEL >>>>>>>>>>>>>>>>>>>>>>>>")
        model_load_path = tf.train.latest_checkpoint(ae_path)
        print("model_load_path", model_load_path)
        if model_load_path is not None:
            logging.info("    >>> restored from {}".format(model_load_path))
            model.load_weights(model_load_path)
    else:
        logging.info("<<<<<<<<<<<<<<<<<< TRAIN WITH NEW MODEL >>>>>>>>>>>>>>>>>>>>>>>>")
        
    def get_encoder_decoder(autoencoder):
        decoder_input_index = None
        layerName = 'decoder'
        for idx, layer in enumerate(autoencoder.layers):
            if layer.name == layerName:
                decoder_input_index = idx
                break

        print("decoder input idx: ", decoder_input_index)
                
        # encoder = keras.Model(autoencoder.input, autoencoder.get_layer(name = 'encoder').output, name='encoder1')
        encoder = tf.keras.Sequential(name='encoder1')
        for layer in autoencoder.layers[:decoder_input_index]:
            encoder.add(layer)
        decoder = tf.keras.Sequential(name='decoder1')
        for layer in autoencoder.layers[decoder_input_index:]:
            decoder.add(layer)

        return encoder, decoder
    
    ### NOTE: Above is PNC/Shehab's code and below is Deepak's Networking Code.

    def get_encoder_decoder_from_joint(ae_joint):
        ae = tf.keras.Sequential(name='ae_extracted')
        for layer in ae_joint.layers[1].layers[:4]:
            ae.add(layer)
        ae.summary()
        encoder, decoder = get_encoder_decoder(ae)

        return encoder, decoder
    
    def encoded_data_send():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
            s_sock.connect((host,port))
            s_sock.sendall(data_b64)
        
    def decoded_data_recv(host,port,data_b64):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
            s_sock.bind((host, port))
            s_sock.listen()
            iot_sock, address = s_sock.accept()
            with iot_sock:
                while True:
                    recv_data = iot_sock.recv()
                    if not recv_data:
                        break 

    def chunk_data(data, chunk_size):
        """Chunks data into smaller pieces."""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
                
    def get_object_size(obj):
        return sys.getsizeof(obj)
    
    def calculate_SE_per_frame(A,B):
        A, B = np.array(A), np.array(B)
        return (np.sum((A - B) ** 2))
    
    def recvall(sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: 
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf
    
    def get_object_size(obj):
        return sys.getsizeof(obj)
             
    def diff_between_tf(x,y):
        return x - y[:,None]
    
    def partition_frame(frame,start,end):
        return frame[:,:,:,start:end]

    def zero_padding(frame, target_shape):
        pad_needed = target_shape[-1] - frame.shape[-1]
        padding = ((0, 0), (0, 0), (0, 0), (0, pad_needed)) 
        padded_frame = np.pad(frame, padding, mode='constant', constant_values=0)
        print("padded_frame", padded_frame.shape)
        return padded_frame 
    
    def optimize_frame_end(throughput):
        frame_end = 0
        if throughput <= 500000:
            frame_end = 5
        elif 500000 <= throughput <= 1000000:
            frame_end = 7
        else:
            frame_end = 10
        return frame_end
    
    def measure_throughput(start_time, data_received):
        end_time = time.time()
        #print("time:",end_time - start_time)
        throughput = data_received / (end_time - start_time)  # bits per second
        print(throughput)
        return int(throughput)
    
    encoder, decoder = get_encoder_decoder(model) 
    deadlines = [200] 
    first_one_flag = True
    
    if args.mode == 2: # no network required-used for running locally!
        random.seed(314156)
        frame_to_data_transmission = {} # temp placeholder dict for transmission

        for file, input_image, output_image in ae_test_dataset:
            video_img_frame = "".join(file.numpy().decode("utf-8").split("/")[-1][:-4]) + ".jpg"
            video = "".join(video_img_frame[:-4].split("_")[:-1])
            # print("video_img_frame", video_img_frame)
            # print("video", video)
            encoded_data = np.array(encoder.predict(tf.expand_dims(input_image, axis=0))) # expand_dims() adds a (batch) dimension at the 0th index of 1. .precict() expects a batch of images, so we need to add a batch dimension
            encoded_data_pnc = np.copy(encoded_data)
            # print(encoder.layers[0].summary()) # print the summary of the first Functional layer (there's only 1 anyway LOL) of the encoder; this consists of multiple other sub-layers (refer to PNC paper)
            random_num_features_keep = random.randint(1, 10) # 3
            encoded_data_pnc[:,:,:,random_num_features_keep:] = 0 # PNC: zero out the last x features
            frame_to_data_transmission[video_img_frame] = encoded_data_pnc            

        PNC_received_directory = "PNC_received_imgs"
        FrameCorr_received_directory = "FrameCorr_received_imgs"
        input_image_directory = "PNC_FrameCorr_input_imgs"
        output_image_directory = "PNC_FrameCorr_output_imgs"
        if not os.path.exists(output_image_directory):
            os.makedirs(output_image_directory)
        if not os.path.exists(input_image_directory):
            os.makedirs(input_image_directory)
        if not os.path.exists(PNC_received_directory):
            os.makedirs(PNC_received_directory)
        if not os.path.exists(FrameCorr_received_directory):
            os.makedirs(FrameCorr_received_directory)

        # NOTE: treat this next code as the receiver/client
        PNC_video_frame_mse, FrameCorr_video_mse = defaultdict(list), defaultdict(list)
        prev_frame_enc_data, prev_video = None, ""
        for file, input_image, output_image in ae_test_dataset:
            video_img_frame = "".join(file.numpy().decode("utf-8").split("/")[-1][:-4]) + ".jpg"
            video = "".join(video_img_frame[:-4].split("_")[:-1])
            enc_data = frame_to_data_transmission[video_img_frame]

            # fill missing/zeroed out features with previous and future frames
            if prev_video != video:
                print("prev_video and next video: ", prev_video, video)
            if prev_frame_enc_data is not None and prev_video == video:
                for i in range(enc_data.shape[-1]):
                    if np.all(enc_data[0, :, :, i] == 0):
                        if i == 0:
                            enc_data[0, :, :, i] = prev_frame_enc_data[0, :, :, i + 1]
                        elif i == enc_data.shape[-1] - 1:
                            enc_data[0, :, :, i] = prev_frame_enc_data[0, :, :, i - 1]
                        else:
                            enc_data[0, :, :, i] = (prev_frame_enc_data[0, :, :, i - 1] + prev_frame_enc_data[0, :, :, i + 1]) / 2

            decoded_data = decoder.predict(enc_data) 

            prev_frame_enc_data, prev_video = enc_data, video
            # print(decoder.layers[0].summary()) # print the summary of the first Functional layer (there's only 1 anyway LOL) of the decoder; this consists of multiple other sub-layers (refer to PNC paper)

            plt.imsave("{}/{}".format(PNC_received_directory, video_img_frame), decoded_data[0,:,:,:])
            # plt.imsave("{}/{}".format(FrameCorr_received_directory, video_img_frame), decoded_data[0,:,:,:])

            pnc_mse = calculate_SE_per_frame(output_image, decoded_data)
            # framecorr_mse = calculate_SE_per_frame(output_image, decoded_data)
            PNC_video_frame_mse[video].append(pnc_mse)
            # FrameCorr_video_mse[video].append(framecorr_mse)

        print("PNC Video Frame MSE")
        for k, v in PNC_video_frame_mse.items():
            # print("Video: {} MSE: {}".format(k, np.mean(v)))
            print(np.mean(v))

        print("FrameCorr Video Frame MSE")
        for k, v in FrameCorr_video_mse.items():
            # print("Video: {} MSE: {}".format(k, np.mean(v)))
            print(np.mean(v))


    if args.mode == 0: # sender
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
            # print("send buffer size", s_sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
            s_sock.connect((host, port))
            for file, input_image, output_image in ae_test_dataset:
                video_img_frame = "".join(file.numpy().decode("utf-8").split("/")[-1][:-4]) + ".jpg"

                if first_one_flag or s_sock.recv(1024).decode().strip() == "ACK":
                    #encoded_data is of dimension (1, 32, 32, 10). It is one frame's encoding. This will be sent over the network
                    encoded_data = np.array(encoder.predict(tf.expand_dims(input_image, axis=0)))
                    feature_end = encoded_data.shape[-1]
                    print(type(encoded_data), encoded_data.shape, feature_end)
                    encoded_data = partition_frame(encoded_data, 0, 10)

                    # <<<<< DEBUG encode/decode without sending! >>>>>
                    # decoded_data = decoder.predict(encoded_data)
                    # received_directory = "PNC_FrameCorr_received_imgs"
                    # if not os.path.exists(received_directory):
                    #     os.makedirs(received_directory)
                    # plt.imsave("{}/{}".format(received_directory, video_img_frame), decoded_data[0,:,:,:])
                    # <<< END DEBUG >>>

                    video_img_frame = "".join(file.numpy().decode("utf-8").split("/")[-1][:-4]) + ".jpg"
                    print("total num_bytes of frame", str(video_img_frame), get_object_size(encoded_data.tobytes())) # should I include .tobytes()?
                    start_time_deadline = time.time()
                    feature_bytes_combined = b''
                    for i in range(feature_end): # LOOP THROUGH FEATURES
                        feature_bytes = partition_frame(encoded_data, i, i+1)
                        print(str(video_img_frame), i)
                        feature_bytes = feature_bytes.tobytes()
                        print("feature_bytes", get_object_size(feature_bytes))
                        delta_timeline = time.time() - start_time_deadline
                        print("current time elapsed", delta_timeline)
                        if delta_timeline >= deadlines[0] or i > 7: # or i > random.randint(2, feature_end):
                            s_sock.sendall(feature_bytes + END_FRAME_DELIMITER)
                            break
                        else:
                            s_sock.sendall(feature_bytes + DELIMITER)

                    # s_sock.sendall(END_FRAME_DELIMITER)
                    first_one_flag = False

            s_sock.close()
            print("socket_closed")   

    elif args.mode == 1: # receiver
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_sock:
            s_sock.bind((host, port))
            s_sock.listen()
            sock_conn, sock_addr = s_sock.accept()
            iter = 0
            metrics, video_MSE = defaultdict(list), defaultdict(list) 
            with sock_conn:
                total_start_time = time.time()
                for file, input_image, output_image in ae_test_dataset:
                    video_img_frame = "".join(file.numpy().decode("utf-8").split("/")[-1])
                    video = "".join(video_img_frame[:-4].split("_")[:-1])
                    image_array = np.zeros((1, 32, 32, 10), dtype=np.float32)
                    buffer = b''
                    moveonto_next_frame = False
                    for i in range(10):
                        while not moveonto_next_frame and buffer.find(DELIMITER) == -1:
                            if END_FRAME_DELIMITER in buffer:
                                moveonto_next_frame = True
                                break
                            start_time = time.time()
                            chunk = sock_conn.recv(4096)
                            if not chunk:
                                break
                            buffer += chunk

                        if DELIMITER in buffer:
                            image_bytes, _, buffer = buffer.partition(DELIMITER)
                        elif END_FRAME_DELIMITER in buffer:
                            image_bytes, _, buffer = buffer.partition(END_FRAME_DELIMITER)
                        image_bytes_size = get_object_size(image_bytes)
                        # print("image_bytes_size", image_bytes_size)
                        image_bytes = np.frombuffer(image_bytes, dtype=np.float32)
                        image_bytes = image_bytes.reshape(1, 32, 32, 1)
                        frame_end = image_bytes_size // (1 * 32 * 32 * 4)
                        image_bytes = np.squeeze(image_bytes)
                        # print("image_array", image_array[:,:,:,i] .shape)
                        image_array[:,:,:,i] = image_bytes[:,:]
                        if moveonto_next_frame: break

                    image_array_zp = zero_padding(frame=image_array, target_shape=(1,32,32,10))
                    decoded_data = decoder.predict(image_array_zp)

                    received_directory = "PNC_FrameCorr_received_imgs"
                    if not os.path.exists(received_directory):
                        os.makedirs(received_directory)
                    plt.imsave("{}/{}".format(received_directory, video_img_frame), decoded_data[0,:,:,:])

                    try:
                        sock_conn.send(b"ACK")
                    except socket.error as e:
                        print(e)
                
                print("Total time taken", time.time() - total_start_time)
                s_sock.close()
                print("socket_closed")

                for k, v in video_MSE.items():
                    print("Video: {} MSE: {}".format(k, np.mean(v)))

