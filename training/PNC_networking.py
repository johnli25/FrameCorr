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
    data_info = extract_info_of_dataset(image_to_label_map)
    # print(data_info)

    # # prepare data
    img_folder = "../new_video_frames_dataset"
    assert(os.path.exists(img_folder))

    # img_paths = sorted(glob.glob(img_folder+'/*'))[:50000]
    # logging.info("Number of imgs in the folder: {}".format(len(img_paths)))
    input_size = (args.input_size, args.input_size)

    # imagenet_utils=imagenetUtils(size=input_size)

    ae_train_dataset, ae_val_dataset, ae_test_dataset = prepare_data_AE(img_folder, data_info, img_size=input_size, args=args)
    # cls_train_dataset, cls_val_dataset, cls_test_dataset = prepare_data_CLS(img_paths, gts, img_size=input_size)

    # for data in ae_test_dataset:
    #     print(data[0] - data[1])
    #     exit()


    ae_path = args.ae_path
    joint_path = args.joint_path
    ModelObject = ae_model_loader(args.model)
    model = ModelObject(out_size=10).asym_ae(tailDrop=True if args.mode != 19 else False)
    if (not args.restart_training) and os.path.exists(ae_path):
        logging.info("<<<<<<<<<<<<<<<<<< LOAD PREVIOUS MODEL >>>>>>>>>>>>>>>>>>>>>>>>")
        model_load_path = tf.train.latest_checkpoint(ae_path)
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

        # print(decoder_input_index)
                
        # encoder = keras.Model(autoencoder.input, autoencoder.get_layer(name = 'encoder').output, name='encoder1')
        encoder = tf.keras.Sequential(name='encoder1')
        for layer in autoencoder.layers[:decoder_input_index]:
            encoder.add(layer)
        decoder = tf.keras.Sequential(name='decoder1')
        for layer in autoencoder.layers[decoder_input_index:]:
            decoder.add(layer)

        return encoder, decoder

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
                    
    host = "172.22.154.247"
    port = 50000
    MAX_PAYLOAD = 500
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
            
    encoder, decoder = get_encoder_decoder(model)
    # if args.mode == 0:
        
    #     i = 0 
    #     for file, input_image, output_image in ae_test_dataset:
    #         encoded_data = encoder.predict(tf.expand_dims(input_image, axis=0))
    #         encoded_data_bytes = encoded_data.tobytes()
    #         i += 1
    #         with open('sendfiles/encoded_data'+ str(i) +'.bin', 'wb') as f:
    #             f.write(encoded_data_bytes)
    #encode the data and send them over the network
    if args.mode == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s_sock:        
            for file, input_image, output_image in ae_test_dataset:
                #encoded_data is of dimension (1, 32, 32, 10). It is one frame's encoding. This will be sent over the network
                encoded_data = encoder.predict(tf.expand_dims(input_image, axis=0))
                print(encoded_data.shape,type(encoded_data))
                image_bytes = encoded_data.tobytes() + DELIMITER
                for chunk in chunk_data(image_bytes, MAX_PAYLOAD):
                    time.sleep(0.01)
                    try:
                        s_sock.sendto(chunk, (host, port))
                    except BrokenPipeError:
                        print("Broken Pipe detected")
        s_sock.sendto(b"closetheconnection", (host, port))                 
        s_sock.close()
    elif args.mode == 1:
        #receive the encoded_data of one frame from the sender. The following code assumes the dimension of the encoded_data is the same
        # as while it was sent: (1, 32, 32, 10)
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
                    decoded_data = decoder.predict(image_array)
                    print(decoded_data.shape)