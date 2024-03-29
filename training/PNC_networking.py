import argparse
from pathlib import Path
import logging
import os, datetime
from rateless_ae_imagenet import prepare_labels, extract_info_of_dataset, prepare_data_AE, ae_model_loader
from utils.utils_preprocess import boolean_string
import tensorflow as tf

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
    encoder, decoder = get_encoder_decoder(model)
    #encode the data and send them over the network
    if args.mode == 0:        
        for file, input_image, output_image in ae_test_dataset:
            #encoded_data is of dimension (1, 32, 32, 10). It is one frame's encoding. This will be sent over the network
            encoded_data = encoder.predict(tf.expand_dims(input_image, axis=0))
            
    elif args.mode == 1:
        #receive the encoded_data of one frame from the sender. The following code assumes the dimension of the encoded_data is the same
        # as while it was sent: (1, 32, 32, 10)
        decoded_data = decoder.predict(encoded_data)
        print(decoded_data.shape)