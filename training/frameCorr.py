import argparse
from rateless_ae_imagenet import prepare_labels, extract_info_of_dataset, ae_model_loader, def_value_1, prepare_data_AE
from utils.utils_model_pnc import FrameCorr, PNC_Decoder
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.tf_helper import ModelState, MetricLogger
import numpy as np
from collections import defaultdict

def create_image_paths(img_folder, data_info, slice_info, prev_frames, isTest = False):
    img_paths = []
    for i, class_name in enumerate(data_info):
        # start, end = slice_info[i][0], slice_info[i][1]
        for j in slice_info[i]:
            for k in range(1 if isTest else prev_frames + 1, data_info[class_name][str(j)] + 1):
                curr_paths = []
                for l in range(k - prev_frames, k + 1):
                    if l < 1:
                        curr_paths.append("None")
                    else:
                        curr_paths.append(img_folder + "/" + class_name + "_" + str(j) + "_" + str(l).zfill(3) + ".jpg")
                img_paths.append(curr_paths)
    return img_paths

def get_info_from_image_name(image_name):
    frame_no = -1
    video = ''
    underscore_encountered = False
    unicode_chars = tf.strings.unicode_split(image_name, "UTF-8")
    for i in range(tf.strings.length(image_name) - 1, -1, -1):
        char = tf.strings.substr(image_name, pos=i, len=1)
        if char == "_" and not underscore_encountered:
            underscore_encountered = True
            frame_no = tf.strings.to_number(tf.strings.substr(image_name, pos=i+1, len=tf.strings.length(image_name) - tf.constant(4) - (i + 1)), out_type=tf.int32)
            video = tf.strings.substr(image_name, pos = 0, len = i + 1)
            break
    return video, frame_no

def prepare_data_PNC_decoder(img_folder, data_info, class_to_video, prevFrames, encoder, img_size, args, frame_corr):
    train, val, test = [], [], []
    for class_name in data_info:
        number_of_videos = len(data_info[class_name])
        train_no = int(0.6 * number_of_videos)
        val_no = int(0.25 * number_of_videos)
        test_no = number_of_videos - train_no - val_no
        
        train.append(class_to_video[class_name][:train_no])
        val.append(class_to_video[class_name][train_no : train_no + val_no])
        test.append(class_to_video[class_name][train_no + val_no:])
    img_height, img_width = img_size[0], img_size[1]
    # step 1
    train_img_paths = create_image_paths(img_folder, data_info, train, prevFrames)
    val_img_paths = create_image_paths(img_folder, data_info, val, prevFrames)
    test_img_paths = create_image_paths(img_folder, data_info, test, prevFrames, True)

    train_img_paths = tf.constant(train_img_paths)
    val_img_paths = tf.constant(val_img_paths)
    test_img_paths = tf.constant(test_img_paths)

    # # step 2: create a dataset returning slices of `filenames`
    train_dataset_pnc_decoder = tf.data.Dataset.from_tensor_slices((train_img_paths))
    val_dataset_pnc_decoder = tf.data.Dataset.from_tensor_slices((val_img_paths))
    test_dataset_pnc_decoder = tf.data.Dataset.from_tensor_slices((test_img_paths))

    def read_image(filename):
        # print(filename)
        if filename == "None":
            return tf.zeros(shape=(img_height, img_width, 3))
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image /= 255.0
        image_resized = tf.image.resize(image, (img_height, img_width))
        return image_resized
    
    def get_data_for_pnc_decoder(image_files):
        images = []
        for i in range(prevFrames + 1):
            image = read_image(image_files[i])
            compressed_data = encoder(tf.expand_dims(image, axis=0))[0]
            if tf.reduce_all(tf.equal(image, 0)):
                compressed_data = tf.zeros_like(compressed_data)
            images.append(compressed_data)
        predicted_data = frame_corr(tf.expand_dims(images[:-1], axis = 0))
        images = tf.concat((predicted_data, tf.expand_dims(images[-1], axis = 0)), axis = 0)
        return images, image

    train_dataset_pnc_decoder = train_dataset_pnc_decoder.map(lambda x: get_data_for_pnc_decoder(x)).batch(args.batch_size)
    val_dataset_pnc_decoder = val_dataset_pnc_decoder.map(lambda x: get_data_for_pnc_decoder(x)).batch(args.batch_size)
    
    return train_dataset_pnc_decoder, val_dataset_pnc_decoder

def prepare_data_FrameCorr(img_folder, data_info, class_to_video, prevFrames, encoder, img_size, args):
    train, val, test = [], [], []
    for class_name in data_info:
        number_of_videos = len(data_info[class_name])
        train_no = int(0.6 * number_of_videos)
        val_no = int(0.25 * number_of_videos)
        test_no = number_of_videos - train_no - val_no
        
        train.append(class_to_video[class_name][:train_no])
        val.append(class_to_video[class_name][train_no : train_no + val_no])
        test.append(class_to_video[class_name][train_no + val_no:])
    img_height, img_width = img_size[0], img_size[1]
    # step 1
    train_img_paths = create_image_paths(img_folder, data_info, train, prevFrames)
    val_img_paths = create_image_paths(img_folder, data_info, val, prevFrames)
    test_img_paths = create_image_paths(img_folder, data_info, test, prevFrames, True)

    train_img_paths = tf.constant(train_img_paths)
    val_img_paths = tf.constant(val_img_paths)
    test_img_paths = tf.constant(test_img_paths)

    # # step 2: create a dataset returning slices of `filenames`
    train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_img_paths))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_img_paths))

    def read_image(filename):
        # print(filename)
        if filename == "None":
            return tf.zeros(shape=(img_height, img_width, 3))
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image /= 255.0
        image_resized = tf.image.resize(image, (img_height, img_width))
        return image_resized

    # Define function to get predecessor images
    def get_predecessor_images(image_files):
        images = []
        for i in range(prevFrames + 1):
            image = read_image(image_files[i])
            compressed_data = encoder(tf.expand_dims(image, axis=0))[0]
            if tf.reduce_all(tf.equal(image, 0)):
                compressed_data = tf.zeros_like(compressed_data)
            images.append(compressed_data)
        return images[:-1], images[-1]
    
    def get_predecessor_images_test(image_files):
        pred_images = get_predecessor_images(image_files)
        return image_files[-1], pred_images[0], pred_images[1]

    train_dataset = train_dataset.map(lambda x: get_predecessor_images(x)).batch(args.batch_size)
    # train_dataset = train_dataset.map(lambda x : x)#.batch(args.batch_size)
    val_dataset = val_dataset.map(lambda x: (get_predecessor_images(x))).batch(args.batch_size)
    test_dataset = test_dataset.map(lambda x: (get_predecessor_images_test(x)))#.batch(args.batch_size)
    # train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, 35000, 5000, 10000)
    
    return train_dataset, val_dataset, test_dataset

def check_valid_framecorr_data(predecessor_images):
    for image in predecessor_images:
        if tf.reduce_all(tf.equal(image, 0)).numpy():
            return False
    return True

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0, help='0 for training, 1 for inference.')
    parser.add_argument('--prev_frames', type=int, default=2)
    parser.add_argument('--input_size', type=int, default=224, help='Size of the input images.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--model', type=str, default='PNC', help='Folder to save the model.')
    parser.add_argument('--ae_path', type=str, default='saved_models/default', help='Folder to save the model.')
    parser.add_argument('--framecorr_path', type=str, default='saved_models/frame_corr', help='Folder to save the model.')
    parser.add_argument('--pnc_decoder_path', type=str, default='saved_models/pnc_decoder', help='Folder to save the model.')
    parser.add_argument('--joint_path', type=str, default='saved_models/default', help='Folder to save the model.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of training.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training.')
    parser.add_argument('--out_size', type=int, default=10, help='Compressed frame size.')
    parser.add_argument('--tail_drop_features', type=int, default=0, help='Number of features to drop from tail')
    parser.add_argument('--apply_framecorr', type=int, default=1, help='1 to enable framecorr, 0 to disable')
    args = parser.parse_args()

    # # prepare labels
    image_to_label_map = prepare_labels()#[:50000]
    data_info, class_to_video = extract_info_of_dataset(image_to_label_map)
    # print(image_to_label_map)
    # exit()

    # # prepare data
    img_folder = "../new_video_frames_dataset"
    assert(os.path.exists(img_folder))

    input_size = (args.input_size, args.input_size)

    ae_path = args.ae_path
    joint_path = args.joint_path
    ModelObject = ae_model_loader(args.model)
    model = ModelObject(out_size=args.out_size).asym_ae(tailDrop=False)
    if os.path.exists(ae_path):
        model_load_path = tf.train.latest_checkpoint(ae_path)
        if model_load_path is not None:
            model.load_weights(model_load_path)
    
    encoder, decoder = get_encoder_decoder(model)

    train_dataset, val_dataset, test_dataset = prepare_data_FrameCorr(img_folder, data_info, class_to_video, args.prev_frames, encoder, input_size, args)
    _, _, ae_test_dataset = prepare_data_AE(img_folder, data_info, class_to_video, args)
    
    # Get the first batch from the train_dataset
    # first_batch = next(iter(val_dataset))

    # # Extract predecessor_images and successor_image from the first batch
    # predecessor_images, successor_image = first_batch

    # # Get the shape of predecessor_images and successor_image
    # print("Shape of predecessor_images:", predecessor_images.shape)
    # print("Shape of successor_image:", successor_image.shape)

    frame_corr = FrameCorr(args.prev_frames, args.out_size).frame_corr()

    frame_corr.summary()

    if args.mode == 0:
        last_model_path = os.path.join(args.framecorr_path, "last_checkpoint")
        best_model_path = os.path.join(args.framecorr_path, "best_checkpoint")

        checkpoint = ModelCheckpoint(
                best_model_path+"_epoch_{epoch:03d}_val_loss_{val_loss:.4f}", 
                monitor='val_loss', 
                verbose=1, 
                save_best_only=True, 
                save_weights_only=True, 
                mode='min'
            )
        checkpoint_last = ModelCheckpoint(
            last_model_path, 
            monitor='val_loss', 
            verbose=1, 
            save_weights_only=True, 
        )

        model_state = ModelState(os.path.join(args.framecorr_path,"state.json"), ['val_loss'], [tf.math.less])
        metricloggercallback = MetricLogger(monitor='val_loss', monitor_op=tf.math.less, best=np.inf)

        if  model_state.state['best_values']:
            checkpoint.best = model_state.state['best_values']['val_loss']
            metricloggercallback.best = model_state.state['best_values']['val_loss']
        def lr_step_decay(epoch, lr):
            if epoch != 0 and epoch % 10 == 0:
                return lr*0.3
            return lr
        
        train_dataset.cache()
        val_dataset.cache()

        frame_corr.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss="MSE")

        frame_corr.fit(
            train_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1),
                checkpoint, 
                checkpoint_last,
                model_state,
                metricloggercallback, 
            ],
            validation_data=val_dataset,
            initial_epoch=model_state.state['epoch_count']-1
        )
    
    elif args.mode == 1:
        if os.path.exists(args.framecorr_path):
            model_load_path = tf.train.latest_checkpoint(args.framecorr_path)
            if model_load_path is not None:
                frame_corr.load_weights(model_load_path)
        pnc_decoder = PNC_Decoder(args.prev_frames, input_size, args.out_size).decoder(tailDrop=False)
        if os.path.exists(args.pnc_decoder_path):
            model_load_path = tf.train.latest_checkpoint(args.pnc_decoder_path)
            if model_load_path is not None:
                pnc_decoder.load_weights(model_load_path)
        mse_per_video = defaultdict(def_value_1)
        frame_per_video = defaultdict(def_value_1)
        print(len(ae_test_dataset), len(test_dataset))
        for test_ae, test_framecorr in zip(ae_test_dataset, test_dataset):
            file, predecessor_images, curr_image = test_framecorr
            is_framecorr = check_valid_framecorr_data(predecessor_images)
            _, org_image, _ = test_ae
            #encoded_data is of dimension (1, 32, 32, 10). It is one frame's encoding. This will be sent over the network
            # print(predecessor_images.shape)
            # print(curr_image.shape)
            if is_framecorr and args.apply_framecorr == 1:
                predicted_encoding_from_framecorr = frame_corr.predict(tf.expand_dims(predecessor_images, axis = 0))
                if args.tail_drop_features != 0:
                    received_data = tf.concat([tf.slice(curr_image, [0, 0, 0], [curr_image.shape[0], curr_image.shape[1], args.out_size - args.tail_drop_features]), tf.slice(predicted_encoding_from_framecorr[0], [0, 0, args.out_size - args.tail_drop_features], [curr_image.shape[0], curr_image.shape[1], args.tail_drop_features])], axis = -1)
                else:
                    received_data = curr_image
                # print(received_data.shape, tf.concat([tf.expand_dims(received_data, axis = 0), tf.expand_dims(tf.zeros_like(received_data), axis = 0)], axis = 0).shape)
                decoded_data = pnc_decoder.predict(tf.expand_dims(tf.concat([tf.expand_dims(received_data, axis = 0), tf.expand_dims(tf.zeros_like(received_data), axis = 0)], axis = 0), axis = 0))    
            else:
                if args.tail_drop_features != 0:
                    received_data = tf.concat([tf.slice(curr_image, [0, 0, 0], [curr_image.shape[0], curr_image.shape[1], args.out_size - args.tail_drop_features]), tf.zeros(shape=[curr_image.shape[0], curr_image.shape[1], args.tail_drop_features])], axis = -1)
                else:
                    received_data = curr_image
                decoded_data = decoder.predict(tf.expand_dims(received_data, axis = 0))

            mse = tf.reduce_sum(tf.math.square(decoded_data - org_image), axis=None)
            video = "".join(file.numpy().decode("utf-8").split("/")[-1][:-4].split("_")[:-1])
            mse_per_video[video] += mse
            frame_per_video[video] += 1
        
        for video in mse_per_video:
            mse_per_video[video] /= frame_per_video[video]
            print(mse_per_video[video].numpy())
    
    elif args.mode == 2:
        if os.path.exists(args.framecorr_path):
            model_load_path = tf.train.latest_checkpoint(args.framecorr_path)
            if model_load_path is not None:
                frame_corr.load_weights(model_load_path)
        train_pnc_decoder, val_pnc_decoder = prepare_data_PNC_decoder(img_folder, data_info, class_to_video, args.prev_frames, encoder, input_size, args, frame_corr)
        pnc_decoder = PNC_Decoder(args.prev_frames, input_size, args.out_size).decoder()

        last_model_path = os.path.join(args.pnc_decoder_path, "last_checkpoint")
        best_model_path = os.path.join(args.pnc_decoder_path, "best_checkpoint")

        checkpoint = ModelCheckpoint(
                best_model_path+"_epoch_{epoch:03d}_val_loss_{val_loss:.4f}", 
                monitor='val_loss', 
                verbose=1, 
                save_best_only=True, 
                save_weights_only=True, 
                mode='min'
            )
        checkpoint_last = ModelCheckpoint(
            last_model_path, 
            monitor='val_loss', 
            verbose=1, 
            save_weights_only=True, 
        )

        model_state = ModelState(os.path.join(args.pnc_decoder_path,"state.json"), ['val_loss'], [tf.math.less])
        metricloggercallback = MetricLogger(monitor='val_loss', monitor_op=tf.math.less, best=np.inf)

        if  model_state.state['best_values']:
            checkpoint.best = model_state.state['best_values']['val_loss']
            metricloggercallback.best = model_state.state['best_values']['val_loss']
        def lr_step_decay(epoch, lr):
            if epoch != 0 and epoch % 10 == 0:
                return lr*0.3
            return lr
        
        train_pnc_decoder.cache()
        val_pnc_decoder.cache()

        pnc_decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss="MSE")

        pnc_decoder.fit(
            train_pnc_decoder,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1),
                checkpoint, 
                checkpoint_last,
                model_state,
                metricloggercallback, 
            ],
            validation_data=val_pnc_decoder,
            initial_epoch=model_state.state['epoch_count']-1
        )