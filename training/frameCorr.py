import argparse
from rateless_ae_imagenet import prepare_labels, extract_info_of_dataset, ae_model_loader
from utils.utils_model_pnc import FrameCorr
import os
import tensorflow as tf

def create_image_paths(img_folder, data_info, slice_info, prev_frames):
    img_paths = []
    for i, class_name in enumerate(data_info):
        # start, end = slice_info[i][0], slice_info[i][1]
        for j in slice_info[i]:
            for k in range(prev_frames + 1, data_info[class_name][str(j)] + 1):
                curr_paths = []
                for l in range(k - prev_frames, k + 1):
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
    test_img_paths = create_image_paths(img_folder, data_info, test, prevFrames)

    train_img_paths = tf.constant(train_img_paths)
    val_img_paths = tf.constant(val_img_paths)
    test_img_paths = tf.constant(test_img_paths)

    # # step 2: create a dataset returning slices of `filenames`
    train_dataset = tf.data.Dataset.from_tensor_slices(train_img_paths)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_img_paths)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_img_paths))

    def read_image(filename):
        print(filename)
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
            compressed_data = encoder(tf.expand_dims(image, axis=0))
            images.append(compressed_data)
        return images
        # Get index of current image file
        # video, frame_no = get_info_from_image_name(image_file)
        # print(video, frame_no)
        # Get index of predecessor image files
        # if frame_no > prevFrames:
        #     print("Inside", frame_no)
        #     images = []
        #     for i in range(frame_no - prevFrames, frame_no + 1, 1):
        #         file_name = video + str(i) + ".jpg"
        #         image_i = read_image(file_name)
        #         images.append(image_i)
        #     return images
        # else:
        #     return [None for _ in range(prevFrames)]

    train_dataset = train_dataset.map(lambda x: get_predecessor_images(x)).batch(args.batch_size)
    # train_dataset = train_dataset.map(lambda x : x)#.batch(args.batch_size)
    val_dataset = val_dataset.map(lambda x: (get_predecessor_images(x))).batch(args.batch_size)
    # test_dataset = test_dataset.map(_parse_function_ae_test)#.batch(args.batch_size)
    # train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, 35000, 5000, 10000)
    
    return train_dataset, val_dataset#, test_dataset

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
    parser.add_argument('--prev_frames', type=int, default=2)
    parser.add_argument('--input_size', type=int, default=224, help='Size of the input images.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--model', type=str, default='PNC', help='Folder to save the model.')
    parser.add_argument('--ae_path', type=str, default='saved_models/default', help='Folder to save the model.')
    parser.add_argument('--joint_path', type=str, default='saved_models/default', help='Folder to save the model.')
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
    model = ModelObject(out_size=10).asym_ae(tailDrop=False)
    if os.path.exists(ae_path):
        model_load_path = tf.train.latest_checkpoint(ae_path)
        if model_load_path is not None:
            model.load_weights(model_load_path)
    
    encoder, decoder = get_encoder_decoder(model)

    train_dataset, val_dataset = prepare_data_FrameCorr(img_folder, data_info, class_to_video, args.prev_frames, encoder, input_size, args)

    frame_corr = FrameCorr(10, args.prev_frames).frame_corr()

    frame_corr.summary()