import argparse
from utils.utils_imagenet import * 
from utils.utils_model_pnc import * 
from utils.utils_preprocess import boolean_string
from utils.tf_helper import ModelState, MetricLogger, AllLogger, customize_lr_sccheduler
import os
import glob
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging
from pathlib import Path
import datetime
from collections import defaultdict
# import tensorflow_model_optimization as tfmot
# quantize_model = tfmot.quantization.keras.quantize_model

def def_value_1():
    return 0

def def_value(): 
    return defaultdict(def_value_1)

def extract_info_of_dataset(image_to_label):
    video_info = defaultdict(def_value)
    class_to_video = defaultdict(list)
    for image_name in image_to_label:
        image_name = image_name[:-4]
        underscore_encountered = False
        j = -1
        for i in range(len(image_name) - 1, -1, -1):
            if image_name[i] == "_" and not underscore_encountered:
                frame_no = image_name[i+1:]
                underscore_encountered = True
                j = i
            elif image_name[i] == "_":
                video_no = image_name[i+1:j]
                class_name = image_name[:i]
                break
        # class_name, video_no, frame_no = image_name.split("_")
        video_info[class_name][video_no] += 1
        if video_no not in class_to_video[class_name]:
            class_to_video[class_name].append(video_no)
    return video_info, class_to_video # class_to_video[class_name] is a list of video numbers

def prepare_labels():
    # label
    # label_path = 'data/ImageNetLabels.txt'
    # with open(label_path, "r", encoding="UTF8") as lbfile:
    #     labels = lbfile.read().splitlines()

    # ground truths
    gt_path = 'data/new_video_numidx_labels.txt'
    with open(gt_path,"r") as lbfile:
        lines = lbfile.readlines()
        gts = {}
        for x in lines:
            gts[x.split(' ')[0]] = int(x.split(' ')[1].splitlines()[0])
    # gts = np.array(gts) + 1
    # gts = np.array(gts)
    # print(gts)
    return gts

def prepare_clssifier(model_folder = "../image_classifiers/", model_name="efficientnet_b0_classification_1"):
    model_path = os.path.join(model_folder, model_name)
    classifier = tf.keras.models.load_model(model_path)
    classifier._name = model_name

    # classifier.build([None, img_height, img_width, 3])
    classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')])
    classifier.trainable = False
    classifier.summary()
    return classifier

def train_val_test_split(dataset, train_size, val_size, test_size):
    train_dataset = dataset.take(train_size).batch(args.batch_size)
    train_dataset = train_dataset.shuffle(1000, reshuffle_each_iteration=True)
    val_dataset = dataset.skip(train_size).take(5000).batch(args.batch_size)
    test_dataset = dataset.skip(40000).batch(args.batch_size)
    return train_dataset, val_dataset, test_dataset

def create_image_paths(img_folder, data_info, slice_info):
    img_paths = []
    for i, class_name in enumerate(data_info):
        # start, end = slice_info[i][0], slice_info[i][1]
        for j in slice_info[i]:
            for k in range(1, data_info[class_name][str(j)] + 1):
                img_paths.append(img_folder + "/" + class_name + "_" + str(j) + "_" + str(k).zfill(3) + ".jpg")
    return img_paths

def prepare_data_AE(img_folder, data_info, class_to_video, args, img_size=(224, 224)):
    train, val, test = [], [], []
    for class_name in data_info:
        number_of_videos = len(data_info[class_name])
        train_no = int(0.6 * number_of_videos)
        val_no = int(0.25 * number_of_videos)
        # test_no = number_of_videos - train_no - val_no # NOT USED ACTUALLY
        
        train.append(class_to_video[class_name][:train_no])
        val.append(class_to_video[class_name][train_no : train_no + val_no])
        test.append(class_to_video[class_name][train_no + val_no:])
    img_height, img_width = img_size[0], img_size[1]
    # step 1
    train_img_paths = create_image_paths(img_folder, data_info, train)
    val_img_paths = create_image_paths(img_folder, data_info, val)
    test_img_paths = create_image_paths(img_folder, data_info, test)

    train_img_paths = tf.constant(train_img_paths)
    val_img_paths = tf.constant(val_img_paths)
    test_img_paths = tf.constant(test_img_paths)

    # # step 2: create a dataset returning slices of `filenames`
    train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_img_paths))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_img_paths))

    # # step 3: parse every image in the dataset using `map`
    def _parse_function_ae(filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image /= 255.0
        image = tf.image.resize(image, (img_height, img_width))
        return image, image
    
    def _parse_function_ae_test(filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image /= 255.0
        image = tf.image.resize(image, (img_height, img_width))
        return filename, image, image

    train_dataset = train_dataset.map(_parse_function_ae).batch(args.batch_size)
    val_dataset = val_dataset.map(_parse_function_ae).batch(args.batch_size)
    test_dataset = test_dataset.map(_parse_function_ae_test)#.batch(args.batch_size)
    # train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, 35000, 5000, 10000)
    
    return train_dataset, val_dataset, test_dataset

def prepare_data_CLS(img_paths, gts, img_size=(224, 224)):
    img_height, img_width = img_size[0], img_size[1]
    # step 1
    filenames = tf.constant(img_paths)
    tf_labels = tf.constant(gts)

    # step 2: create a dataset returning slices of `filenames`
    dataset = tf.data.Dataset.from_tensor_slices((filenames, tf_labels))

    # step 3: parse every image in the dataset using `map`
    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image /= 255.0
        image = tf.image.resize(image, (img_height, img_width))
        return image, label

    dataset = dataset.map(_parse_function)

    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, 35000, 5000, 10000)
    return train_dataset, val_dataset, test_dataset

def prepare_data_MSE_CLS(img_folder, data_info, class_to_video, args, img_size=(224, 224)):
    train, val, test = [], [], []
    for class_name in data_info:
        number_of_videos = len(data_info[class_name])
        train_no = int(0.6 * number_of_videos)
        val_no = int(0.25 * number_of_videos)
        
        train.append(class_to_video[class_name][:train_no])
        val.append(class_to_video[class_name][train_no : train_no + val_no])
        test.append(class_to_video[class_name][train_no + val_no:])

    img_height, img_width = img_size[0], img_size[1]

    # step 1
    train_img_paths = create_image_paths(img_folder, data_info, train)
    val_img_paths = create_image_paths(img_folder, data_info, val)
    test_img_paths = create_image_paths(img_folder, data_info, test)

    train_img_paths = tf.constant(train_img_paths)
    val_img_paths = tf.constant(val_img_paths)
    test_img_paths = tf.constant(test_img_paths)

    # step 2: create a dataset returning slices of `filenames`
    train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_img_paths))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_img_paths))

    # step 3: parse every image in the dataset using `map`
    def _parse_function_ae(filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image /= 255.0
        image = tf.image.resize(image, (img_height, img_width))
        return image, {"ae_model":image, "efficientnet_b0_classification_1":image}

    def _parse_function_ae_test(filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image /= 255.0
        image = tf.image.resize(image, (img_height, img_width))
        return filename, image, {"ae_model":image, "efficientnet_b0_classification_1":image}

    train_dataset = train_dataset.map(_parse_function_ae).batch(args.batch_size)
    val_dataset = val_dataset.map(_parse_function_ae).batch(args.batch_size)
    test_dataset = test_dataset.map(_parse_function_ae_test) #.batch(args.batch_size)

    # for inputs, targets in train_dataset.take(1): # Print shapes of train_dataset-Only take a single example
    #     print(f'Train Dataset - inputs shape: {inputs.shape}, targets shape: {targets.shape}')

    # for inputs, targets in val_dataset.take(1): # Print shapes of val_dataset-Only take a single example
    #     print(f'Validation Dataset - inputs shape: {inputs.shape}, targets shape: {targets.shape}')
    
    return train_dataset, val_dataset, test_dataset


# def prepare_data_MSE_CLS(img_paths, gts, img_size=(224, 224)):
#     img_height, img_width = img_size[0], img_size[1]
#     # step 1
#     filenames = tf.constant(img_paths)
#     tf_labels = tf.constant(gts)

#     # step 2: create a dataset returning slices of `filenames`
#     dataset = tf.data.Dataset.from_tensor_slices((filenames, tf_labels))

#     # step 3: parse every image in the dataset using `map`
#     def _parse_function(filename, label):
#         image_string = tf.io.read_file(filename)
#         image_decoded = tf.image.decode_jpeg(image_string, channels=3)
#         image = tf.cast(image_decoded, tf.float32)
#         image /= 255.0
#         image = tf.image.resize(image, (img_height, img_width))
#         return image, {"ae_model":image, "efficientnet_b0_classification_1":label}

#     dataset = dataset.map(_parse_function)

#     train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, 35000, 5000, 10000)
#     return train_dataset, val_dataset, test_dataset

def prepare_data_MSE_KL(img_paths, gts, img_size=(224, 224)):
    img_height, img_width = img_size[0], img_size[1]

    prob_dist = np.load('data/b0_logits_224.npy')
    # step 1
    filenames = tf.constant(img_paths)
    tf_labels = tf.constant(prob_dist)

    # step 2: create a dataset returning slices of `filenames`
    dataset_pd = tf.data.Dataset.from_tensor_slices((filenames, tf_labels))

    # step 3: parse every image in the dataset using `map`
    def _parse_function(filename, label):
    #     print(filename)
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image /= 255.0
        image = tf.image.resize(image, (img_height, img_width))
        return image, {"ae_model":image, "efficientnet_b0_classification_1":label}

    dataset_pd = dataset_pd.map(_parse_function)
    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset_pd, 35000, 5000, 10000)
    return train_dataset, val_dataset, test_dataset


def fine_tune_AE(autoencoder, cls, learning_rate, train_dataset, val_dataset):
    class AllLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logging.info(">>>>>>>>At epoch {}: {}".format(epoch, logs))
                
    allLogger = AllLogger()
    joint_model = imagenet_utils.joint_AE_cls(autoencoder, cls, learning_rate)
    if os.path.exists(model_save_path+"joint"):
        logging.info("Load existing joint weight from: {}".format(model_save_path+"joint"))
        joint_model.load_weights(model_save_path+"joint")
    checkpoint = ModelCheckpoint(model_save_path+"joint", monitor='val_top_5_accuracy', verbose=1, save_best_only=True, mode='max')
    joint_model.fit(
        train_dataset,
        epochs=args.epochs,
        verbose=1,
        shuffle=True,
        validation_data=val_dataset,
        callbacks=[checkpoint, allLogger]
    )

    return joint_model


def fine_tune_AE_MSE_CROSS(autoencoder, cls, learning_rate, train_dataset, val_dataset):
    class AllLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logging.info(">>>>>>>>At epoch {}: {}".format(epoch, logs))
                
    ##########################################################################
    # Prepare for training
    ##########################################################################
    
    # Checkpoints
    checkpoint = ModelCheckpoint(
        best_model_path+"_epoch_{epoch:03d}_val_acc_{val_efficientnet_b0_classification_1_top_5_accuracy:.3f}", 
        monitor='val_efficientnet_b0_classification_1_top_5_accuracy', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='max'
    )
    checkpoint_last = ModelCheckpoint(
        last_model_path, 
        verbose=1, 
        save_weights_only=True, 
    )

    # Training state saver
    model_state = ModelState(os.path.join(joint_path, "state.json"), ["val_efficientnet_b0_classification_1_top_5_accuracy"], [tf.math.greater])

    # Metric Logger
    metricloggercallback = MetricLogger(monitor='val_efficientnet_b0_classification_1_top_5_accuracy', monitor_op=tf.math.greater, best=-np.inf)
    # Restore metric values
    if model_state.state['best_values']:
        checkpoint.best = model_state.state['best_values']['val_efficientnet_b0_classification_1_top_5_accuracy']
        metricloggercallback.best = model_state.state['best_values']['val_efficientnet_b0_classification_1_top_5_accuracy']
    
    print("set learning rate", learning_rate)

    ## Create model and load weights
    joint_model = imagenet_utils.joint_AE_cls_mse_crossentropy(autoencoder, cls, learning_rate)
    if (not args.restart_training) and os.path.exists(joint_path):
        logging.info("<<<<<<<<<<<<<<<<<< JOINT: LOAD PREVIOUS MODEL >>>>>>>>>>>>>>>>>>>>>>>>")
        model_load_path = tf.train.latest_checkpoint(joint_path)
        if model_load_path is not None:
            logging.info("    ``````` restored from {} ```````    ".format(model_load_path))
            joint_model.load_weights(model_load_path)
    else:
        logging.info("<<<<<<<<<<<<<<<<<< JOINT: TRAIN WITH NEW MODEL >>>>>>>>>>>>>>>>>>>>>>>>")


    joint_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                            loss=['mse', keras.losses.SparseCategoricalCrossentropy()],
                            loss_weights=[10,1],
                            metrics={"efficientnet_b0_classification_1":tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')}
                            ) # top_5_categorical_accuracy_customize

    print("JOINT MODEL SUMMARY BELOW:")                
    joint_model.summary()
    keras.backend.set_value(joint_model.optimizer.learning_rate, learning_rate)
    
    # joint_model.evaluate(val_dataset)
    print("JOINT MODEL INPUT SHAPE: ", joint_model.input_shape)
    print("JOINT MODEL OUTPUT SHAPE: ", joint_model.output_shape)
    joint_model.fit(
        train_dataset,
        epochs=args.epochs,
        verbose=1,
        shuffle=True,
        validation_data=val_dataset,
        callbacks=[
            keras.callbacks.LearningRateScheduler(customize_lr_sccheduler(15, 0.3), verbose=1),
            checkpoint, 
            checkpoint_last,
            model_state,
            metricloggercallback, 
        ],
        initial_epoch=model_state.state['epoch_count']-1
    )

    return joint_model


def fine_tune_AE_MSE_KL(autoencoder, cls, learning_rate, train_dataset, val_dataset):
    
                
    ##########################################################################
    # Prepare for training
    ##########################################################################
    
    checkpoint = ModelCheckpoint(
        best_model_path+"_epoch_{epoch:03d}_val_acc_{val_efficientnet_b0_classification_1_top_5_categorical_accuracy_customize:.3f}", 
        monitor='val_efficientnet_b0_classification_1_top_5_categorical_accuracy_customize', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='max'
    )
    checkpoint_last = ModelCheckpoint(
        last_model_path, 
        verbose=1, 
        save_weights_only=True, 
    )

    model_state = ModelState(os.path.join(joint_path, "state.json"), ["val_efficientnet_b0_classification_1_top_5_categorical_accuracy_customize"], [tf.math.greater])

    metricloggercallback = MetricLogger(monitor='val_efficientnet_b0_classification_1_top_5_categorical_accuracy_customize', monitor_op=tf.math.greater, best=-np.inf)
    
    if  model_state.state['best_values']:
        checkpoint.best = model_state.state['best_values']['val_efficientnet_b0_classification_1_top_5_categorical_accuracy_customize']
        metricloggercallback.best = model_state.state['best_values']['val_efficientnet_b0_classification_1_top_5_categorical_accuracy_customize']


    print("set learning rate", learning_rate)
    joint_model = imagenet_utils.joint_AE_cls_mse_crossentropy_single(autoencoder, cls)

    # if (not args.restart_training) and os.path.exists(joint_path):
    #     logging.info("<<<<<<<<<<<<<<<<<< JOINT: LOAD PREVIOUS MODEL >>>>>>>>>>>>>>>>>>>>>>>>")
    #     model_load_path = tf.train.latest_checkpoint(joint_path)
    #     if model_load_path is not None:
    #         logging.info("    ``````` restored from {} ```````    ".format(model_load_path))
    #         joint_model.load_weights(model_load_path)
    # else:
    #     logging.info("<<<<<<<<<<<<<<<<<< JOINT: TRAIN WITH NEW MODEL >>>>>>>>>>>>>>>>>>>>>>>>")

    def top_5_categorical_accuracy_customize(y_true, y_pred):
        one_hott = tf.squeeze(tf.one_hot(tf.cast(tf.math.argmax(y_true, axis=-1), dtype=tf.int32), y_pred.shape[1]))
        return tf.keras.metrics.top_k_categorical_accuracy(one_hott, y_pred, k=5)

    joint_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                            loss=['mse', tf.keras.losses.KLDivergence()],
                            loss_weights=[10,1],
                            # metrics={"efficientnet_b0_classification_1":tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')}
                            metrics={"efficientnet_b0_classification_1":top_5_categorical_accuracy_customize}
                            ) # top_5_categorical_accuracy_customize
                        
    joint_model.summary()

    # joint_model.evaluate(val_dataset.take(1000))

    keras.backend.set_value(joint_model.optimizer.learning_rate, learning_rate)
    joint_model.fit(
        train_dataset,
        epochs=args.epochs,
        verbose=1,
        shuffle=True,
        validation_data=val_dataset,
        callbacks=[
            keras.callbacks.LearningRateScheduler(customize_lr_sccheduler(15, 0.3), verbose=1),

            checkpoint, 
            checkpoint_last,
            model_state,
            metricloggercallback, 
        ],
        initial_epoch=model_state.state['epoch_count']-1
    )

    return joint_model



def get_encoder_decoder(autoencoder):
    decoder_input_index = None
    layerName = 'Decoder'
    for idx, layer in enumerate(autoencoder.layers):
        if layer.name == layerName:
            decoder_input_index = idx
            break

    print(decoder_input_index)
            
    # encoder = keras.Model(autoencoder.input, autoencoder.get_layer(name = 'encoder').output, name='encoder1')
    encoder = tf.keras.Sequential(name='encoder1')
    for layer in autoencoder.layers[:2]:
        encoder.add(layer)
    decoder = tf.keras.Sequential(name='decoder1')
    for layer in autoencoder.layers[decoder_input_index:]:
        decoder.add(layer)

    return encoder, decoder


def ae_model_loader(name):
    if name == 'PNC':
        model = AsymAE_two_conv_PNC
    # if name == 'AsymAE_two_conv':
    #     model = AsymAE_two_conv
    if name == 'as_deeper_2':
        model = AsymAE_deeper_2

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--memo', type=str, default='default_memo')
    parser.add_argument('--mode', type=int, default=0, help='0 for MSE reconstruction, 1 for joint training.')
    parser.add_argument('--model', type=str, default='PNC', help='Folder to save the model.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--input_size', type=int, default=224, help='Size of the input images.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of training.')
    parser.add_argument('--ae_path', type=str, default='saved_models/default2', help='Folder to save the model.')
    parser.add_argument('--joint_path', type=str, default='saved_models/default2', help='Folder to save the model.')
    parser.add_argument('--log_save_path', type=str, default='./', help='Path to save the logs.')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id.')
    parser.add_argument('--if_mem_constr', type=boolean_string, default=True, help='GPU memory growth.')
    parser.add_argument('--restart_training', type=boolean_string, default=False, help='If start from scratch.')
    parser.add_argument('--tail_drop', type=int, default=0, help='Number of features from tail to drop.')
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

    # # Use GPU for training
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.if_mem_constr:
        physical_devices = tf.config.list_physical_devices('GPU')
        try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except: pass
    
    # prepare labels
    image_to_label_map = prepare_labels() #[:50000]
    data_info, class_to_video = extract_info_of_dataset(image_to_label_map)
    print("DATA INFO:: ", data_info)
    print("CLASS_TO_VIDEO:: ", class_to_video)
    # exit()

    # # prepare data
    img_folder = "../new_video_frames_dataset"
    assert(os.path.exists(img_folder))

    # img_paths = sorted(glob.glob(img_folder+'/*'))[:50000]
    # logging.info("Number of imgs in the folder: {}".format(len(img_paths)))
    input_size = (args.input_size, args.input_size)

    imagenet_utils=imagenetUtils(size=input_size)

    ae_train_dataset, ae_val_dataset, ae_test_dataset = prepare_data_AE(img_folder, data_info, class_to_video, img_size=input_size, args=args)
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
    
    if args.mode == 0:
        logging.info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        logging.info("vvvvvvvvvvvvvvvvv  Enter MSE Training  vvvvvvvvvvvvvvvvv")
        logging.info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        last_model_path = os.path.join(ae_path, "last_checkpoint")
        best_model_path = os.path.join(ae_path, "best_checkpoint")
        ##########################################################################
        # Prepare for training
        ##########################################################################
        
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

        # class ModelState(keras.callbacks.Callback):
        #     def __init__(self, state_path):
        #         self.state_path = state_path
        #         if os.path.isfile(state_path):
        #             logging.info('Loading existing .json state: {}'.format(state_path))
        #             with open(state_path, 'r') as f:
        #                 try:
        #                     self.state = json.load(f)
        #                 except:
        #                     self.state = { 'epoch_count': 0,
        #                         'best_values': {},
        #                         'best_epoch': {}
        #                         }
        #         else:
        #             self.state = { 'epoch_count': 0,
        #                         'best_values': {},
        #                         'best_epoch': {}
        #                         }
        #         self.state['epoch_count'] += 1

        #     def on_train_begin(self, logs={}):
        #         logging.info('\n' + '===='*10 + '\n' + 'Start Training... with {} = {}'.format('val_loss', self.state['best_values']['val_loss']))

        #     def on_epoch_end(self, batch, logs={}):
        #         # Currently, for everything we track, lower is better
        #         for k in logs:
        #             if k not in self.state['best_values'] or logs[k] < self.state['best_values'][k]:
        #                 self.state['best_values'][k] = float(logs[k])
        #                 self.state['best_epoch'][k] = self.state['epoch_count']

        #         with open(self.state_path, 'w') as f:
        #             json.dump(self.state, f, indent=4)
        #         logging.info('Completed epoch: {}'.format(self.state['epoch_count']))

        #         self.state['epoch_count'] += 1
        model_state = ModelState(os.path.join(ae_path,"state.json"), ['val_loss'], [tf.math.less])
        
        metricloggercallback = MetricLogger(monitor='val_loss', monitor_op=tf.math.less, best=np.inf)
        
        if  model_state.state['best_values']:
            checkpoint.best = model_state.state['best_values']['val_loss']
            metricloggercallback.best = model_state.state['best_values']['val_loss']
        def lr_step_decay(epoch, lr):
            if epoch != 0 and epoch % 10 == 0:
                logging.info("LearningRateScheduler(MSE) setting learning rate to: {}.".format(lr*0.3))
                return lr*0.3
            return lr

        ae_train_dataset.cache()
        ae_val_dataset.cache()
        ##########################################################################
        # Training AE only. MSE loss.
        ##########################################################################
        logging.info("vvvvvvvvvvvvvvvvv Start AE MSE Training vvvvvvvvvvvvvvvvv")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss="MSE")
        model.summary()
        model.fit(
            ae_train_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            callbacks=[
                keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1),
                checkpoint, 
                checkpoint_last,
                model_state,
                metricloggercallback, 
            ],
            validation_data=ae_val_dataset,
            initial_epoch=model_state.state['epoch_count']-1
        )

        logging.info("^^^^^^^^^^^^^^^^^^ Finish AE MSE Training ^^^^^^^^^^^^^^^^^^")
    elif args.mode == 19:
        # avg_mse = 0
        # frame_count = 0
        # for input_image, output_image in ae_val_dataset:
        #     predicted_image = model.predict(input_image)
        #     mse = tf.reduce_sum(tf.math.square(predicted_image - output_image), axis=None)
        #     avg_mse += mse
        #     frame_count += args.batch_size
        # avg_mse /= frame_count
        # print(avg_mse)
        
        mse_per_video = defaultdict(def_value_1)
        frame_per_video = defaultdict(def_value_1)
        for file, input_image, output_image in ae_test_dataset:
            predicted_image = model.predict(tf.expand_dims(input_image, axis=0))
            # for i in range(args.tail_drop):
            if args.tail_drop != 0:
                predicted_image[:,:,:,-args.tail_drop:] = 0
            mse = tf.reduce_sum(tf.math.square(predicted_image - output_image), axis=None)
            video = "".join(file.numpy().decode("utf-8").split("/")[-1][:-4].split("_")[:-1])
            mse_per_video[video] += mse
            frame_per_video[video] += 1
            # break
        for video in mse_per_video:
            mse_per_video[video] /= frame_per_video[video]
            print(video, mse_per_video[video].numpy())
    
    # elif args.mode == 20:
    #     encoder = intermediate_layer_model = tf.keras.Model(inputs=model.input,
    #                                        outputs=model.get_layer("encoder").output)
    #     print(encoder.summary())
            
    elif args.mode == 1:
        logging.info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        logging.info("vvvvvv  Enter Joint Training (NON-Trainable CLS)  vvvvvv")
        logging.info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        last_model_path = os.path.join(joint_path, "last_checkpoint")
        best_model_path = os.path.join(joint_path, "best_checkpoint")

        classifier = imagenet_utils.img_classifier(trainable=False)
        # cls.summary()
        # joint_train_dataset, joint_val_dataset, joint_test_dataset = prepare_data_MSE_CLS(img_paths, gts, img_size=input_size)
        joint_train_dataset, joint_val_dataset, joint_test_dataset = prepare_data_MSE_CLS(img_folder, data_info, class_to_video, args=args, img_size=input_size)
        joint_train_dataset.cache()
        joint_val_dataset.cache()
        joint_model = fine_tune_AE_MSE_CROSS(model, classifier, args.learning_rate, joint_train_dataset, joint_val_dataset)

    # elif args.mode == 2:
    #     model = ModelObject(out_size=10).asym_ae(tailDrop=True, encoder_trainable=False)
    #     print("Trying to load model from {}".format(ae_path))
    #     if (not args.restart_training) and os.path.exists(ae_path):
    #         logging.info("<<<<<<<<<<<<<<<<<< LOAD PREVIOUS MODEL >>>>>>>>>>>>>>>>>>>>>>>>")
    #         model_load_path = tf.train.latest_checkpoint(ae_path)
    #         if model_load_path is not None:
    #             logging.info("    >>> restored from {}".format(model_load_path))
    #             model.load_weights(model_load_path)
    #         else:
    #             model.load_weights(model_load_path)
    #     else:
    #         logging.info("<<<<<<<<<<<<<<<<<< TRAIN WITH NEW MODEL >>>>>>>>>>>>>>>>>>>>>>>>")
    
    #     logging.info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    #     logging.info("vvvvvvvv  Enter Joint Training (Trainable CLS)  vvvvvvvv")
    #     logging.info("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    #     last_model_path = os.path.join(joint_path, "last_checkpoint")
    #     best_model_path = os.path.join(joint_path, "best_checkpoint")

    #     classifier = imagenet_utils.img_classifier(trainable=False)
    #     # cls.summary()
    #     joint_train_dataset, joint_val_dataset, joint_test_dataset = prepare_data_MSE_KL(img_paths, gts, img_size=input_size)
    #     joint_train_dataset.cache()
    #     joint_val_dataset.cache()
    #     joint_model = fine_tune_AE_MSE_KL(model, classifier, args.learning_rate, joint_train_dataset, joint_val_dataset)

    elif args.mode == 99:
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
        encoder.summary()
    #     classifier = imagenet_utils.img_classifier(trainable=True)
    #     joint_model = imagenet_utils.joint_AE_cls_mse_crossentropy_single(model, classifier)
    #     # cls.summary()
    #     _,_,eval_test_dataset = prepare_data_CLS(img_paths, gts, img_size=input_size)
    #     eval_test_dataset.cache()
    #     model_load_path = "saved_models_as_deeper_2/joint_KL/last_checkpoint"
    #     if (not args.restart_training) and os.path.exists(model_load_path):
    #         logging.info("<<<<<<<<<<<<<<<<<< JOINT: LOAD PREVIOUS MODEL >>>>>>>>>>>>>>>>>>>>>>>>")
    #         if model_load_path is not None:
    #             logging.info("    ``````` restored from {} ```````    ".format(model_load_path))
    #             joint_model.load_weights(model_load_path)
    #     else:
    #         logging.info("<<<<<<<<<<<<<<<<<< JOINT: TRAIN WITH NEW MODEL >>>>>>>>>>>>>>>>>>>>>>>>")

    #     print("Spliting encoder and decoder...")
    #     encoder, decoder = get_encoder_decoder_from_joint(joint_model)
    #     # encoder.save(joint_path+"encoder")
    #     # decoder.save(joint_path+"decoder")
        
    #     # if not (args.save_only):
    #     print("Evaluation for accuray...")
    #     EVAL = evaluate_ae_cls(encoder, decoder, classifier, eval_test_dataset, size=input_size)
    #     acc_compressed = [EVAL.eval_for_k_dim_pipeline(k) for k in [10]]
