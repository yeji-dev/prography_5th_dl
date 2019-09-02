# Detect objects in photos with mask rcnn model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import listdir
from csv_to_xml import csv_to_xml
from xml.etree import ElementTree
import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.model import load_image_gt
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.visualize import my_display_instances
import argparse

# select mode
TRAIN_MODE = False
TEST_MODE = True

# class that defines and loads the refrigerator dataset
class RefriDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # guess what the new image
        if dataset_dir[0:13] != 'refri_dataset':
            self.add_class("dataset", 1, "apple")
            self.add_class("dataset", 2, "beer")
            self.add_class("dataset", 3, "egg")
            self.add_class("dataset", 4, "mandarin")
            self.add_class("dataset", 5, "milk")
            self.add_class("dataset", 6, "soju")
            self.add_image('dataset', image_id=0, path=dataset_dir, annotation='none_annots.xml')

        # images and annots
        else:
            # define classes
            self.add_class("dataset", 1, "apple")
            self.add_class("dataset", 2, "beer")
            self.add_class("dataset", 3, "egg")
            self.add_class("dataset", 4, "mandarin")
            self.add_class("dataset", 5, "milk")
            self.add_class("dataset", 6, "soju")
            # define data locations
            images_dir = dataset_dir + '/images/'
            annotations_dir = dataset_dir + '/annots/'
            # find all images
            for filename in listdir(images_dir):
                # extract image id (trim '.xml')
                image_id = filename[:-4]
                img_path = images_dir + filename
                ann_path = annotations_dir + image_id + '.xml'
                # add to dataset
                # print('load_dataset, image_id:' + str(image_id))
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # load all bounding boxes for an image
    def extract_boxes(self, filename):
        # load and parse the file
        root = ElementTree.parse(filename)
        boxes = list()
        # extract each bounding box
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        index_name = path.split("/")[3]
        index_name = index_name[:-9]

        # print('path: ', path)
        # print('in: ', index_name)
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1

            class_ids.append(self.class_names.index(index_name))
            # print('load_mask, image_id: ' + str(image_id))
            # print('class_ids:', class_ids)
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# define a configuration for Training
class RefriConfig(Config):
    # define the name of the configuration
    NAME = "refri_cfg"
    # number of classes (background + refri)
    NUM_CLASSES = 1 + 6
    # number of training steps per epoch
    STEPS_PER_EPOCH = 1341
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# define the prediction configuration for Testing
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "refri_cfg"
    # number of classes (background + refri)
    NUM_CLASSES = 1 + 6
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        print('eval_imageid: ',image_id)
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP


# plot a image # include a custom func
def plot_test(img_url, model, cfg):
    # load image (only one for cmd testing)
    image = img_url.load_image(0)
    # expects an RGB image (or array of images) and subtracts
    scaled_image = image.astype(np.float32) - np.array([123.7, 116.8, 103.9])
    sample = expand_dims(scaled_image, 0)
    yhat = model.detect(sample, verbose=0)[0]

    class_names = list()
    class_names.insert(0, "none")
    class_names.insert(1, "apple")
    class_names.insert(2, "beer")
    class_names.insert(3, "egg")
    class_names.insert(4, "mandarin")
    class_names.insert(5, "milk")
    class_names.insert(6, "soju")

    # show the figure
    my_display_instances(image, yhat['rois'], yhat['class_ids'], class_names) # without masks pyplot


def main():
    # handle command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, metavar='[model name]', default='mask_rcnn_refri_cfg_0030.h5',
                        help="Please input the model name. and you can also omit model options.")
    parser.add_argument('--image', type=str, metavar='[input image]',
                        help="Please input the image name.")

    args = parser.parse_args()
    model_url = args.model
    image_url = args.image


    # load the train dataset
    train_set = RefriDataset()
    train_set.load_dataset('refri_dataset/train', is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    # load the validation dataset
    valid_set = RefriDataset()
    valid_set.load_dataset('refri_dataset/validation', is_train=True)
    valid_set.prepare()
    print('Validation: %d' % len(valid_set.image_ids))

    # load the test dataset
    test_set = RefriDataset()
    test_set.load_dataset('refri_dataset/test', is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    if TRAIN_MODE:
        # convert csv to xml
        csv_to_xml(False)

        # prepare config
        cfg1 = RefriConfig()
        cfg1.display()
        # define the model (for training)
        model = MaskRCNN(mode='training', model_dir='./', config=cfg1)
        # load weights (mscoco) and exclude the output layers
        model.load_weights('mask_rcnn_coco.h5', by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        # train weights (layers='heads', learning_rate=0.001)
        model.train(train_set, valid_set, learning_rate=cfg1.LEARNING_RATE, epochs=30, layers='heads')

    if TEST_MODE:
        # create config
        cfg2 = PredictionConfig()
        # define the model (for inference)
        model = MaskRCNN(mode='inference', model_dir='./', config=cfg2)
        # load model weights
        model_path = 'refri_cfg20190902T0621/'
        model.load_weights(model_path + model_url, by_name=True)

        # load the cmd image
        cmd_image = RefriDataset()
        cmd_image.load_dataset(image_url, is_train=False)
        cmd_image.prepare()
        cmd_cfg = RefriConfig()
        # plot test for input image
        plot_test(cmd_image, model, cmd_cfg)
        '''
        # evaluate model on training dataset
        train_mAP = evaluate_model(train_set, model, cfg2)
        print("Train mAP: %.3f" % train_mAP)
        # evaluate model on test dataset
        test_mAP = evaluate_model(test_set, model, cfg2)
        print("Test mAP: %.3f" % test_mAP)
        '''


if __name__=="__main__":
    main()

