import os
import sys
import cv2
from utils import calc_center
# cuDNNが使えないエラー回避
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/dl-box/atsushi/github/Mask_RCNN")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

from mrcnn import visualize
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def Inference_model():
    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model_path = model.find_last()
    model.load_weights(model_path, by_name=True)
    print('weight loaded!')
    return model


def render(result, rgb_image, target):
    N = result['rois'].shape[0]  # 検出数
    result_image = rgb_image.copy()
    colors = visualize.random_colors(N)
    for i in range(N):
        '''クラス関係なく1物体ごと処理を行う'''
        if class_names[result['class_ids'][i]] in target:
            # Color
            color = colors[i]
            rgb = (round(color[0] * 255), round(color[1] * 255), round(color[2] * 255))
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Bbox
            result_image = visualize.draw_box(result_image, result['rois'][i], rgb)
            # Class & Score
            text_top = class_names[result['class_ids'][i]] + ':' + str(result['scores'][i])
            result_image = cv2.putText(result_image, text_top,
                                       (result['rois'][i][1], result['rois'][i][0]),
                                       font, 0.7, rgb, 1, cv2.LINE_AA)
            # Mask
            mask = result['masks'][:, :, i]
            result_image = visualize.apply_mask(result_image, mask, color)
            # log
            print('class: {} | Score: {}'.format(class_names[result['class_ids'][i]], result['scores'][i]))
    return result_image, mask
