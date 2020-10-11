# Imports
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('custom_frozen_inference_graph', 'mscoco_label_map.pbtxt')

targeted_objects = "person" # (for counting targeted objects) change it with your targeted objects
is_color_recognition_enabled = 0

object_counting_api.targeted_object_counting(detection_graph, category_index, is_color_recognition_enabled, targeted_objects) # targeted objects counting

