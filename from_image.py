import cv2
import visualization_utils as vis_util
import label_map_util
import numpy as np
import os
import sys
import argparse
import tensorflow as tf

sys.path.append("..")

ap = argparse.ArgumentParser()
ap.add_argument("--image", "-i", required = True, help = "Path to input image")

# args = vars(ap.parse_args())
args = ap.parse_args()

# What model to download.
MODEL_NAME = 'model'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    '', 'labelmap.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
            
        image_np = cv2.imread(args.image)
        
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name(
            'num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
        
        min_score_thresh = 0.60
        bboxes = boxes[scores > min_score_thresh]
        im_width, im_height = image_np.shape[1::-1]
        
        max_boxes_to_draw = 20
        
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=5)
        
        basename = os.path.basename(args.image)

        filename = 'output/'+basename
        cv2.imwrite(filename,image_np)
        
        
        
        cv2.imshow('card_detection', cv2.resize(image_np, (800, 600)))
        cv2.waitKey(0)