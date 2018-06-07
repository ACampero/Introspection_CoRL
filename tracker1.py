import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import pdb
import cv2
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import math

def detect(cfgfile, weightfile, video):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    
    use_cuda = 1
    m.cuda()
   
    all_boxes = []
    frames = []
    cap = cv2.VideoCapture(video)
    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img,(m.width, m.height) )
            frames.append(sized)
        else:
            break
    #pdb.set_trace()
    frames = frames[:20]
    for sized in frames:
        boxes, _ = do_detect(m, sized, 0.5, 0.4, use_cuda)
        all_boxes.append(boxes)
    #pdb.set_trace()
    return frames, all_boxes

def display_object(track, frames):
    for i, sized in enumerate(frames):
        box = track[i]
        name = None #
        name = 'prediction%d.jpg' %(i)
        #pdb.set_trace()
        draw_img = plot_boxes_cv2(sized, [box], name, class_names)
        #cv2.imshow(cfgfile, draw_img)
        #cv2.waitKey(500)
    
def preprocess(total_boxes, object):
    ###['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ### Get only tracks of correct object and make sure it exists in every frame...
    for i in range(len(total_boxes)):
        j = 0
        aux = range(len(total_boxes[i]))
        while j in aux:
            if total_boxes[i][j][6] != object:
                total_boxes[i].remove(total_boxes[i][j])
                j-=1
            j+=1
            aux = range(len(total_boxes[i]))   
    return total_boxes

def dist(boxes_one,boxes_two):
    transition = torch.zeros(len(boxes_one), len(boxes_two))
    for i in range(len(boxes_one)):
        for j in range(len(boxes_two)):
            transition[i][j] = math.sqrt((boxes_one[i][0] - boxes_two[j][0])**2 + (boxes_one[i][1] - boxes_two[j][1])**2)
    return transition

def object_tracker(boxes):
    ##Detect a single object tracker assuming all are detections of same category, that will be specified by the word.
    pdb.set_trace()
    forward_var = torch.Tensor([boxes[0][item][-4] for item in range(len(boxes[0]))])
    best_tagid = [] ## All best paths
    for t in range(1,len(boxes)):
        transition = dist(boxes[t-1],boxes[t])
        next = forward_var.expand(transition.size()) - transition
        next, tag_id = torch.max(next,0)
        best_tagid.append(tag_id)
        #pdb.set_trace()
        forward_var = next + torch.Tensor([boxes[t][item][-4] for item in range(len(boxes[t]))]) 
    
    ##Decode path
    path_score , tag_id = torch.max(forward_var,0)
    tag_id = tag_id[0]
    best_path = [tag_id]
    for back_t in reversed(best_tagid):
        tag_id = back_t[tag_id]
        best_path.append(tag_id)
    
    best_path.reverse()
    track = []
    for i in range(len(best_path)):
        track.append(boxes[i][best_path[i]]) 
    return track , best_path
    
if __name__ == '__main__':

    cfgfile = 'cfg/yolov3.cfg'
    weightfile = 'yolov3.weights'
    #video = '00060-2633-2713.avi'
    video = 'video1.mov'
    video = 'data/pretrained/corpus/clips/00028-1340-1430.avi'
    object = 0

    class_names = load_class_names('data/coco.names')

    frames, total_boxes = detect(cfgfile, weightfile, video)
    boxes_per_object = preprocess(total_boxes, object)
    object_track , _ = object_tracker(boxes_per_object)
    display_object(object_track, frames)          

