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
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import math
import copy

def detect(cfgfile, weightfile, video):
    m = Darknet(cfgfile)
    
    m.load_weights(weightfile)
    m.print_network()
    print('Loading weights from %s... Done!' % (weightfile))    
    use_cuda = 1
    m.cuda()

    all_boxes = []
    all_convrep = []
    frames = []
    cap = cv2.VideoCapture(video)
    while True:
        res, img = cap.read()
        if res:
            sized = cv2.resize(img,(m.width, m.height) )
            frames.append(sized)
        else:
            break
    frames = frames[:20]

    for sized in frames:
        boxes, convrep = do_detect(m, sized, 0.5, 0.4, use_cuda)
        pdb.set_trace()
        all_convrep.append(convrep)
        all_boxes.append(boxes)
    return frames, all_boxes, all_convrep


def display_object(track, frames):
    for i, sized in enumerate(frames):
        box = track[i]
        name = None #'prediction%d.jpg' %(i)        
        draw_img = plot_boxes_cv2(sized, [box], name, class_names)
        cv2.imshow(cfgfile, draw_img)
        cv2.waitKey(500)
    
def preprocess(total_boxes, object):
    ###['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ### Get only tracks of correct object and make sure it exists in every frame...
    for i in range(len(total_boxes)):
        j == 0
        aux = range(len(total_boxes[i]))
        while j in aux:
            if total_boxes[i][j][6] != object:
                total_boxes[i].remove(total_boxes[i][j])
                j-=1
            j+=1
       	    aux	= range(len(total_boxes[i]))
    return total_boxes

def dist(trans_boxes):
    transition = torch.zeros(len(trans_boxes[0]), len(trans_boxes[1]))
    for i in range(len(trans_boxes[0])):
        for j in range(len(trans_boxes[1])):
            transition[i][j] = math.log(1-math.sqrt((trans_boxes[0][i][0] - trans_boxes[1][j][0])**2 + \
                                                   (trans_boxes[0][i][1] - trans_boxes[1][j][1])**2))
    return transition

#else:
#    transition = torch.zeros(len(trans_boxes[0]), len(trans_boxes[2]), len(trans_boxes[1]), len(trans_boxes[3]))
#    for i in range(len(trans_boxes[0])):
#        for j in range(len(trans_boxes[2])):
#            for x in range(len(trans_boxes[1])):
#                for y in range(len(trans_boxes[3])):
#                    transition[i][x][j][y] += math.log(1-math.sqrt((trans_boxes[0][i][0] - trans_boxes[2][j][0])**2 + \
#                                                                   (trans_boxes[0][i][1] - trans_boxes[2][j][1])**2))
#                    transition[i][x][j][y] += math.log(1-math.sqrt((trans_boxes[1][x][0] - trans_boxes[3][y][0])**2 + \
#                                                                  (trans_boxes[1][x][1] - trans_boxes[3][y][1])**2))
#return transition

class word_rnn_tracker(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_s=1, dropout=0.2):
        super(word_rnn_tracker, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.linear = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_s=1):
        return (Variable(torch.zeros(num_layers,batch_s,hidden_size).cuda()),
         Variable(torch.zeros(num_layers,batch_s,hidden_size).cuda())) 

    def forward(self, detection, hidden):
        output, hidden = self.lstm(detection, hidden)
        output = self.linear(output)
        output = F.logsigmoid(output)
        return output, hidden

def crop_image(img, box, optical_size):
    width = img.shape[1]
    height = img.shape[0]
    x1 = int(round((box[0] - box[2]/2.0) * width))
    y1 = int(round((box[1] - box[3]/2.0) * height))
    x2 = int(round((box[0] + box[2]/2.0) * width))
    y2 = int(round((box[1] + box[3]/2.0) * height))
    crop_image = img[y1:y2, x1:x2]
    img = cv2.resize(crop_image,(optical_size, optical_size) )
    return corp_image

def optical_flow(img1,img2):
    hsv = np.zeros(img1.shape, dtype=np.uint8)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    #hsv[:, :, 0] = 255
    #hsv[:, :, 1] = 255
    #mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #hsv[..., 0] = ang * 180 / np.pi / 2
    #hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    #rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow

def train(frames, all_boxes, all_convrep, word, hidden_size, optical_size):
    input_size = (1024 + optical_size**2)*len(word)
    model_rnn = word_rnn_tracker(input_size, hidden_size)
    model_rnn.cuda()

    learning_rate = .001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_rnn.parameters(), lr = learning_rate)
    ###Temporal
    batch_temporal = dict()
    batch_temporal['target'] = 1
    train_iter = [batch_temporal]
    for batch in train_iter: 
        model_rnn.zero_grad()    
        tracks, log_likelihood = object_tracker(frames,all_boxes, all_convrep, word, model_rnn, optical_size)
        loss = criterion(log_likelihood, batch['target'])

        loss.backward
        optimizer.step()

    return model_rnn, tracks

def object_tracker(frames, all_boxes, convrep, word, model_rnn, optical_size):
    detections = []
    all_hidden = dict()
    for i, object in enumerate(word):
        detections.append(preprocess(all_boxes, object)) ##All detections of same category
    if len(detections)==1:
        #time 0
        forward_var = torch.log(torch.Tensor([all_boxes[0][item][5] for item in range(len(all_boxes[0]))]))
        #pdb.set_trace()
        for i, item in enumerate(detections[0][0]):
            input = convrep[0][item[9]/2][:,:,item[7],item[8]].contiguous().view(-1) 
            input = torch.cat((input,torch.zeros(1,optical_size**2)),0)
            input = Variable(input.cuda())
            hidden = model_rnn.init_hidden()
            output, hidden = model_rnn(input, hidden)
            all_models[(i)] = hidden
            forward_var[i] += output

        #time t
        best_tagid = [] ## All best paths
        for t in range(1,len(all_boxes)):
            for dim in len(detections):
                trans_boxes = []
                trans_boxes.append(detections[dim][t-1])
                trans_boxes.append(detections[dim][t])   
                transition = dist(trans_boxes)           
            next = forward_var.expand(transition.size()) + transition
            _, tag_id = torch.max(next,0)
            best_tagid.append((tag_id))
            next = torch.sum(next,0)

            forward_var = next

            forward_var += torch.log(torch.Tensor([boxes[t][item][5] for item in range(len(boxes[t]))])) 
            all_models_new = dict() 
            opt_flow = optical_flow(frames[t-1],frames[t])
            for i, item in enumerate(detections[0][t]):
                input = convrep[t][item[9]/2][:,:,item[7],item[8]].contiguous().view(-1)
                opt_flow = crop_image(opt_flow, item, optical_size)
                opt_flow = torch.from_numpy(opt_flow).view(-1)
                input =torch.cat((input,opt_flow),0)
                input = Variable(input.cuda())

                hidden = all_models[(tag_id)]
                output, hidden = model_rnn(input, hidden)
                all_models_new[(i)] = hidden
                forward_var[i] += output            
            all_models = copy.deepcopy(all_models_new) 

        path_score , tag_id = torch.max(forward_var,1)
        tag_id = tag_id[0][0]
        best_path = [tag_id]
        for back_t in reversed(best_tagid):
            tag_id = back_t[0][0][tag_id]
            best_path.append(tag_id)
        
        best_path.reverse()
        track = []
        for i in range(len(best_path)):
            track.append(boxes[i][best_path[i]]) 
        tracks = [track]

    else: ##2 detections= 2 objects
        #time 0
        for i,boxes in enumerate(detections):
            if i==0:
                forward_aux1 = torch.log(torch.Tensor([boxes[0][item][5] for item in range(len(boxes[0]))])).view(-1,1)
            if i==1:
                forward_aux2 = torch.log(torch.Tensor([boxes[0][item][5] for item in range(len(boxes[0]))]))
        #pdb.set_trace()
        forward_var = torch.zeros(forward_aux1.size()[1],forward_aux2.size()[0])
        forward_aux1.expand(forward_var.size())
        forward_aux2.expand(forward_var.size())
        forward_var = forward_aux1 + forward_aux2
    
        for i, item in enumerate(detections[0][0]):
            input = convrep[0][item[9]/2][:,:,item[7],item[8]].contiguous().view(-1)
            input = torch.cat((input,torch.zeros(1,optical_size**2)),0)
            for j, item2 in enumerate(detections[1][0]):
                input = torch.cat((input,convrep[0][item2[9]/2][:,item2[7],item2[8]].contiguous().view(-1)),0)
                input = torch.cat((input,torch.zeros(1,optical_size**2)),0)
                input = Variable(input.cuda())
                hidden = model_rnn.init_hidden()
                output, hidden = model_rnn(input,hidden)
                all_models[(i,j)] = hidden
                forward_var[i,j] += output
        #time t    
        best_tagid = [] ## All best paths
        for t in range(1,len(all)):
            distances = []
            size_future = []
            best_tagid_aux = dict()
            for dim in len(detections):
                trans_boxes = []
                trans_boxes.append(detections[dim][t-1])
                trans_boxes.append(detections[dim][t])   
                transition = dist(trans_boxes)
                distances.append(transition)
                size_future.append(transition.size()[1])
            
            future_var = torch.zeros(size_future)
            for future_i in future_var.size()[0]:
                for future_j in future_var.size()[1]:
                    current_var = copy.deepcopy(forward_var)
                    for current_i in current_var.size()[0]:
                        for current_j in current_var.size()[1]:
                            current_var[current_i][current_j] += distances[0][current_i][future_i] + \
                                                                 distances[1][current_j][future_j]
                    future_var[future_i][future_j] += current_var.sum()
                    aux ,tag_id_1 = torch.max(current_var,0)
                    _, tag_id_2 = torch.max(aux,0)
                    tag_id = (tag_id_1[tag_id_2],tag_id_2)
                    best_tagid_aux[(future_i,future_j)]=(tag_id)
            best_tag_id.append(best_tagid_aux)

            forward_var = future_var    
            for i,boxes in enumerate(detections):
                if i==0:
                    forward_aux = torch.log(torch.Tensor([boxes[t][item][5] for item in range(len(boxes[t]))])).view(-1,1)
                if i==1:
                    forward_aux = torch.log(torch.Tensor([boxes[t][item][5] for item in range(len(boxes[t]))]))
                forward_aux.expand(forward_var.size())
                forward_var += forward_aux

            all_models_new = dict()
            opt_flow = optical_flow(frames[t-1],frames[t])
            for i, item in enumerate(detections[0][t]):
                input = convrep[t][item[9]/2][:,:,item[7],item[8]].contiguous().view(-1)
                opt_flow = crop_image(opt_flow, item, optical_size)
                opt_flow = torch.from_numpy(opt_flow).view(-1)
                input =torch.cat((input,opt_flow),0)
                input = Variable(input.cuda())
                for j, item2 in enumerate(detections[1][t]):
                    input = torch.cat((input,convrep[t][item2[9]/2][:,item2[7],item2[8]].contiguous().view(-1)),0)

                    opt_flow = crop_image(opt_flow, item2, optical_size)
                    opt_flow = torch.from_numpy(opt_flow).view(-1)
                    input =torch.cat((input, opt_flow),0)
                    input = Variable(input.cuda())

                    hidden = all_models(tag_id)
                    output, hidden = model_rnn(input,hidden)
                    all_models_new[(i,j)] = hidden
                    forward_var[i,j] += output
            all_models = copy.deecopy(all_models_new)

        ##Decode path
        aux, tag_id_1 = torch.max(forward_var, 0)
        path_score, tag_id_2 = torch.max(aux,0)
        tag_id = (tag_id_1[tag_id_2],tag_id_2)

        best_path = [tag_id]
        for back_t in reversed(best_tagid):
            tag_id = back_t[0][tag_id]
            best_path.append(tag_id)
        
        best_path.reverse()
        track1 = []
        track2 = []
        for i in range(len(best_path)):
            track1.append(boxes[i][best_path[0][i]])
            track2.append(boxes[i][best_path[1][i]])
        tracks = [track1,track2]     


    return tracks , path_score
        
if __name__ == '__main__':

    cfgfile = 'cfg/yolov3.cfg'
    weightfile = 'yolov3.weights'
    #video = 'data/pretrained/corpus/clips/00028-1340-1430.avi'
    video = 'video1.mov'
    hidden_size = 100
    optical_size = 10
    lexicon = dict()
    lexicon['chase'] = [0,0]
    

    class_names = load_class_names('data/coco.names')

    frames, total_boxes, all_convrep = detect(cfgfile, weightfile, video)
    
    model_trained, object_tracks = train(frames, total_boxes, all_convrep, lexicon['chase'], hidden_size, optical_size)
    display_object(object_tracks, frames)



