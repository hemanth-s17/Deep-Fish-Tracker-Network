from keras.models import load_model
from keras import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Lambda,LSTM,BatchNormalization,LeakyReLU,PReLU, Bidirectional
from keras import Sequential
from keras.optimizers import RMSprop,Adam
from layers import AttentionWithContext
from keras.utils import get_custom_objects
import numpy.random as rng
import keras.backend as K
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import os
import cv2
import keras


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def iou(a,b):
    x1 = a[0]
    y1 = a[1]
    w1 = a[2]
    h1 = a[3]

    x2 = b[0]
    y2 = b[1]
    w2 = b[2]
    h2 = b[3]

    xA = a[0] + a[2]
    yA = a[1] + a[3]

    xB = b[0] + b[2]
    yB = b[1] + b[3]

    return get_iou([x1,y1,xA,yA],[x2,y2,xB,yB])

def get_centre(bnd_box):
    x = bnd_box[0]
    y = bnd_box[1]
    w = bnd_box[2]
    h = bnd_box[3]

    return [(4*x + 2*w)/4.0,(4*y + 2*h)/4.0]


def crop(image,x,y,w,h):
    img = np.array(cv2.imread(image)).astype('uint8')
    try:
        if img is not None:
            roi = img[y:y+h,x:x+w]
            roi = np.array(roi).astype('uint8')
            roi = cv2.resize(roi,(121,53),interpolation = cv2.INTER_AREA)
            return roi
    except:
        roi = img
        roi = cv2.resize(roi,(121,53),interpolation = cv2.INTER_AREA)
        print('Error')
        return roi
    

print('Loading LSTM Model.......')
Attentionmodel = load_model('LSTM.h5',custom_objects= {'AttentionWithContext':AttentionWithContext})
print('LSTM Model Loaded')
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

class CustomInitializer:
    def __call__(self, shape, dtype=None):
        return b_init(shape)

get_custom_objects().update({'b_init': CustomInitializer})

print('Loading Siamese Model.......')

Siamesemodel = load_model('Siamese.h5')

print('Siamese Model Loaded')
=
# for test videos
fish = '/home/hemanth12/Paper/TestData_MOT/'
filenames = os.listdir(fish)

for file in filenames:

    print('Loading'+ str(file) +'......')
    cols = ['frame','track','x','y','w','h','c','wx','wy','wz']
    data = pd.read_csv(fish+str(file)+'/det/GT/det.txt',names = cols)
    # print(data)
    print('Data Loaded......')
    image_folder = fish+str(file)+'/img1/'
    images = os.listdir(image_folder)

    for i in range(len(images)):
        images[i] = int(images[i].split('.')[0])

    images = sorted(images)
    images = images

    track_id = 0
    init = True
    missing = False
    # this x is to terminate all tracks when there rae no detection upto 5 consecutive frames
    x = 0
    tracks = {}
    for image in images:
        # Loading Data
        frame_data = data[data['frame'] == image]
        missing = False
        if len(frame_data) == 0:
            x += 1
            init = True
            missing = True
        else:
            x = 0

        img = image_folder+str(image)+'.jpg'
        bnd_boxes = []
        coords = []
        for i in frame_data.index:
            bnd_boxes.append(crop(img,int(frame_data['x'][i]),int(frame_data['y'][i]-frame_data['h'][i]),int(frame_data['w'][i]),int(frame_data['h'][i])))
            coords.append([frame_data['x'][i],frame_data['y'][i]-frame_data['h'][i],frame_data['w'][i],frame_data['h'][i]])

        if init == True and missing == False:
            init = False
            # this loop will work only for first frame and initialize tracks for the detections of first frame
            for i in range(len(bnd_boxes)):
                tracks[track_id] = [[bnd_boxes[i]],[coords[i]],0]
                print(image,track_id,frame_data['x'][frame_data.index[i]],frame_data['y'][frame_data.index[i]],frame_data['w'][frame_data.index[i]],frame_data['h'][frame_data.index[i]],-1,-1,-1)
                with open('/home/hemanth12/Paper/Results/'+file+'.txt','a+') as resfile:
                    resfile.write(str(image)+','+str(track_id)+','+str(frame_data['x'][frame_data.index[i]])+','+str(frame_data['y'][frame_data.index[i]])+','+str(frame_data['w'][frame_data.index[i]])+','+str(frame_data['h'][frame_data.index[i]])+','+'1,'+'-1,'+'-1\n')                    
                track_id += 1
        elif init == True and missing == True:
            if x >= 5:
                tracks = {}

        # Computing Cost Matrix        
        else:
            matrix = []
            for i in range(len(bnd_boxes)):
                track_row = []
                lstm_row = []
                for track in tracks:
                    # Siamese Score
                    track_row.append(1 - Siamesemodel.predict([tracks[track][0][-1].reshape(1,53,121,3),bnd_boxes[i].reshape(1,53,121,3)])[0][0])
                    # Attention LSTM Score
                    if len(tracks[track][1]) < 3:
                        iou_box = iou(tracks[track][1][-1],coords[i])
                        lstm_row.append((1 - iou_box))
                    else:
                        boxes = []
                        for j in range(len(tracks[track][1])-3,len(tracks[track][1])):
                            boxes.append(tracks[track][1][j])
                        inp = np.stack(boxes)
                        inp = np.array(inp).reshape(1,3,4)
                        pred = Attentionmodel.predict(inp)[0]
                        score = iou(pred,coords[i])
                        iou_box = iou(tracks[track][1][-1],coords[i])
                        lstm_row.append(1 - ((iou_box+score)*1.0/2.0) )
                        # lstm_row.append(1-score)


                    # below score is for Siamese + LSTM&IOU
                matrix.append(0.1 * np.array(track_row) + 0.9 * np.array(lstm_row))

            matrix = np.array(matrix)
            rows,cols = linear_sum_assignment(matrix)

        # Assigning bnd boxes to tracks    

            for i in range(len(rows)):
                tracks[list(tracks)[cols[i]]][0].append(bnd_boxes[rows[i]])
                tracks[list(tracks)[cols[i]]][1].append(coords[rows[i]])
                print(image,list(tracks)[cols[i]],frame_data['x'][frame_data.index[rows[i]]],frame_data['y'][frame_data.index[rows[i]]],frame_data['w'][frame_data.index[rows[i]]],frame_data['h'][frame_data.index[rows[i]]],-1,-1,-1)
                with open('/home/hemanth12/Paper//Results/'+file+'.txt','a') as resfile:
                    resfile.write(str(image)+','+str(list(tracks)[cols[i]])+','+str(frame_data['x'][frame_data.index[rows[i]]])+','+str(frame_data['y'][frame_data.index[rows[i]]])+','+str(frame_data['w'][frame_data.index[rows[i]]])+','+str(frame_data['h'][frame_data.index[rows[i]]])+',1,-1,-1\n')

                    # below code is deleting a particular existing track if does not get any assignmemnt upto 5 frames
            del_tracks = []
            for i in range(len(tracks)):
                if i not in cols:
                    tracks[list(tracks)[i]][2] += 1
                else:
                    tracks[list(tracks)[i]][2] = 0

                if tracks[list(tracks)[i]][2] == 5:
                    del_tracks.append(list(tracks)[i])

            for track in del_tracks:
                del tracks[track]




                # rows whichis not assigned any existing track start a new track for them 
            for i in range(len(bnd_boxes)):
                if i not in rows:
                    tracks[track_id] = [[bnd_boxes[i]],[coords[i]],0]
                    print(image,track_id,frame_data['x'][frame_data.index[i]],frame_data['y'][frame_data.index[i]],frame_data['w'][frame_data.index[i]],frame_data['h'][frame_data.index[i]],-1,-1,-1)
                    with open('/home/hemanth12/Paper/Results/'+file+'.txt','a') as resfile:
                        resfile.write(str(image)+','+str(track_id)+','+str(frame_data['x'][frame_data.index[i]])+','+str(frame_data['y'][frame_data.index[i]])+','+str(frame_data['w'][frame_data.index[i]])+','+str(frame_data['h'][frame_data.index[i]])+','+'1,'+'-1,'+'-1\n')


                    track_id += 1



