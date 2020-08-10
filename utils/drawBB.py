# This code is used to draw bounding boxes in each frame

import os
import cv2
import pandas as pd

data_folder = '/media/shilpi/Backup Plus/Evaluation/Test/NewMot/'
folders = os.listdir(data_folder)
gtcols = ['frame','t','x','y','w','h','c','wx','wy','wz']

for folder in folders:
    print(folder)
    gt_data = pd.read_csv(data_folder+folder+'/det/GT/det.txt',names=gtcols,header=None)
    images = os.listdir(data_folder+folder+'/img1/')
    os.makedirs(data_folder+folder+'/det/GT/Frames+gt')
    for image in images:
        frame = image.split('.')[0]
        image = cv2.imread(data_folder+folder+'/img1/'+image)
        frame_gt_data = gt_data[gt_data['frame'] == int(frame)]
        for i in range(len(frame_gt_data['frame'].index)):
            cv2.rectangle(image,(int(frame_gt_data['x'][frame_gt_data.index[i]]),int(frame_gt_data['y'][frame_gt_data.index[i]])-int(frame_gt_data['h'][frame_gt_data.index[i]])),(int(frame_gt_data['x'][frame_gt_data.index[i]])+int(frame_gt_data['w'][frame_gt_data.index[i]]),int(frame_gt_data['y'][frame_gt_data.index[i]])),(0,255,0),2)

        cv2.imwrite(data_folder+folder+'/det/GT/Frames+gt/'+frame+'.jpg',image)




