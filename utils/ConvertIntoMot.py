# This file is used to convert the data into MOT format

import os
import cv2
import pandas as pd

# read data video

filenames = os.listdir('/home/hemanth12/Paper/TestData_MOT/')

# Writing gt.txt file into gt folder


gtcols = ['frame','TrackID','x','y','w','h']

for file in filenames:
    print(file)
    file = file.split('.')[0]
    gt = pd.read_csv('/home/hemanth12/Paper/TestData_MOT/'+file+'/gt/gt_vitbat.txt')
    print(gt['y'])
    gt['x'] = gt['x'].astype('uint8')
    gt.loc[gt['x'] < 0,'x'] = 1

    # Bottom Left x,y
    gt.loc[gt['y'] < 0,'y'] = 1

    gt['y'] = (gt['y']+(gt['h'])).astype('uint8')
    # Top Left x,y
    # gt['y'] = gt['y'].astype('uint8')
    gt['w'] = gt['w'].astype('uint8')
    gt['h'] = gt['h'].astype('uint8')
    gt['wx'] = [1.0] * len(gt['frame'])
    gt['wy'] = [-1.0] * len(gt['frame'])
    gt['wz'] = [-1.0] * len(gt['frame'])
    # os.makedirs('NewMot/'+file+'/gt/')
    gt.to_csv('/home/hemanth12/Paper/TestData_MOT/'+file+'/gt/gt.txt',header = False,index=False)

# # Writing det.txt into det folder
detcols = ['frame','TrackID','x','y','w','h']

for file in filenames:
    print(file)
    file = file.split('.')[0]
    gt = pd.read_csv('/home/hemanth12/Paper/TestData_MOT/'+file+'/det/GT/det_vitbat.txt')
    print(gt)
    gt['x'] = gt['x'].astype('uint8')
    gt.loc[gt['x'] < 0,'x'] = 1

    # Bottom Left x,y
    gt.loc[gt['y'] < 0,'y'] = 1
    gt['y'] = (gt['y']+(gt['h'])).astype('uint8')

    # Top Left x,y
    # gt['y'] = gt['y'].astype('uint8')
    gt['w'] = gt['w'].astype('uint8')
    gt['h'] = gt['h'].astype('uint8')
    gt['c'] = [1.0] * len(gt['frame'])
    gt['wx'] = [1.0] * len(gt['frame'])
    gt['wy'] = [-1.0] * len(gt['frame'])
    gt['wz'] = [-1.0] * len(gt['frame'])
    gt['TrackID'] = [-1.0] * len(gt['frame'])

    # os.makedirs('NewMot/'+file+'/det/')
    gt.to_csv('/home/hemanth12/Paper/TestData_MOT/'+file+'/det/GT/det.txt',header = False,index=False)


