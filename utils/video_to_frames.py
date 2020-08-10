# This code can be used to convert videos to frames

import os
import cv2
import pandas as pd

# read data video
filenames = os.listdir('/home/hemanth12/Paper/TestData_MOT/')

# # Converting Videos into Frames
for file in filenames:
	count = 1
	file = file.split('.')[0]
	print(file)
	os.makedirs('/home/hemanth12/Paper/TestData_MOT/'+file+'/img2/')
	video = cv2.VideoCapture('/home/hemanth12/Paper/FLV test dataset/'+file+'.flv')
	success = True
	while success:
		success,image = video.read()
		if success:
			cv2.imwrite('/home/hemanth12/Paper/TestData_MOT/'+file+'/img2/'+str(count)+'.jpg',image)
			count += 1
