# Deep-Fish-Tracker-Network

### Requirments

Python 3.6  
Tensorflow 1.14.0  
Keras 2.3.1  

### Files Required to run the code are det.txt, gt.txt and every frame of the video in jpg format

Each line in both det.txt and gt.txt files should contain the following 10 values seperated by comma

1. frame_id -> Id of the frame in which detection is present
2. track_id -> -1 for det.txt file and id of the track for gt.txt
3. bb_top_left_x -> x co-ordinate of the top left corner of bounding box
4. bb_top_left_y -> y co-ordinate of the top left corner of bounding box
5. bb_width -> width of the bounding box
6. bb_height -> height of the bounding box
7. conf -> confidence of the detection 
8. x, y, z -> can ignore them for the 2D tracking. Fill -1 for these values.

