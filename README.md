# Football Object Detection
This project utilizes object detection algorithms to analyze football matche video by finding the position of different objects on the football pitch and classifying them into 4 different classes:  
-  Player left leg (Touch count)  
-  Player right leg (Touch count)
-  Ball Rotation
-  Player Movement Velocity.

## Explanation of work - 

###### Touch Counting
Pose estimation + collision check between ball and foot.
Leg identified via ankle keypoint proximity.

###### Ball Rotation
Optical flow analysis on cropped ball region.
Rotation direction inferred from dominant vector field.

###### Player Velocity

Tracking centroid across frames.
Velocity calculated as displacement/time.


