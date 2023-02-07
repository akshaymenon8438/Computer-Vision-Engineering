# Computer-Vision-Engineering in Badminton Sports 
Shuttle Cock tracking is required for examining the trajectory of the shuttle
cock. Player service fault analysis identifies service faults during badminton matches
. The match point scored by players are analyzed by first referee through shuttle
cock landing point and player service faults . If the first referee cannot make de-
cision, they use technology such as a third umpire system to assist . The current
challenge with the third umpire system is based on high number of marginal error
for predicting match score . This research proposes a Machine Learning Framework
to improve the accuracy of Shuttlecock Tracking and Player service Fault Detection
. The proposed framework combines a shuttlecock trajectory model and a player
service fault model . The shuttlecock trajectory model is implemented using Pre-
trained Convolutional neural network (CNN) such as Tracknet.The player service
fault model uses Google MediaPipe Pose Pre-trained CNN model to classify player
service fault using Random Forest Classifier.The framework is trained using the
Badminton world federation channel dataset.The dataset consist of 100000 images
of badminton player and shuttle cock position ..The models are evaluated using a
confusion matrix, loss,accuracy , precision , f1 and recall. The Optimised Track-
Net Model has accuracy of 90% with less positioning error for shuttlecock tracking
whereas Player service fault detection can classify player fault with 90% accuracy
.The combined machine learning algorithm on shuttlecock tracking and player ser-
vice fault would benefit Badminton World Federation (BWF) for enhancing match
score analysis.


*************************How to Use The Code************************************************

# For shuttlecock tracking algorithm
1) The .MP4 badminton video is first converted from video to image.
2) The Image frames are then labelled using Microsoft visual tagging tool  and the output is exported in CSV formated with shuttle cock location on the each image frame.
3) The image frames are imported and fed into Heat map generation.
4) Post heat map generation images are split into test and train images .
5) The training of the trained images are fed into trackNet algorithm.
6) Post training of the images change the specify weight path , labelling csv file path , epoch as 500 and  sample size as 256.   

# Train with custom dataset
python train.py --save_weights_path=weights/model.h5 --training_images_name="training.csv" --epochs=500 --n_classes=256 --  input_height=512 --input_width=288 --load_weights=2 --step_per_epochs=200 --batch_size=2

7) Parameter of training
Parameter	Value
Image size	512 x 288
Heat map ball radius	2.5 pixel
Batch size	2
Learning rate	1.0
Epochs	500
Optimizer	Adadelta
Number of training images	~156

8) Once we get the saved weight apply the weight in the algorithm to find accuracy of the model.

TP, FP1, FP2, TN, FN are defined as below:

TP: True positive, center distance of ball between prediction and ground truch is smaller than 5 pixel
FP1: False positive, center distance of ball between prediction and ground truch is larger than 5 pixel
FP2: Fasle positive, if ball is not in ground truth but in prediction.
TN: True negative.
FN: False positive.
Metric	Formula	Value
Accuracy	(TP+TN)/(TP+TN+FP1+FP2+FN)	0.909
Precision	TP/(TP+FP1+FP2)	0.939
Recall	TP/(TP+FN)	0.953


Setup
Clone the repository:https://github.com/akshaymenon8438/Computer-Vision-Engineering
Run pip3 install -r requirements.txt to install packages required.

