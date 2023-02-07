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
