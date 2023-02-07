#!/usr/bin/env python
# coding: utf-8

# # Player Service Fault Detection 

# In[22]:


import cv2
from cv2 import destroyAllWindows
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# In[28]:


cap = cv2.VideoCapture('C:/Users/JAYA MENON/OneDrive - Livent Corporation/Desktop/DAPA-CA2/rally_video/video/frame/Recording #4.mp4')
 # Check if video path is correct else apply correct path
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
        
        # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
            
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
           
            
            # Concate rows
            row = pose_row
            
            # Append class name 
            row.insert(0, class_name)
            
            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
            
        except:
            pass
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[29]:


for lndmrk in mp_pose.PoseLandmark: ##### 33 body landmarks
    print(lndmrk)


# In[30]:


len(landmarks)


# #As we need to calculate FOOT NOT STATIONARY service fault detection we will calculate knee movement with respect to foot

# In[32]:


FOOT = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
HEEL = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]


# In[34]:


FOOT,KNEE,HEEL


# In[35]:


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


# In[43]:


angle_knee = calculate_angle(FOOT,KNEE,HEEL)
angle_knee


# In[45]:


knee_angle = 180-angle_knee
knee_angle


# In[101]:



cap = cv2.VideoCapture('C:/Users/JAYA MENON/OneDrive - Livent Corporation/Desktop/DAPA-CA2/rally_video/video/frame/Recording #4.mp4')
# Initiate pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = pose.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        
        except:
            pass
       
        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[99]:


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# In[102]:


results.pose_landmarks.landmark[0].visibility


# In[103]:


# Get coordinates on left leg
hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
heel_l = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z]
f_index_l = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

       


# In[114]:


hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
hip_l


# In[105]:


angle_l = calculate_angle(f_index_l, knee_l, heel_l)
angle_l


# In[109]:


angle_left = 180-angle_l
angle_left


# In[104]:


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# In[157]:


angle_l = calculate_angle(f_index_l, knee_l, heel_l)
angle_left = 180-angle_l
angle_left


# In[233]:


import csv
import os
import numpy as np


# In[234]:


num_coords = len(results.pose_landmarks.landmark)
num_coords


# In[235]:


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


# In[236]:


landmarks


# In[237]:


# exporting body landmark on csv
with open('C:/Users/JAYA MENON/OneDrive - Livent Corporation/Desktop/DAPA-CA2/rally_video/video/frame/coordinates.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


# In[238]:


class_name = " service fault detection"


# In[112]:


pose = results.pose_landmarks.landmark
pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
pose_row


# In[248]:


cap = cv2.VideoCapture('C:/Users/JAYA MENON/OneDrive - Livent Corporation/Desktop/DAPA-CA2/rally_video/video/frame/Recording #4.mp4')
 # Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
# We need to set resolutions.


# Curl counter variables
stage = None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True: 
                
            
        # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print (landmarks)
            
            # Get coordinates
            #COORDINATES 
        #FOOT =[landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        #HEEL =[landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        #KNEE =[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
            # Calculate angle
            #angle = calculate_angle(shoulder, elbow, wrist)
            #angle_foot = calculate_angle(KNEE,FOOT,HEEL)
            #foot_angle = 180-angle_foot
            
            # Visualize angle
            cv2.putText(image, str(knee_l), 
                           tuple(np.multiply(KNEE, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Curl counter logic
            if angle_left > 150:
                stage = "Foot Not stationary"
            else :
                stage="Not Foul"
                #print (stage)
                       
        except:
            pass
        
        # Render curl counter
        # Setup status box
        #cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        
        # Stage data
        cv2.putText(image, 'Service Foul detection', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 ) 
        
        
        cv2.imshow('Mediapipe Feed', image)
        # Write the frame into the
        # file 'filename.avi'
        #result.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            # When everything done, release 
            # the video capture and video 
            # write objects
            
    
    #result.release()

    cap.release()
    cv2.destroyAllWindows()


# In[ ]:


#The Pose Landmark keypoints are flattened to get data frame


# In[260]:


pose = results.pose_landmarks.landmark
pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
len(pose_row)           
           
            
            
             


# In[261]:


# Concate rows
row = pose_row
row.insert(0, 'Not Foul')
row            
          


# In[262]:


row


# In[263]:


with open('C:/Users/JAYA MENON/OneDrive - Livent Corporation/Desktop/DAPA-CA2/rally_video/video/frame/coordinates.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(pose_row)
      


# # Train Model

# In[420]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[421]:


df = pd.read_csv('C:/Users/JAYA MENON/OneDrive - Livent Corporation/Desktop/DAPA-CA2/rally_video/video/frame/coords.csv')


# In[422]:


df.head()


# In[423]:


df.tail()


# In[401]:


df1=df.dropna(axis='columns')


# In[402]:


df1


# In[478]:


df[df['class']=='Foot Not stationary']


# In[403]:


print("cleaned dataset shape =", df1.shape)


# In[404]:


from sklearn.preprocessing import LabelEncoder

#create instance of label encoder
lab = LabelEncoder()

#perform label encoding on 'team' column
df1['class'] = lab.fit_transform(df1['class'])


# In[424]:


df1.head()


# In[ ]:





# In[405]:


print(df1.dtypes)


# In[479]:


X = df.drop('class', axis=1) # features
y = df['class'] # target value


# In[569]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
#from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
import seaborn as sns


# In[570]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[571]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[573]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[574]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[576]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) #svc model fit
svm = SVC(random_state=42)
svm.fit(X_train, y_train)


# In[577]:


svm_disp = RocCurveDisplay.from_estimator(svm, X_test, y_test)
plt.show()


# In[579]:


rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
svm_disp.plot(ax=ax, alpha=0.8)
plt.show()


# In[581]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[584]:


pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    'DT': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}


# In[585]:


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model


# In[586]:


fit_models


# In[590]:


import warnings
warnings.filterwarnings("ignore")


# In[591]:


fit_models['rc'].predict(X_test)


# In[532]:


fit_models['lr'].predict(X_test)


# In[588]:


#DATA MINING  MODEL

models = {
    
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    #"KNN": KNeighborsClassifier()
   #"Naive Bayes": GaussianNB(),
    "SVC": SVC()

}


# In[533]:


from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve


# In[589]:


for name, model in models.items():
    print(f'Training Model {name} \n--------------')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Training Accuracy: {accuracy_score(y_train, model.predict(X_train))}')
    print(f'Testing Accuracy: {accuracy_score(y_test, y_pred)}')
   # print(f'Testing Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    print(f'Testing Recall: {recall_score(y_test, y_pred )}')    
    print(f'Testing Precesion: {precision_score(y_test, y_pred)}')
    print(f'Testing F-1: {f1_score(y_test, y_pred)}')
    #print(f'Testing F-Beta: {fbeta_score(y_test, y_pred, beta=0.5)}')
    print(f'Testing classification:{classification_report(y_test, y_pred)}')
    print('-'*30)


# In[495]:


fit_models['rf'].predict(X_test)


# In[496]:


y_test


# In[ ]:


################ As Random Forest is the best model we will use RF for testing of the machine learning model#####################


# In[497]:


with open('service_foul.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)


# In[498]:


with open('service_foul.pkl', 'rb') as f:
    model = pickle.load(f)


# In[500]:


model


# In[520]:


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
cap = cv2.VideoCapture('C:/Users/JAYA MENON/OneDrive - Livent Corporation/Desktop/DAPA-CA2/rally_video/video/frame/Recording #4.mp4')
 # Check if video path entered properly
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.

result = cv2.VideoWriter('C:/Users/JAYA MENON/OneDrive - Livent Corporation/Desktop/DAPA-CA2/rally_video/video/frame/mediapipe.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (frame_width,frame_height))

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )


        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Hand landmarks
            hand= results.right_hand.landmark
            hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in hand]).flatten())
            
            # Concate rows
            row = pose_row+hand_row
            
#             # Append class name 
#             row.insert(0, class_name)
            
#             # Export to CSV
#             with open('coords.csv', mode='a', newline='') as f:
#                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 csv_writer.writerow(row) 

            # Make Detections
            X = pd.DataFrame([row])
            service_fault_class = model.predict(X)[0]
            service_fault_prob = model.predict_proba(X)[0]
            print(service_fault_class, service_fault_prob)
            
            # Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y))
                        , [640,480]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)
        result.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            

        
            
    
result.release()

cap.release()
cv2.destroyAllWindows()


# In[521]:


tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x, 
results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y)), [640,480]).astype(int))

