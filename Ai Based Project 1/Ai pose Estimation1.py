#!/usr/bin/env python
# coding: utf-8

# In[77]:


get_ipython().system('pip install mediapipe opencv.python')


# In[78]:


import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
my_pose = mp.solutions.pose


# In[79]:


import tensorflow as tf




# In[80]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('Mediapipe Feed',frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[81]:


cap = cv2.VideoCapture(0)
with my_pose.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret,frame = cap.read()
    
    #detect stuff and render
    #recolor image
    #converting into bgr
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
    #make detection
        results = pose.process(image)
        image.flags.writeable = True
    #converting into rgb
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        
     
    #render detection
        mp_drawing.draw_landmarks(image,results.pose_landmarks,my_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66),thickness = 2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                 )
    
    
        cv2.imshow('Mediapipe Feed',image)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


# In[82]:


get_ipython().run_line_magic('pinfo2', 'mp_drawing.DrawingSpec')


# In[83]:


#Determining the joints of the body


# In[84]:


# <img src="https://learnopencv.com/wp-content/uploads/2022/03/MediaPipe-pose-BlazePose-Topology.jpg" alt="Alt text">


# In[85]:


cap = cv2.VideoCapture(0)
with my_pose.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret,frame = cap.read()
    
    #detect stuff and render
    #recolor image
    #converting into bgr
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
    #make detection
        results = pose.process(image)
        image.flags.writeable = True
    #converting into rgb
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass
        
        
     
    #render detection
        mp_drawing.draw_landmarks(image,results.pose_landmarks,my_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66),thickness = 2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                 )
    
    
        cv2.imshow('Mediapipe Feed',image)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:





# In[86]:


len(landmarks)


# In[87]:


#mqpping of landmarks
for lndmark in my_pose.PoseLandmark:
    print(lndmark.value)


# In[88]:


landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].visibility


# In[89]:


landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value]


# In[90]:


landmarks[my_pose.PoseLandmark.LEFT_WRIST.value]


# In[91]:


my_pose.PoseLandmark.LEFT_SHOULDER


# In[92]:


#Calculate Angles


# In[93]:


def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle>180.0:
        angle = 360-angle
    
    return angle


# In[94]:


shoulder = [landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].y]
wrist = [landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].y]


# In[95]:


shoulder,elbow,wrist


# In[96]:


calculate_angle(shoulder,elbow,wrist)


# In[99]:


tuple(np.multiply(elbow,[640,480]).astype(int))


# In[104]:


cap = cv2.VideoCapture(0)
with my_pose.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret,frame = cap.read()
    
    #detect stuff and render
    #recolor image
    #converting into bgr
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
    #make detection
        results = pose.process(image)
        image.flags.writeable = True
    #converting into rgb
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            #get coodinate
            shoulder = [landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            #calculate angle
            angle = calculate_angle(shoulder,elbow,wrist)
            
            #visualize 
            cv2.putText(image,str(angle),
                       tuple(np.multiply(elbow,[640,480]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
                             )

            print(landmarks)
        except:
            pass
        
        
     
    #render detection
        mp_drawing.draw_landmarks(image,results.pose_landmarks,my_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66),thickness = 2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                 )
    
    
        cv2.imshow('Mediapipe Feed',image)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


# In[64]:


print(shoulder,elbow,wrist)


# In[108]:


#curl counter
cap = cv2.VideoCapture(0)

#curl counter variables
counter = 0
stage = None

with my_pose.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret,frame = cap.read()
    
    #detect stuff and render
    #recolor image
    #converting into bgr
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
    
    #make detection
        results = pose.process(image)
        image.flags.writeable = True
    #converting into rgb
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            #get coodinate
            shoulder = [landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            #calculate angle
            angle = calculate_angle(shoulder,elbow,wrist)
            
            #visualize 
            cv2.putText(image,str(angle),
                       tuple(np.multiply(elbow,[640,480]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA
                             )

            #curl counter logic
            if angle>160:
                stage = "down"
            if angle<30 and stage=="down":
                stage = "up"
                counter+=1
                print(counter)
                
        except:
            pass
        
        
        #render curl counter
        #setup status box
        cv2.rectangle(image,(0,0),(225,73),(245,117,16),-1)
        
        #rep data
        cv2.putText(image,'REPS',(15,12),
                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,str(counter),
                   (10,60),
                   cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
        
        
        #rep data
        cv2.putText(image,'STAGE',(65,12),
                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(image,stage,
                   (60,60),
                   cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
        
     
    #render detection
        mp_drawing.draw_landmarks(image,results.pose_landmarks,my_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66),thickness = 2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)
                                 )
    
    
        cv2.imshow('Mediapipe Feed',image)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




