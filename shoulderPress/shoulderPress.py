# https://www.youtube.com/watch?v=We1uB79Ci-w

import mediapipe as mp
import cv2
import numpy as np
import pickle
import pandas as pd

from landmarks import landmarks

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with open('shoulderPress.pkl', 'rb') as f:
    model1 = pickle.load(f)

# with open('lean.pkl', 'rb') as f:
#     model2 = pickle.load(f)

# with open('stance.pkl', 'rb') as f:
#     model3 = pickle.load(f)

cap = cv2.VideoCapture(0)
counter = 0
current_stage = ''

# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # video feed
    while(cap.isOpened()):
        ret,frame = cap.read()

        # recolor image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # make detection
        results = pose.process(image)
        
        # recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        
        # pose landmarks
        mp_drawing.draw_landmarks(image, 
                                  results.pose_landmarks, 
                                  mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
        # detect body language
        try:
            row = np.array([[res.x,res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks)
            body_language_class = model1.predict(X)[0]
            body_language_prob = model1.predict_proba(X)[0]

            # lean_class = model2.predict(X)[0]
            # stance_class = model3.predict(X)[0]
            
            if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] >= 0.6:
                current_stage = 'down'
            elif current_stage == 'down' and body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] >= 0.6:
                current_stage = 'up'
                counter +=1

            
            
            #Get status box
            cv2.rectangle(image, (0,0), (300,60), (245,117,16), -1)

            #Display Class
            cv2.putText(image, 'CLASS'
                        , (115,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (110,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2, cv2.LINE_AA)
            
            #Display Lean
            cv2.putText(image, 'PROB'
                        , (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2))
            # cv2.putText(image, lean_class.split(' ')[0]
                        , (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # #Display Stance
            # cv2.putText(image, 'STANCE'
            #             , (150,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            # #cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2))
            # cv2.putText(image, stance_class.split(' ')[0]
            #             , (145,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            #Display Counter
            cv2.putText(image,'COUNT'
                       , (250,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter)
                       , (245,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2, cv2.LINE_AA)
        except:
            print('passed')
            pass
        
        cv2.imshow("Raw Webcam Feed", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()