# Show raised hands
## Sample code[Pose]
 - Display "right" when you raise your right hand and "left" when you raise your left hand. In addition, display "both" when you raise your both hands. 
 - The following code flips the image horizontally.
```python
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
```
<image src="../image/pose_q2-1.png" width="30%" height="30%"> <image src="../image/pose_q2-2.png" width="30%" height="30%"> <image src="../image/pose_q2-3.png" width="30%" height="30%"><br>

Type the following template. It's OK copy and paste.

```python
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

device = 0 # cameera device number

def main():
    # For webcam input:
    cap = cv2.VideoCapture(device)
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # judge
            cv2.putText(image, judge_raise_hand(image, results.pose_landmarks), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

# Judgment of raising hand
def judge_raise_hand(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # Convert the obtained landmark values x, y, z to the coordinates on the image
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append(np.array([landmark_x, landmark_y, landmark_z], dtype=int))

    if len(landmark_point) != 0:
        if landmark_point[0][1] > landmark_point[20][1] and landmark_point[0][1] > landmark_point[19][1]:
            return "both"
        elif landmark_point[0][1] > landmark_point[20][1]:
            return "left"
        elif landmark_point[0][1] > landmark_point[19][1]:
            return "right"
        else:
            return ""

if __name__ == '__main__':
    main()
```
