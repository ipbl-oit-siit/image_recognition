# Show face orientation
## Sample code[Face]
 - Display "smile" when smile. 
 - The position of facial parts is the basic information for classifying facial expressions.<br>
<image src="../image/smile.gif" width="30%" height="30%"><br>

Type the following template. It's OK copy and paste.

```python
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

device = 0 # cameera device number

def main():
    # For webcam input:
    cap = cv2.VideoCapture(device)

    face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        results = face_mesh.process(frame)

        # Draw the face mesh annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                smile_judgement(frame, face_landmarks)
        cv2.imshow('MediaPipe FaceMesh', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

def smile_judgement(image, landmarks):
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

    cx = 0
    cy = 0       
    if len(landmark_point) != 0:
        for i in range(0, len(landmark_point)):
            cx += landmark_point[i][0]
            cy += landmark_point[i][1]

        cx = int(cx/len(landmark_point))
        cy = int(cy/len(landmark_point))

        cv2.circle(image, (int(landmark_point[13][0]),int(landmark_point[13][1])), 5, (255, 255, 255), 3)
        cv2.circle(image, (int(landmark_point[14][0]),int(landmark_point[14][1])), 5, (255, 255, 255), 3)
        cv2.circle(image, (int(landmark_point[78][0]),int(landmark_point[78][1])), 5, (255, 255, 255), 3)
        cv2.circle(image, (int(landmark_point[308][0]),int(landmark_point[308][1])), 5, (255, 255, 255), 3)

        if landmark_point[13][1] < landmark_point[14][1]:
            if landmark_point[78][1] < landmark_point[13][1]:
                if landmark_point[308][1] < landmark_point[13][1]:
                    cv2.putText(image, "smile", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)

        
if __name__ == '__main__':
    main()
```
