# Show face orientation
## Sample code[Face]
 - Display "left" or "right" according to the orientation of the face.
 - There are several judgment methods, but the simple one is a judgment method that compares the x-coordinates of several landmarks.<br>
    <image src="../image/q2_face.gif" width="30%" height="30%"><br>

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
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
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
                #mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)
                face_orientation(frame, face_landmarks)
        cv2.imshow('MediaPipe FaceMesh', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

def face_orientation(image, landmarks):
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
            cv2.circle(image, (int(landmark_point[i][0]),int(landmark_point[i][1])), 1, (0, 255, 0), 1)
        cx = int(cx/len(landmark_point))
        cy = int(cy/len(landmark_point))

        # top
        cv2.circle(image, (int(landmark_point[10][0]),int(landmark_point[10][1])), 5, (255, 255, 0), 1)
        # bottom
        cv2.circle(image, (int(landmark_point[152][0]),int(landmark_point[152][1])), 5, (255, 255, 0), 1)
        # left
        cv2.circle(image, (int(landmark_point[234][0]),int(landmark_point[234][1])), 5, (255, 255, 0), 1)
        # right
        cv2.circle(image, (int(landmark_point[454][0]),int(landmark_point[454][1])), 5, (255, 255, 0), 1)
        # center
        cv2.circle(image, (int(landmark_point[0][0]),int(landmark_point[0][1])), 5, (255, 255, 0), 1)

        l = cx - landmark_point[234][0]
        r = landmark_point[454][0] - cx

        if abs(l) > 5*abs(r):
            cv2.putText(image, "right", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
        elif 5*abs(l) < abs(r):
            cv2.putText(image, "left", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
        
if __name__ == '__main__':
    main()
```
