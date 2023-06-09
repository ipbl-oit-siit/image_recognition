# Show the angle pointed by the index finger
## Sample code[Hands]
 - Display the angle between the vertical upward direction and the direction pointed by the index finger as shown in the figure below.
 - In this program we need to calculate the angle using 2D vectors.<br>
    <image src="../image/angle.gif" width="30%" height="30%"><br>

Type the following template. It's OK copy and paste.

```python
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
 
device = 0 # cameera device number

def main():
    # For webcam input:
    cap = cv2.VideoCapture(device)
    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
    hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7)
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
        results = hands.process(image)

        # Draw the index finger annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                show_angle(image, hand_landmarks)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

# check the index finger
def show_angle(image, landmarks):
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

        if len(landmark_point) != 0 and len(landmark_point)==21:
            cv2.circle(image, (landmark_point[6][0], landmark_point[6][1]), 7, (0, 0, 255), 3)
            cv2.circle(image, (landmark_point[7][0], landmark_point[7][1]), 7, (0, 0, 255), 3)
            cv2.circle(image, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 255), 3)
            cv2.line(image, (landmark_point[6][0], landmark_point[6][1]), (landmark_point[7][0], landmark_point[7][1]), (0, 255, 0))
            cv2.line(image, (landmark_point[7][0], landmark_point[7][1]), (landmark_point[8][0], landmark_point[8][1]), (0, 255, 0))

            # calculate the angle between the index finger and vertical direction
            ifinger_vec = (int(landmark_point[8][0]-landmark_point[6][0]), 
            int(landmark_point[8][1]-landmark_point[6][1]))

            vertical_vec = (0, -1)

            A = np.array(ifinger_vec)
            B = np.array(vertical_vec)
            angle = int(v_angle(A, B))

            if(landmark_point[8][0] - landmark_point[6][0] < 0):
                angle = 360 - angle

            cv2.putText(image, str(angle), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)

def v_angle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)

    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)

    return np.degrees(np.arccos(cos_theta))

if __name__ == '__main__':
    main()
```
