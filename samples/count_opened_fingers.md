# Count opened fingers
## Sample code[Hands]
 - Count and display the number of fingers opened.
    - It is necessary to think about the judgment method with each finger.
    - To be robust to changes in hand orientation, we need to use 3D vectors to calculate the angle.<br>
    <image src="../image/num.gif" width="30%" height="30%"><br>

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
                count_fingers(image, hand_landmarks)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

# Count and display the number of fingers
def count_fingers(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    tipids = [4,8,12,16,20]   #list of all landmarks of the tips of fingers

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # Convert the obtained landmark values x, y, z to the coordinates on the image
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append(np.array([landmark_x, landmark_y, landmark_z], dtype=int))

        if len(landmark_point) != 0 and len(landmark_point)==21:
            fingerlist=[]

            #thumb and dealing with flipping of hands
            if landmark_point[12][0] > landmark_point[20][0]:
                if landmark_point[tipids[0]][0] > landmark_point[tipids[0]-1][0]:
                    fingerlist.append(1)
                else:
                    fingerlist.append(0)
            else:
                if landmark_point[tipids[0]][0] < landmark_point[tipids[0]-1][0]:
                    fingerlist.append(1)
                else:
                    fingerlist.append(0)

            #others
            for index in range (1,5):
                vec1 = (landmark_point[tipids[index]-3][0] - landmark_point[tipids[index]-2][0], 
                landmark_point[tipids[index]-3][1] - landmark_point[tipids[index]-2][1], 
                landmark_point[tipids[index]-3][2] - landmark_point[tipids[index]-2][2])
                vec2 = (landmark_point[tipids[index]-1][0] - landmark_point[tipids[index]-2][0], 
                landmark_point[tipids[index]-1][1] - landmark_point[tipids[index]-2][1], 
                landmark_point[tipids[index]-1][2] - landmark_point[tipids[index]-2][2])
                cv2.circle(image, (landmark_point[tipids[index]-3][0], landmark_point[tipids[index]-3][1]), 7, (0, 0, 255), 3)
                cv2.circle(image, (landmark_point[tipids[index]-2][0], landmark_point[tipids[index]-2][1]), 7, (0, 0, 255), 3)
                cv2.circle(image, (landmark_point[tipids[index]-1][0], landmark_point[tipids[index]-1][1]), 7, (0, 0, 255), 3)
                cv2.line(image, (landmark_point[5][0], landmark_point[5][1]), (landmark_point[6][0], landmark_point[6][1]), (0, 255, 0))
                cv2.line(image, (landmark_point[6][0], landmark_point[6][1]), (landmark_point[7][0], landmark_point[7][1]), (0, 255, 0))

                if v_angle3d(vec1, vec2) > 140:
                    fingerlist.append(1)
                else:
                    fingerlist.append(0)


            if len(fingerlist)!=0:
                fingercount=fingerlist.count(1)

            cv2.putText(image, str(fingercount), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)

def v_angle3d(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)

    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)

    return np.rad2deg(np.arccos(cos_theta))

if __name__ == '__main__':
    main()
```
