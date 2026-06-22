# Image processing and control for Hula-JP Drone

[back to the top page](../README.md)

---

## Objectives
- This page explains basics of Hula-JP drone control and real-time image analysis with Python3.
- All implementations prioritize a **non-blocking main loop architecture** to prevent UDP stream lag and video buffering delays.

## Prerequisite
- "[Python Environment for iPBL26](https://github.com/ipbl-oit-siit/portal/blob/main/setup/python%2Bvscode.md)" has already been installed.
- The python programs (.py) have to be put under the directory `C:\oit\py26\ipbl`. 
- The custom libraries `my_av2.py`, `detection_timer.py`, and `safe_drone_watcher.py` must be located under the directory `my_libs`.

---

## :green_square: Pre-Flight Safety & Connection Tests
Before performing any actual flight sequence, always execute these non-takeoff tests to ensure safe hardware and video stream operations.

### :red_square: Step 1: Communication & Battery Status Test
- Establish network synchronization and retrieve the current battery level without starting the motors.

#### :o:Practice[ping_and_battery]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\hula_ping.py`)
- `hula_ping.py`
    ```python
    import sys
    import time
    import pyhula

    DRONE_IP = "192.168.100.116"
    
    def main():
        # 1. Connect first
        try:
            api = pyhula.UserApi()
            print("Connecting to drone at ", DRONE_IP, "...")
            api.connect(DRONE_IP)
            time.sleep(3.0)
        except Exception as e:
            print(f"[ERROR] Failed to setup drone: {e}")
            sys.exit(1)
     
        # 2. Execute target communication task
        print("--- Initiating Drone Communication Test ---")
        try:
            battery = api.get_battery()
            print(f"\n[STATUS] Connection successful. Battery Level: {battery}%")
        except Exception as e:
            print(f"[ERROR] Communication error occurred: {e}")
            sys.exit(1)
     
    if __name__ == "__main__":
        main()
    ```

---

### :red_square: Step 2: Ground Motor Rotation Test (No Takeoff)
- Spin the propellers at a low idle speed on the ground (`plane_fly_arm`) and stop them (`plane_fly_disarm`) to check motor status without generating lift.

#### :o:Practice[motor_test]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\hula_motor_test.py`)
- `hula_motor_test.py`
    ```python
    import sys
    import time
    import pyhula
    from my_libs.safe_drone_watcher import SafeDroneWatcher

    DRONE_IP = "192.168.100.116"
    
    def main():
        # 1. Connect first
        try:
            api = pyhula.UserApi()
            print("Connecting to drone at ", DRONE_IP, "...")
            api.connect(DRONE_IP)
            time.sleep(3.0)
        except Exception as e:
            print(f"[ERROR] Failed to setup drone: {e}")
            sys.exit(1)

        # 2. Activate watcher protection right after connection
        with SafeDroneWatcher(api):
            # 3. Safe flight command logic sequence
            print("\n--- Starting Propeller Rotation Test (Arming) ---")
            api.plane_fly_arm()  
            print("Motors spinning at idle speed... checking hardware status.")
            time.sleep(3)
            
            api.plane_fly_disarm()
            print("Motors stopped safely (Disarmed).")

    if __name__ == "__main__":
        main()
    ```

---

### :red_square: Step 3: Ground Camera Stream Test (No Takeoff)
- Verify the video pipeline and latency by streaming the camera feed to an OpenCV window while the drone stays securely on the ground.

#### :o:Practice[stream_test]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\hula_stream_test.py`)
- `hula_stream_test.py`
    ```python
    import sys
    import time
    import cv2
    import pyhula
    from my_libs.safe_drone_watcher import SafeDroneWatcher
    from my_libs.my_av2 import VideoCapture

    DRONE_IP = "192.168.100.116"
    
    def main():
        # 1. Connect first
        try:
            api = pyhula.UserApi()
            print("Connecting to drone at ", DRONE_IP, "...")
            api.connect(DRONE_IP)
            time.sleep(3.0)
        except Exception as e:
            print(f"[ERROR] Failed to setup drone: {e}")
            sys.exit(1)

        # 2. Activate watcher protection right after connection
        with SafeDroneWatcher(api):
            # 3. Enter main loop stream pipeline
            cap = VideoCapture(api)
            if not cap.isOpened():
                print("[ERROR] Cannot open drone video stream. Check Wi-Fi connection.")
                return

            print("\n--- Video Stream Started ---")
            print("Press 'q' inside the video window to quit.")

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                cv2.imshow("Hula-JP Ground Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Closing video stream...")
                    break

            cap.release()
            cv2.destroyAllWindows()
            print("Resources released successfully.")

    if __name__ == "__main__":
        main()
    ```

---

## :green_square: Flight Control & Safety
### :red_square: Main Control Template with Integrated Failsafes
- A robust boilerplate template using `SafeDroneWatcher`. It automatically tracks telemetry and forces emergency routines (`touchdown` or `disarm`) if the script encounters errors or terminal user interruptions (`Ctrl+C`).

#### :o:Practice[failsafe_template]
- Save the following sample code as a python file and execute it. (`C:\oit\home\ipbl\main_failsafe.py`)
- `main_failsafe.py`
    ```python
    import sys
    import time
    import pyhula
    from my_libs.safe_drone_watcher import SafeDroneWatcher

    DRONE_IP = "192.168.100.116"
    
    def main():
        # 1. Connect first
        try:
            api = pyhula.UserApi()
            print("Connecting to drone at ", DRONE_IP, "...")
            api.connect(DRONE_IP)
            time.sleep(3.0)
        except Exception as e:
            print(f"[ERROR] Failed to setup drone: {e}")
            sys.exit(1)
    
        # 2. Activate watcher protection right after connection
        with SafeDroneWatcher(api):
            # 3. Spin the propellers continuously on the ground to test emergency intervention
            print("\n--- Safe Watchdog Test Loop Activated ---")
            print("[STATUS] Arming motors... Propellers are now spinning at low idle speed.")
            api.plane_fly_arm()
            
            print("\n>>> PRESS [Ctrl + C] IN THIS TERMINAL TO TEST EMERGENCY FAILSAFE! <<<")
            print("The Watchdog system will catch the interrupt and automatically shut down the motors.")
            
            # Keep idling until the user triggers a terminal keyboard interrupt
            while True:
                time.sleep(1.0)
                
    if __name__ == "__main__":
        main()
    ```

---

## :green_square: Real-Time Image Processing & Camera Control
### :red_square: Integration Loop: Continuous Processing & Chattering Prevention
- In the previous module (`imp_time_trigger.py`), you learned how to use `DetectionTimer` to verify a target over continuous milliseconds using your webcam. 
- Now, we apply this exact same **Time-Based Verification System** to actual drone control. In this practice, we inject `DetectionTimer` into a non-blocking flight loop to stabilize manual gimbal adjustments via keyboard inputs without disrupting the real-time video stream pipeline.

#### :o:Practice[camera_angle_control]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\hula_vision_control.py`)
- `hula_vision_control.py`
    ```python
    import sys
    import time
    import cv2
    import pyhula
    from my_libs.safe_drone_watcher import SafeDroneWatcher
    from my_libs.my_av2 import VideoCapture
    from my_libs.detection_timer import DetectionTimer

    DRONE_IP = "192.168.100.116"
    
    def main():
        # 1. Connect first
        try:
            api = pyhula.UserApi()
            print(f"Connecting to drone at {DRONE_IP}...")
            api.connect(DRONE_IP)
            time.sleep(3.0)
        except Exception as e:
            print(f"[ERROR] Failed to setup drone: {e}")
            sys.exit(1)
    
        # 2. Activate watcher protection right after connection
        with SafeDroneWatcher(api):
            # 3. Enter real-time tracking and non-blocking loop structure
            up_timer = DetectionTimer(target_ms=400.0, grace_ms=200.0)
            down_timer = DetectionTimer(target_ms=400.0, grace_ms=200.0)
            camera_angle = 0  

            cap = VideoCapture(api)
            if not cap.isOpened():
                print("[ERROR] Cannot connect to drone video stream.")
                return

            print("Streaming active. Control gimbal using vision logic in non-blocking loop...")
            print("Press 'q' in the video window to stop.")
            
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit requested by user via OpenCV window.")
                        break
                    continue
                
                current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                is_up_detected = False
                is_down_detected = False
                
                key_press = cv2.waitKey(1) & 0xFF
                if key_press == ord('u'):    
                    is_up_detected = True
                elif key_press == ord('d'):  
                    is_down_detected = True
                elif key_press == ord('q'):
                    print("Quit requested by user via OpenCV window.")
                    break

                up_reached = up_timer.update(is_up_detected, current_msec)
                down_reached = down_timer.update(is_down_detected, current_msec)

                if up_reached:
                    if camera_angle < 90:
                        camera_angle += 10
                        # Prepare arguments for the API
                        direction_flag = 0 if camera_angle >= 0 else 1
                        api.Plane_cmd_camera_angle(direction_flag, abs(camera_angle))
                        print(f"[GIMBAL UP] Target Stable. Snapping to: {camera_angle} deg")
                    up_timer.is_reached = False
                    up_timer.start_time = None

                elif down_reached:
                    if camera_angle > -90:
                        camera_angle -= 10
                        # Prepare arguments for the API
                        direction_flag = 0 if camera_angle >= 0 else 1
                        api.Plane_cmd_camera_angle(direction_flag, abs(camera_angle))
                        print(f"[GIMBAL DOWN] Target Stable. Snapping to: {camera_angle} deg")
                    down_timer.is_reached = False
                    down_timer.start_time = None

                cv2.putText(frame, f"Angle: {camera_angle} deg | Time: {int(current_msec)}ms", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Real-Time Tracking & Control Window", frame)

            cap.release()
            cv2.destroyAllWindows()
            print("Video resources cleaned up safely.")

    if __name__ == "__main__":
        main()
    ```

> [!NOTE]
> ### Explanation
> - **`cap.get(cv2.CAP_PROP_POS_MSEC)`**: Retrieves the high-accuracy frame timestamp (in milliseconds) calculated internally by the custom `VideoCapture` class.
> - **`DetectionTimer.update(is_detected, current_msec)`**: 
>   Just like the ARuCo card locking system you practiced earlier (`imp_time_trigger.py`), this method tracks how long a signal stays active to filter out unstable physical chattering.
>   * **Target Lock (400ms)**: Instead of snapping the camera gimbal the exact millisecond a key is touched, it requires the input signal to be held for `target_ms=400.0` before sending commands. This prevents flooding the drone with excessive control packets.
>   * **Debouncing Grace (200ms)**: Standard OS keyboard inputs naturally stutter (briefly drop to `False`) when held down. The built-in `grace_ms=200.0` safety buffer ensures that a fraction of a second of key signal drop won't instantly wipe out your accumulated timer progress.
> - **`api.Plane_cmd_camera_angle(direction, angle)`**:
>   This specific API method requires two distinct arguments to change the gimbal's physical tilt direction:
>   * **`direction`**: Takes `0` for horizontal or upward positions (positive angles), and `1` for downward positions (negative angles).
>   * **`angle`**: Requires a **positive absolute value** (`0` to `90`). Passing a negative number directly will cause an internal system crash (`struct.error`).
>   * *Implementation Tip*: In our code, we map a single, intuitive `camera_angle` integer (`-90` to `90`) into these two hardware parameters dynamically using `direction_flag = 0 if camera_angle >= 0 else 1` and `abs(camera_angle)`.

---

#### :o:Practice[hover_and_failsafe_test]
- Now let's reuse the exact same `DetectionTimer` class to track flight mission durations. We will execute a 5-second hover mission while verifying that emergency interrupts (`q` or `Ctrl+C`) function instantly during active processing.
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\hula_hover_test.py`)
- `hula_hover_test.py`
    ```python
    import sys
    import time
    import cv2
    import pyhula
    from my_libs.safe_drone_watcher import SafeDroneWatcher
    from my_libs.my_av2 import VideoCapture
    from my_libs.detection_timer import DetectionTimer

    DRONE_IP = "192.168.100.116"

    def main():
        try:
            api = pyhula.UserApi()
            print(f"Connecting to drone at {DRONE_IP}...")
            api.connect(DRONE_IP)
            time.sleep(3.0)
        except Exception as e:
            print(f"[ERROR] Failed to setup drone: {e}")
            sys.exit(1)

        with SafeDroneWatcher(api):
            cap = VideoCapture(api)
            if not cap.isOpened():
                print("[ERROR] Cannot connect to drone video stream.")
                return

            # Reuse DetectionTimer to monitor a 5-second (5000ms) continuous hover state
            hover_timer = DetectionTimer(target_ms=5000.0)
            is_takeoff_commanded = False

            print("Video stream loop started.")
            print(">>> TO INTERRUPT & TOUCHDOWN: Press 'q' in the window OR [Ctrl+C] in the terminal <<<")

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    continue

                current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                # Trigger takeoff on the very first loop iteration to avoid blocking window rendering
                if not is_takeoff_commanded:
                    print("\n--- Dispatching Takeoff Command (Non-blocking window open) ---")
                    api.single_fly_takeoff()
                    is_takeoff_commanded = True
                    
                    # Reset the baseline timeline to exclude takeoff initialization delay
                    hover_timer.start_time = None
                    hover_timer.lost_time = None

                # Continuously pass True since the drone is actively maintaining its hover state
                is_hover_completed = hover_timer.update(True, current_msec)

                if is_hover_completed:
                    print("\n[SUCCESS] 5-second hover time elapsed.")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INTERRUPT] 'q' key pressed. Breaking loop for landing.")
                    break

                # Display pure hover status on the video feed
                cv2.putText(frame, f"HOVERING ACTIVE | Time: {int(current_msec)}ms", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Real-Time Flight Control Feed", frame)

            print("Sending safe touchdown command...")
            api.single_fly_touchdown()
            
            cap.release()
            cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()
    ```
---

[back to the top page](../README.md)
