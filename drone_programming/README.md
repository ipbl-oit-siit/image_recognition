# Image processing and control for Hula-JP Drone

[back to the top page](../README.md)

---

## Objectives
- This page explains basics of Hula-JP drone control and real-time image analysis with Python3.
- All implementations prioritize a **non-blocking main loop architecture** to prevent UDP stream lag and video buffering delays.

## Prerequisite
- "[Python Environment for iPBL26](https://github.com/ipbl-oit-siit/portal/blob/main/setup/python%2Bvscode.md)" has already been installed.
- The python programs (.py) have to be put under the directory `C:\oit\py26\ipbl`. 
- The custom libraries `my_av2.py` and `detection_timer.py` must be located under the directory `mylibs`.

---

## :green_square: Pre-Flight Safety & Connection Tests
Before performing any actual flight sequence, always execute these non-takeoff tests to ensure safe hardware and video stream operations.

### :red_square: Step 1: Communication & Battery Status Test
- Establish network synchronization and retrieve the current battery level without starting the motors.

#### :o:Practice[ping_and_battery]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_ping.py`)
- `sample_hula_ping.py`
    ```python
    import pyhula
    import sys
    import time

    DRONE_IP = "192.168.100.101"
     
    try:
        api = pyhula.UserApi()
    except Exception as e:
        print(f"[ERROR] Failed to initialize pyhula: {e}")
        sys.exit(1)
     
    def send_test_command(command: str):
        try:
            print(f"[TX]: {command}")
            if command == "command":
                api.connect(DRONE_IP)
                time.sleep(1.0)
                response = "ok"
            elif command == "battery?":
                res_battery = api.get_battery()
                response = str(res_battery)
            else:
                response = "unknown command"
            print(f"[RX]: {response}")
            return response
        except Exception as e:
            print(f"[ERROR] Communication error occurred: {e}")
            sys.exit(1)
     
    def main():
        print("--- Initiating Drone Communication Test ---")
        send_test_command("command")
        battery = send_test_command("battery?")
        print(f"\n[STATUS] Connection successful. Battery Level: {battery}%")
     
    if __name__ == "__main__":
        main()
    ```

---

### :red_square: Step 2: Ground Motor Rotation Test (No Takeoff)
- Spin the propellers at a low idle speed on the ground (`plane_fly_arm`) and stop them (`plane_fly_disarm`) to check motor status without generating lift.

#### :o:Practice[motor_test]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_motor_test.py`)
- `sample_hula_motor_test.py`
    ```python
    import pyhula
    import time
    import sys

    DRONE_IP = "192.168.100.101"

    try:
        api = pyhula.UserApi()
    except Exception as e:
        print(f"[ERROR] Failed to initialize pyhula: {e}")
        sys.exit(1)

    def main():
        if not api.connect(DRONE_IP):
            print("[!!! EMERGENCY !!!] Connection Failed. Aborting motor test.")
            sys.exit(1)
        
        try:
            print("\n--- Starting Propeller Rotation Test (Arming) ---")
            # Turn on motors at idle ground speed
            api.plane_fly_arm()  
            print("Motors spinning at idle speed... checking hardware status.")
            time.sleep(3)
            
            # Shut down motors
            api.plane_fly_disarm()
            print("Motors stopped safely (Disarmed).")
        except KeyboardInterrupt:
            print("\n[USER INTERRUPT] Stopping motors immediately.")
            api.plane_fly_disarm()

    if __name__ == "__main__":
        main()
    ```

---

### :red_square: Step 3: Ground Camera Stream Test (No Takeoff)
- Verify the video pipeline and latency by streaming the camera feed to an OpenCV window while the drone stays securely on the ground.

#### :o:Practice[stream_test]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_stream_test.py`)
- `sample_hula_stream_test.py`
    ```python
    import cv2
    import pyhula
    import sys
    from mylibs.my_av2 import VideoCapture

    DRONE_IP = "192.168.100.101"

    def main():
        try:
            api = pyhula.UserApi()
        except Exception as e:
            print(f"[ERROR] Failed to initialize pyhula: {e}")
            sys.exit(1)

        if not api.connect(DRONE_IP):
            print("[ERROR] Cannot connect to drone control channel.")
            return

        # Pass the initialized api object directly to enable custom Hula SDK stream mode
        cap = VideoCapture(api)

        # Explicitly verify if the UDP/RTP media stream opened successfully
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
- A robust program template that guarantees a safe touchdown (`single_fly_touchdown`) even if the script encounters errors or user interruptions (`Ctrl+C`).

#### :o:Practice[failsafe_template]
- Save the following sample code as a python file and execute it. (`C:\oit\home\ipbl\sample_main_failsafe.py`)
- `sample_main_failsafe.py`
    ```python
    import pyhula
    import sys

    DRONE_IP = "192.168.100.101"

    try:
        api = pyhula.UserApi()
    except Exception as e:
        print(f"[ERROR] Failed to initialize pyhula: {e}")
        sys.exit(1)

    def main():
        if not api.connect(DRONE_IP):
            print("\n[!!! COM LOSS DETECTED !!!] Aborting program.")
            sys.exit(1)
        
        battery = api.get_battery()
        print(f"[STATUS] Initial Battery Check: {battery}%")

        try:
            print("\n--- Starting Flight Sequence ---")
            # --- WRITE YOUR FLIGHT COMMANDS HERE ---
            # api.single_fly_takeoff()
            pass
            # --------------------------------------
        except KeyboardInterrupt:
            print("\n[USER INTERRUPT] Program stopped by user (Ctrl+C).")
        except Exception as e:
            print(f"\n[UNEXPECTED ERROR] {e}")
        finally:
            print("\n[SAFETY] Sending TOUCHDOWN command.")
            try:
                # Force immediate touchdown sequence for safety
                api.single_fly_touchdown() 
            except Exception:
                print("Failed to send touchdown command. Recover manually.")
            print("Program terminated safely.")

    if __name__ == "__main__":
        main()
    ```

---

## :green_square: Real-Time Image Processing & Camera Control
### :red_square: Integration Loop: Continuous Processing & Chattering Prevention
- **Critical Requirement**: To maintain real-time low latency without frame backlog, the main loop must run **completely non-blocking** by constantly pulling frames via `cap.read()`.
- Inside this loop, we inject `DetectionTimer` to handle **Debouncing (Time Stabilization)** and absorb physical hardware delay by controlling transmission intervals without stopping the video frame pipeline.

#### :o:Practice[camera_angle_control]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_vision_control.py`)
- `sample_hula_vision_control.py`
    ```python
    import cv2
    import pyhula
    import sys
    from mylibs.my_av2 import VideoCapture
    from mylibs.detection_timer import DetectionTimer

    DRONE_IP = "192.168.100.101"

    def main():
        try:
            api = pyhula.UserApi()
        except Exception as e:
            print(f"[ERROR] Failed to initialize pyhula: {e}")
            sys.exit(1)
            
        if not api.connect(DRONE_IP):
            print("[ERROR] Connection failed.")
            return

        # 1. Initialize stable timers for up/down gesture detections (e.g., maintain 400ms)
        up_timer = DetectionTimer(target_ms=400.0, grace_ms=200.0)
        down_timer = DetectionTimer(target_ms=400.0, grace_ms=200.0)
        
        camera_angle = 0  # Internal state tracking for the camera gimbal angle (-90 to 90)

        # 2. Connect to the drone video stream via the custom PyAV engine
        cap = VideoCapture(api)
        
        # Guard clause to ensure stream is opened before starting the real-time loop
        if not cap.isOpened():
            print("[ERROR] Cannot connect to drone video stream.")
            return

        print("Streaming active. Control gimbal using vision logic in non-blocking loop...")
        print("Press 'q' in the video window or press Ctrl+C in the terminal to stop.")
        
        try:
            while True:
                # 3. Read the latest frame in a fast loop to completely flush UDP socket buffer
                ret, frame = cap.read()
                if not ret or frame is None:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit requested by user via OpenCV window.")
                        break
                    continue
                
                # Fetch exact high-accuracy timeline (msec) assigned by custom VideoCapture
                current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                # --- [Vision Processing Section Placeholder] ---
                # Realistically, you would process MediaPipe landmarks here.
                # Example flags:
                is_up_detected = False
                is_down_detected = False
                
                # Temporary key binding test to mimic hand tracking for demonstration:
                key_press = cv2.waitKey(1) & 0xFF
                if key_press == ord('u'):    # Hold 'u' key to simulate Up gesture
                    is_up_detected = True
                elif key_press == ord('d'):  # Hold 'd' key to simulate Down gesture
                    is_down_detected = True
                elif key_press == ord('q'):
                    print("Quit requested by user via OpenCV window.")
                    break
                # -------------------------------------------------

                # 4. Feed detection states and chronological milestones into the timers
                up_reached = up_timer.update(is_up_detected, current_msec)
                down_reached = down_timer.update(is_down_detected, current_msec)

                # 5. Process state machine triggers once timers confirm target hold duration
                if up_reached:
                    if camera_angle < 90:
                        camera_angle += 10
                        # Command the hardware to explicitly snap to the target ABSOLUTE value
                        # API: Plane_cmd_camera_angle(type, data)
                        api.Plane_cmd_camera_angle(0, camera_angle)
                        print(f"[GIMBAL UP] Target Stable. Snapping to: {camera_angle} deg")
                    
                    # Force reset the latch state to prepare for the next targeted hold cycle
                    up_timer.is_reached = False
                    up_timer.start_time = None

                elif down_reached:
                    if camera_angle > -90:
                        camera_angle -= 10
                        # Hula API expects positive magnitude for absolute type-1 downwards request
                        api.Plane_cmd_camera_angle(1, abs(camera_angle))
                        print(f"[GIMBAL DOWN] Target Stable. Snapping to: {camera_angle} deg")
                    
                    down_timer.is_reached = False
                    down_timer.start_time = None

                # Render basic HUD telemetry overlays onto the video frame
                cv2.putText(frame, f"Angle: {camera_angle} | Time: {int(current_msec)}ms", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Real-Time Tracking & Control Window", frame)

        except KeyboardInterrupt:
            print("\n[EMERGENCY] Program interrupted by user via terminal (Ctrl+C).")

        finally:
            print("[SAFETY] Cleaning up resources and stabilizing flight state...")
            cap.release()
            cv2.destroyAllWindows()
            print("Video resources cleaned up safely.")

    if __name__ == "__main__":
        main()
    ```

> [!NOTE]
> ### Explanation
> - **`cap.get(cv2.CAP_PROP_POS_MSEC)`**: Retrieves the high-accuracy frame timestamp (in milliseconds) calculated internally by the custom `VideoCapture` class.
> - **`DetectionTimer.update(...)`**: Tracks transient frame-by-frame gesture detection flags over continuous milliseconds. It prevents control-packet flooding by waiting for a predefined duration (`target_ms`) before triggering drone commands, ensuring physical hardware has enough time to catch up.

---

## :green_square: Emergency Management & Safety Routines
### :red_square: Advanced Failsafes & Emergency Methods
- Modular functions to prevent crashes and safely handle flight anomalies based on battery levels.

#### :o:Practice[emergency_routines]
- Save the following sample code as a python file, and execute it. (`C:\oit\home\ipbl\sample_hula_safety.py`)
- `sample_hula_safety.py`
    ```python
    import pyhula

    def trigger_emergency_touchdown(api: pyhula.UserApi):
        """
        Forces the drone to stop any autonomous mission and land immediately.
        Use this handler for fatal vision tracking loss or external flight anomalies.
        """
        print("[EMERGENCY] Failsafe triggered. Forcing immediate touchdown sequence.")
        try:
            api.single_fly_touchdown()
            return True
        except Exception as e:
            print(f"[CRITICAL] Touchdown command failed to dispatch: {e}")
            return False
    ```

---

[back to the top page](../README.md)
