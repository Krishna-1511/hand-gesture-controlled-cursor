import cv2
import mediapipe as mp
import math
import numpy as np
import pyautogui
import time
from collections import deque


SMOOTH_WINDOW = 5
FRAME_W = 1280
FRAME_H = 720  

pyautogui.FAILSAFE = False


prev_volume = 0.0          
last_volume_time = 0.0
last_zoom_time = 0.0
last_vsign_time = 0.0
last_action_time = 0.0
prev_pinch_right = None

VOLUME_COOLDOWN = 0.25
ZOOM_COOLDOWN = 0.20
V_SIGN_COOLDOWN = 0.7


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_module = mp_hands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.7,
                              min_tracking_confidence=0.7)

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    joints = [2, 6, 10, 14, 18]
    fingers = []

    #Thumb
    try:
        fingers.append(1 if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[joints[0]].x else 0)
    except:
        fingers.append(0)

    #Other fingers
    for i in range(1, 5):
        try:
            fingers.append(1 if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[joints[i]].y else 0)
        except:
            fingers.append(0)

    return fingers

def detect_gesture(f):
    if f == [0,0,0,0,0]:
        return "fist"
    elif f == [1,1,1,1,1]:
        return "open"
    elif f == [1,1,1,0,0]:
        return "point"
    elif f == [1,1,0,0,0]:
        return "pinch_mode"
    elif f == [0,1,1,0,0]:
        return "v_sign"
    elif f == [0,1,0,0,0]:
        return "cursor"
    return "unknown"

def norm_dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def main():
    global prev_volume, last_volume_time, last_zoom_time, last_vsign_time
    global prev_pinch_right, last_action_time

    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_W)
    cap.set(4, FRAME_H)

    pts_history = deque(maxlen=SMOOTH_WINDOW)
    vel_history = deque(maxlen=6)
    volume_history = deque(maxlen=3)

    print("Hand Gesture System Running...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_module.process(rgb)

        gesture_display = ""
        action_display = ""
        hand_display = ""

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label 
                hand_display = label

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)

                pinch_px = math.hypot(x2 - x1, y2 - y1)

                fingers = fingers_up(hand_landmarks)
                gesture = detect_gesture(fingers) 
                gesture_display = gesture

                if gesture == "open":
                    if time.time() - last_action_time > 0.5:
                        pyautogui.press('space')
                        action_display = "Play/Pause"
                        last_action_time = time.time()

                elif gesture == "point":
                    if time.time() - last_action_time > 0.4:
                        if thumb.x < index.x:
                            pyautogui.hotkey('shift', 'n')
                            action_display = "Next"
                        else:
                            pyautogui.hotkey('shift', 'p')
                            action_display = "Previous"
                        last_action_time = time.time()

                
                elif gesture == "v_sign":
                    if time.time() - last_vsign_time > V_SIGN_COOLDOWN:
                        pyautogui.hotkey('ctrl', 'win', 'right')
                        action_display = "Next Desktop"
                        last_vsign_time = time.time()

                
                elif gesture == "fist":
                    mx = np.interp(x1, [0, w], [0, pyautogui.size()[0]])
                    my = np.interp(y1, [0, h], [0, pyautogui.size()[1]])
                    pyautogui.moveTo(int(mx), int(my))
                    action_display = "Move Cursor"

              
                elif gesture == "cursor":
                    if time.time() - last_action_time > 1:
                        pyautogui.click()
                        action_display = "Click"
                        last_action_time = time.time()

            
                elif gesture == "pinch_mode":

                    
                    if label == "Left":
                        volume_history.append(pinch_px)

                        if len(volume_history) == 3:
                            smooth_dist = sum(volume_history) / 3
                            volume = np.interp(smooth_dist, [20, 200], [0, 100])

                            if abs(volume - prev_volume) > 7 and (time.time() - last_volume_time > VOLUME_COOLDOWN):
                                if volume > prev_volume:
                                    pyautogui.press("up")
                                    action_display = "Volume Up"
                                else:
                                    pyautogui.press("down")
                                    action_display = "Volume Down"
       
                                prev_volume = volume
                                last_volume_time = time.time()

                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    
                    elif label == "Right":                            
                        now = time.time()

                        if prev_pinch_right is None:
                            prev_pinch_right = pinch_px

                        diff = pinch_px - prev_pinch_right

                        if abs(diff) > 5 and (now - last_zoom_time) > ZOOM_COOLDOWN:
                            if diff > 0:
                                pyautogui.hotkey('ctrl', '+')
                                action_display = "Zoom In"
                            else:
                                pyautogui.hotkey('ctrl', '-')
                                action_display = "Zoom Out"

                            last_zoom_time = now

                        prev_pinch_right = pinch_px

       
        cv2.rectangle(frame, (0, FRAME_H - 90), (460, FRAME_H), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {gesture_display}", (10, FRAME_H - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"Action: {action_display}", (10, FRAME_H - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break
      
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    

   
