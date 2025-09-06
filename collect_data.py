import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create data directory
os.makedirs("data", exist_ok=True)
X, y = [], []

def extract_features(landmarks):
    # landmarks: list of (x,y,z) in [0,1] image coords
    pts = np.array(landmarks, dtype=np.float32)
    
    # Normalize: translate to wrist (idx 0) & scale by hand size
    wrist = pts[0].copy()
    pts -= wrist
    
    # Use distance to middle MCP as scale
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    
    return pts.flatten()  # 63-dimensional feature vector

def main():
    cap = cv2.VideoCapture(1)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        
        last_save = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for mirror effect
            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = hands.process(rgb)
            
            # Draw landmarks if hand detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
            
            # Add instructions and sample count
            cv2.putText(img, "r=rock, p=paper, s=scissors, q=quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Samples collected: {len(y)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Data Collection", img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key in [ord('r'), ord('p'), ord('s')]:
                if results.multi_hand_landmarks:
                    # Extract landmarks
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = [
                        (hand_landmarks.landmark[i].x, 
                         hand_landmarks.landmark[i].y, 
                         hand_landmarks.landmark[i].z) 
                        for i in range(21)
                    ]
                    
                    # Extract features and add to dataset
                    features = extract_features(landmarks)
                    X.append(features.tolist())
                    
                    # Map key to label
                    label_map = {"r": 0, "p": 1, "s": 2}
                    y.append(label_map[chr(key)])
                    
                    # Print progress every 2 seconds
                    if time.time() - last_save > 2:
                        gesture_counts = {0: 0, 1: 0, 2: 0}
                        for label in y:
                            gesture_counts[label] += 1
                        print(f"Samples - Rock: {gesture_counts[0]}, "
                              f"Paper: {gesture_counts[1]}, "
                              f"Scissors: {gesture_counts[2]}")
                        last_save = time.time()
                else:
                    print("No hand detected! Make sure your hand is visible.")
                    
            elif key == ord('q'):
                break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Save data
    if len(X) > 0:
        data = {"X": X, "y": y}
        with open("data/rps_landmarks.json", "w") as f:
            json.dump(data, f)
        
        # Print final statistics
        gesture_counts = {0: 0, 1: 0, 2: 0}
        for label in y:
            gesture_counts[label] += 1
    else:
        print("No data collected!")

if __name__ == "__main__":
    main()
