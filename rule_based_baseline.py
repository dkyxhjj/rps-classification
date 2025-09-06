import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calculate_finger_angle(landmarks, finger_indices):
    """Calculate angle at finger joint"""
    p1 = np.array([landmarks[finger_indices[0]].x, landmarks[finger_indices[0]].y])
    p2 = np.array([landmarks[finger_indices[1]].x, landmarks[finger_indices[1]].y])
    p3 = np.array([landmarks[finger_indices[2]].x, landmarks[finger_indices[2]].y])
    
    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = math.acos(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(angle)

def is_finger_extended(landmarks, finger_indices, threshold=160):
    """Check if finger is extended based on angle"""
    angle = calculate_finger_angle(landmarks, finger_indices)
    return angle > threshold

def classify_gesture_rule_based(landmarks):
    """Rule-based gesture classification"""
    # Finger landmark indices: [MCP, PIP, TIP] for each finger
    finger_indices = {
        'thumb': [2, 3, 4],
        'index': [5, 6, 8],
        'middle': [9, 10, 12],
        'ring': [13, 14, 16],
        'pinky': [17, 18, 20]
    }
    
    # Check which fingers are extended
    extended_fingers = []
    
    # Thumb (special case - check x-coordinate difference)
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    if abs(thumb_tip.x - thumb_mcp.x) > 0.05:  # Threshold for thumb extension
        extended_fingers.append('thumb')
    
    # Other fingers
    for finger, indices in finger_indices.items():
        if finger != 'thumb' and is_finger_extended(landmarks, indices):
            extended_fingers.append(finger)
    
    num_extended = len(extended_fingers)
    
    # Classification logic
    if num_extended == 0 or num_extended == 1:
        return "rock", 0.8
    elif num_extended >= 4:
        return "paper", 0.8
    elif num_extended == 2 and 'index' in extended_fingers and 'middle' in extended_fingers:
        return "scissors", 0.8
    else:
        return "unknown", 0.5

def main():
    print("Rule-Based Rock-Paper-Scissors Recognition")
    print("This is a baseline using finger counting. Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = hands.process(rgb)
            
            # Default prediction
            prediction_text = "No hand detected"
            confidence_text = ""
            
            # If hand detected, make prediction
            if results.multi_hand_landmarks:
                # Draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                
                # Rule-based classification
                hand_landmarks = results.multi_hand_landmarks[0]
                gesture, confidence = classify_gesture_rule_based(hand_landmarks.landmark)
                
                prediction_text = gesture.upper()
                confidence_text = f"Confidence: {confidence:.2f}"
                
                # Color based on confidence
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 255, 255)  # Yellow
            else:
                color = (255, 255, 255)  # White
            
            # Add text overlay
            cv2.putText(img, prediction_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            if confidence_text:
                cv2.putText(img, confidence_text, (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add method label
            cv2.putText(img, "Rule-Based Method", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add instructions
            cv2.putText(img, "Press 'q' to quit", (50, img.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Rule-Based RPS Recognition", img)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped.")

if __name__ == "__main__":
    main()
