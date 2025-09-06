import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load model
def load_model():
    """Load the trained classifier"""
    model_path = "models/rps_landmark_clf.joblib"
    if not os.path.exists(model_path):
        print("Error: No trained model found!")
        print("Please run train_model.py first to train the classifier.")
        return None
    return joblib.load(model_path)

def extract_features(landmarks):
    """Extract normalized features from hand landmarks"""
    pts = np.array(landmarks, dtype=np.float32)
    
    # Normalize: translate to wrist (idx 0) & scale by hand size
    wrist = pts[0].copy()
    pts -= wrist
    
    # Use distance to middle MCP as scale
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    
    return pts.flatten()

def main():
    print("Rock-Paper-Scissors Live Recognition")
    print("Show gestures to your webcam. Press 'q' to quit.")
    
    # Load classifier
    classifier = load_model()
    if classifier is None:
        return
    
    # Class labels
    labels = ["rock", "paper", "scissors"]
    
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    
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
                
                # Extract features and predict
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [
                    (hand_landmarks.landmark[i].x, 
                     hand_landmarks.landmark[i].y, 
                     hand_landmarks.landmark[i].z) 
                    for i in range(21)
                ]
                
                features = extract_features(landmarks).reshape(1, -1)
                
                # Get prediction and confidence
                probabilities = classifier.predict_proba(features)[0]
                predicted_class = int(np.argmax(probabilities))
                confidence = probabilities[predicted_class]
                
                prediction_text = labels[predicted_class].upper()
                confidence_text = f"Confidence: {confidence:.2f}"
                
                # Color based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
            else:
                color = (255, 255, 255)  # White for no detection
            
            # Add text overlay
            cv2.putText(img, prediction_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            if confidence_text:
                cv2.putText(img, confidence_text, (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add instructions
            cv2.putText(img, "Press 'q' to quit", (50, img.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Rock Paper Scissors Recognition", img)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped.")

if __name__ == "__main__":
    main()
