import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import random

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class RPSGame:
    def __init__(self):
        self.classifier = self.load_model()
        self.labels = ["rock", "paper", "scissors"]
        self.emojis = ["✊", "✋", "✌️"]
        self.player_score = 0
        self.computer_score = 0
        self.round_number = 1
        self.game_state = "waiting"  # waiting, countdown, playing, result
        self.countdown_start = 0
        self.result_start = 0
        self.computer_choice = None
        self.player_choice = None
        self.last_prediction = None
        self.prediction_confidence = 0
        
    def load_model(self):
        model_path = "models/rps_landmark_clf.joblib"
        if not os.path.exists(model_path):
            print("Warning: No trained model found! Using rule-based method.")
            return None
        return joblib.load(model_path)
    
    def extract_features(self, landmarks):
        pts = np.array(landmarks, dtype=np.float32)
        wrist = pts[0].copy()
        pts -= wrist
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts /= scale
        return pts.flatten()
    
    def predict_gesture(self, landmarks):
        if self.classifier is not None:
            features = self.extract_features(landmarks).reshape(1, -1)
            probabilities = self.classifier.predict_proba(features)[0]
            predicted_class = int(np.argmax(probabilities))
            confidence = probabilities[predicted_class]
            return predicted_class, confidence
        else:
            return self.rule_based_prediction(landmarks)
    
    def rule_based_prediction(self, landmarks):
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        extended = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:  # tip above pip
                extended += 1
        
        if extended <= 1:
            return 0, 0.7  # rock
        elif extended >= 4:
            return 1, 0.7  # paper
        else:
            return 2, 0.7  # scissors
    
    def get_winner(self, player, computer):
        if player == computer:
            return 0
        elif (player == 0 and computer == 2) or \
             (player == 1 and computer == 0) or \
             (player == 2 and computer == 1):
            return 1
        else:
            return 2
    
    def start_new_round(self):
        """Start a new round"""
        self.game_state = "countdown"
        self.countdown_start = time.time()
        self.computer_choice = random.randint(0, 2)
        self.player_choice = None
    
    def update_game_state(self, prediction, confidence):
        current_time = time.time()
        
        if self.game_state == "waiting":
            pass
        
        elif self.game_state == "countdown":
            # 3-second countdown
            elapsed = current_time - self.countdown_start
            if elapsed >= 3.0:
                self.game_state = "playing"
                # Capture player's gesture at end of countdown
                if prediction is not None and confidence > 0.6:
                    self.player_choice = prediction
                else:
                    self.player_choice = None  # No valid gesture detected
        
        elif self.game_state == "playing":
            # Show result for 3 seconds
            if current_time - self.countdown_start >= 3.2:  # Small delay after countdown
                self.game_state = "result"
                self.result_start = current_time
                
                # Update scores
                if self.player_choice is not None:
                    winner = self.get_winner(self.player_choice, self.computer_choice)
                    if winner == 1:
                        self.player_score += 1
                    elif winner == 2:
                        self.computer_score += 1
        
        elif self.game_state == "result":
            # Show result for 3 seconds
            if current_time - self.result_start >= 3.0:
                self.game_state = "waiting"
                self.round_number += 1
    
    def draw_game_ui(self, img, prediction, confidence):
        """Draw game UI on the image"""
        h, w = img.shape[:2]
        
        # Draw scores
        cv2.putText(img, f"Player: {self.player_score}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Computer: {self.computer_score}", (w-250, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Round: {self.round_number}", (w//2-50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.game_state == "waiting":
            cv2.putText(img, "Press SPACE to start new round", (w//2-200, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        elif self.game_state == "countdown":
            elapsed = time.time() - self.countdown_start
            countdown = max(0, 3 - int(elapsed))
            if countdown > 0:
                cv2.putText(img, str(countdown), (w//2-30, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
            else:
                cv2.putText(img, "SHOOT!", (w//2-80, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        elif self.game_state == "result":
            # Show choices
            player_text = "No gesture" if self.player_choice is None else self.labels[self.player_choice]
            computer_text = self.labels[self.computer_choice]
            
            cv2.putText(img, f"You: {player_text}", (50, h-150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Computer: {computer_text}", (50, h-100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show winner
            if self.player_choice is None:
                result_text = "No gesture detected!"
                color = (0, 0, 255)
            else:
                winner = self.get_winner(self.player_choice, self.computer_choice)
                if winner == 0:
                    result_text = "TIE!"
                    color = (255, 255, 0)
                elif winner == 1:
                    result_text = "YOU WIN!"
                    color = (0, 255, 0)
                else:
                    result_text = "COMPUTER WINS!"
                    color = (0, 0, 255)
            
            cv2.putText(img, result_text, (w//2-100, h//2+100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Show current prediction
        if prediction is not None and confidence > 0.5:
            pred_text = f"{self.labels[prediction]} ({confidence:.2f})"
            cv2.putText(img, pred_text, (50, h-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    
    game = RPSGame()
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
            
            # Default values
            prediction = None
            confidence = 0
            
            # If hand detected, make prediction
            if results.multi_hand_landmarks:
                # Draw landmarks (commented out to remove balloon-like visuals)
                # for hand_landmarks in results.multi_hand_landmarks:
                #     mp_drawing.draw_landmarks(
                #         img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                #     )
                
                # Get prediction
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [
                    (hand_landmarks.landmark[i].x, 
                     hand_landmarks.landmark[i].y, 
                     hand_landmarks.landmark[i].z) 
                    for i in range(21)
                ]
                
                prediction, confidence = game.predict_gesture(landmarks)
            
            # Update game state
            game.update_game_state(prediction, confidence)
            
            # Draw game UI
            game.draw_game_ui(img, prediction, confidence)
            
            # Show frame
            cv2.imshow("Rock Paper Scissors Game", img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and game.game_state == "waiting":
                game.start_new_round()
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"Final Score - Player: {game.player_score}, Computer: {game.computer_score}")

if __name__ == "__main__":
    main()
