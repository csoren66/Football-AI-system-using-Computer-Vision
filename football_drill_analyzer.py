import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
import mediapipe as mp
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt

@dataclass
class DrillMetrics:
    drill_type: str
    count: int
    intensity: float
    intensity_level: str
    ball_rotation: str
    keypoints: List[Tuple[int, int]]

class DrillDetector:
    """Detects and classifies football drills based on movement patterns."""
    
    def __init__(self):
        self.movement_history = deque(maxlen=30)
        self.pattern_templates = {
            'square': np.array([[0,0], [1,0], [1,1], [0,1]]),  # Square pattern
            'triangle': np.array([[0,0], [1,0.5], [0,1]])      # Triangle pattern
        }
    
    def classify_movement(self, keypoints: List[Tuple[int, int]]) -> str:
        """Classify the movement pattern as either square or triangle."""
        if len(self.movement_history) < 10:
            self.movement_history.append(keypoints)
            return "unknown"
            
        # Normalize movement pattern
        pattern = np.array(list(self.movement_history))
        pattern = self._normalize_pattern(pattern)
        
        # Compare with templates
        square_score = self._pattern_similarity(pattern, self.pattern_templates['square'])
        triangle_score = self._pattern_similarity(pattern, self.pattern_templates['triangle'])
        
        return 'square' if square_score > triangle_score else 'triangle'
    
    def _normalize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Normalize the movement pattern to a standard scale."""
        min_coords = np.min(pattern, axis=0)
        max_coords = np.max(pattern, axis=0)
        return (pattern - min_coords) / (max_coords - min_coords)
    
    def _pattern_similarity(self, pattern: np.ndarray, template: np.ndarray) -> float:
        """Calculate similarity between observed pattern and template."""
        # Reshape pattern to (number of frames, number of keypoints * 2)
        num_frames = pattern.shape[0]
        # Get the number of keypoints from the template instead of the pattern
        num_keypoints = template.shape[0] 
        # Select only the relevant keypoints from the pattern to match the template
        pattern = pattern[:, :num_keypoints]  
        pattern = pattern.reshape(num_frames, num_keypoints * 2)

        # Reshape template to (1, number of keypoints in template * 2)
        template = template.reshape(1, -1)
        
        # Now broadcasting will work as expected
        return -np.mean(np.min(np.linalg.norm(pattern[:, None] - template, axis=2), axis=1))

class PoseEstimator:
    """Handles pose estimation using MediaPipe."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def get_keypoints(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Extract relevant keypoints from the frame."""
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        keypoints = []
        
        if results.pose_landmarks:
            for landmark in [
                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_KNEE
            ]:
                point = results.pose_landmarks.landmark[landmark]
                keypoints.append((
                    int(point.x * frame.shape[1]),
                    int(point.y * frame.shape[0])
                ))
        
        return keypoints

class BallTracker:
    """Tracks football movement and analyzes rotation."""
    
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.ball_history = deque(maxlen=10)
        self.prev_gray = None
        self.flow_points = np.array([[]])
    
    def track_ball(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """Track ball and determine rotation direction."""
        # Detect ball using YOLO
        results = self.model(frame)
        ball_bbox = None
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls[0]) == 32:  # sports ball class
                    ball_bbox = box.xyxy[0].cpu().numpy()
                    break
        
        if ball_bbox is None:
            return None, "unknown"
        
        # Calculate ball center
        ball_center = np.array([
            (ball_bbox[0] + ball_bbox[2]) / 2,
            (ball_bbox[1] + ball_bbox[3]) / 2
        ])
        
        # Update history and calculate rotation
        self.ball_history.append(ball_center)
        rotation = self._analyze_rotation(frame, ball_bbox)
        
        return ball_bbox, rotation
    
    def _analyze_rotation(self, frame: np.ndarray, bbox: np.ndarray) -> str:
        """Analyze ball rotation using optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return "unknown"
        
        # Calculate optical flow in ball region
        ball_roi = gray[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        prev_roi = self.prev_gray[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_roi, ball_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Determine rotation direction based on dominant flow direction
        if flow.size > 0:
            mean_flow = np.mean(flow, axis=(0,1))
            if abs(mean_flow[0]) > abs(mean_flow[1]):
                return "forward" if mean_flow[0] > 0 else "backward"
        
        self.prev_gray = gray
        return "unknown"

class DrillAnalyzer:
    """Main class for analyzing football drills."""
    
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.ball_tracker = BallTracker()
        self.drill_detector = DrillDetector()
        
        self.count = 0
        self.last_count_time = time.time()
        self.counts_buffer = deque(maxlen=5)
        
        self.INTENSITY_THRESHOLDS = {
            'High': 0.4,
            'Medium': 0.3,
            'Low': 0.2
        }
    
    def process_frame(self, frame: np.ndarray) -> DrillMetrics:
        """Process a single frame and return metrics."""
        # Get pose keypoints
        keypoints = self.pose_estimator.get_keypoints(frame)
        
        # Track ball
        ball_bbox, rotation = self.ball_tracker.track_ball(frame)
        
        # Detect drill type
        drill_type = self.drill_detector.classify_movement(keypoints)
        
        # Update metrics
        if self._detect_repetition(keypoints):
            self.count += 1
            self.counts_buffer.append(time.time())
        
        # Calculate intensity
        intensity, intensity_level = self._calculate_intensity()
        
        return DrillMetrics(
            drill_type=drill_type,
            count=self.count,
            intensity=intensity,
            intensity_level=intensity_level,
            ball_rotation=rotation,
            keypoints=keypoints
        )
    
    def _detect_repetition(self, keypoints: List[Tuple[int, int]]) -> bool:
        """Detect if a repetition has occurred."""
        if not keypoints:
            return False
        
        # Calculate vertical movement of feet
        foot_positions = [p[1] for p in keypoints[:2]]  # Use only foot keypoints
        if len(self.drill_detector.movement_history) > 0:
            prev_positions = [p[1] for p in self.drill_detector.movement_history[-1][:2]]
            
            # Detect significant up-down movement
            movement = np.mean(np.abs(np.array(foot_positions) - np.array(prev_positions)))
            return movement > 20
        
        return False
    
    def _calculate_intensity(self) -> Tuple[float, str]:
        """Calculate current intensity based on counts per second."""
        current_time = time.time()
        time_window = current_time - self.last_count_time
        
        if time_window == 0:
            return 0.0, 'Low'
            
        counts_per_second = len(self.counts_buffer) / time_window
        
        # Determine intensity level
        if counts_per_second >= self.INTENSITY_THRESHOLDS['High']:
            level = 'High'
        elif counts_per_second >= self.INTENSITY_THRESHOLDS['Medium']:
            level = 'Medium'
        else:
            level = 'Low'
            
        return counts_per_second, level

def process_video(video_path: str, output_path: str):
    """Process a video file and save the analyzed output."""
    analyzer = DrillAnalyzer()
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame
        metrics = analyzer.process_frame(frame)
        
        # Draw metrics on frame
        frame = draw_metrics(frame, metrics)
        
        out.write(frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def draw_metrics(frame: np.ndarray, metrics: DrillMetrics) -> np.ndarray:
    """Draw metrics overlay on frame."""
    # Draw drill type
    cv2.putText(frame, f"Drill: {metrics.drill_type}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw count
    cv2.putText(frame, f"Count: {metrics.count}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw intensity
    cv2.putText(frame, f"Intensity: {metrics.intensity:.2f} ({metrics.intensity_level})",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw ball rotation
    cv2.putText(frame, f"Ball Rotation: {metrics.ball_rotation}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw keypoints
    for point in metrics.keypoints:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
    
    return frame

def main():
    # Process both videos
    videos = ['Squares.mp4', 'Triangles.mp4']
    for video in videos:
        output_path = f'analyzed_{video}'
        print(f"Processing {video}...")
        process_video(video, output_path)
        print(f"Completed processing {video}")

if __name__ == "__main__":
    main()