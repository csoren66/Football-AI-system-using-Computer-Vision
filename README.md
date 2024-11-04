# Football-Drill-Video-Analysis-System

This project provides an automated solution for analyzing sports drills, focusing on movement patterns, repetition counts, intensity, and ball rotation direction. By utilizing advanced computer vision techniques and machine learning models, this project aims to deliver detailed insights into player performance.

## Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

## Overview
This project implements a pipeline to analyze drills using pose estimation and ball tracking. The solution involves:
1. Identifying body keypoints.
2. Recognizing drill patterns (Square or Triangle).
3. Counting repetitions.
4. Estimating intensity.
5. Tracking ball rotation.

## Demo
[Watch the Demo](#) *(https://drive.google.com/drive/folders/1xuibJqVU6b82dNTVjYdfeevOC4zDea2f)*

## Features
- **Pose Estimation and Pattern Recognition:** Classifies drills by recognizing the player's movement pattern.
- **Repetition Counting:** Counts the number of completed repetitions based on keypoint analysis.
- **Intensity Calculation:** Measures the player's activity level (High, Medium, or Low).
- **Ball Tracking and Rotation Analysis:** Tracks the ball and determines rotation direction.
- **Dynamic Display:** Displays metrics on the video output with clear overlays.

## Setup and Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/csoren66/Football-Drill-Video-Analysis-System.git
    cd Football-Drill-Video-Analysis-System
    ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the videos (Squares.mp4 and Triangles.mp4) from the provided shared drive and place them in the `data/` directory.

## Usage
To run the drill analysis:
```bash
python football_drill_analyzer.py
```

## Methodology

### Data Acquisition and Preprocessing
1. Load videos (`Squares.mp4` and `Triangles.mp4`), ensuring a consistent frame rate and resolution.

### Pose Estimation and Drill Pattern Recognition
- **Pose Estimation:** Utilizes MediaPipe Pose to extract 3D body keypoints.
- **Pattern Recognition:** Analyzes keypoint movements to classify drills as Square or Triangle.

### Repetition Counting
- Tracks foot keypoints to detect up-down patterns, identifying the completion of repetitions.

### Intensity Calculation
- Uses OpenCV's optical flow to calculate speed and acceleration, categorizing intensity based on counts per second.

### Ball Tracking and Rotation Analysis
- Detects the ball using YOLOv8, analyzing optical flow to determine rotation direction.

### Dynamic Display Generation
- Adds annotations for repetition count, intensity level, and ball rotation direction to the video output.

### Testing and Optimization
- Tests were conducted to validate the accuracy and efficiency of the solution on various drill videos.

## Results
The solution provides comprehensive metrics on player performance, including repetition count, movement intensity, and ball rotation, allowing for valuable insights into sports biomechanics.

## License
This project is licensed under the MIT License.
