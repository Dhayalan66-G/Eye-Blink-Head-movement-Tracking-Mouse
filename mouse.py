import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import csv
from datetime import datetime
import os
import cv2
import pyautogui
import dlib
import threading
import time
import subprocess
import threading
import json
from pynput.mouse import Controller, Button
from AngleBuffer import AngleBuffer
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)
cam = cv2.VideoCapture(0)
# Initialize the Flask app

# Initialize sensitivity variable
sensitivity = 1.0
click_threshold = 0.5  # Threshold for click detection (e.g., distance of gaze from the edge of the screen)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize camera
LEFT_BLINK_COUNT = 0
RIGHT_BLINK_COUNT = 0
LEFT_EYES_BLINK_FRAME_COUNTER = 0
RIGHT_EYES_BLINK_FRAME_COUNTER = 0
USER_FACE_WIDTH = 140 
NOSE_TO_CAMERA_DISTANCE = 600
PRINT_DATA = True
DEFAULT_WEBCAM = 0
SHOW_ALL_FEATURES = True
LOG_DATA = True
LOG_ALL_FEATURES = False
ENABLE_HEAD_POSE = True
LOG_FOLDER = "logs"
SERVER_IP = "127.0.0.1"
SERVER_PORT = 7070
SHOW_ON_SCREEN_DATA = True
TOTAL_BLINKS = 0
EYES_BLINK_FRAME_COUNTER = 0
BLINK_THRESHOLD = 0.51
EYE_AR_CONSEC_FRAMES = 2
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
MOVING_AVERAGE_WINDOW = 10
initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False
PRINT_DATA = True  
DEFAULT_WEBCAM = 0  
SHOW_ALL_FEATURES = True  
LOG_DATA = True  
LOG_ALL_FEATURES = False  
LOG_FOLDER = "logs"  
SERVER_IP = "127.0.0.1" 
SERVER_PORT = 7070  
SHOW_BLINK_COUNT_ON_SCREEN = True 
TOTAL_BLINKS = 0 
EYES_BLINK_FRAME_COUNTER = (
    0  
)
BLINK_THRESHOLD = 0.51  
EYE_AR_CONSEC_FRAMES = (
    2 )
SERVER_ADDRESS = (SERVER_IP, SERVER_PORT)
IS_RECORDING = False  
parser = argparse.ArgumentParser(description="Eye Tracking Application")
parser.add_argument(
    "-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM)
)
args = parser.parse_args()
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  
L_H_RIGHT = [133]  
R_H_LEFT = [362]  
R_H_RIGHT = [263]  
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
_indices_pose = [1, 33, 61, 199, 263, 291]
SERVER_ADDRESS = (SERVER_IP, 7070)
tracking_active = True
if not cam.isOpened():
    print("Error: Could not open camera")
    exit()

# Initialize pynput mouse controller
mouse_controller = Controller()

# Initialize CSV for logging
LOG_FOLDER = "logs"
csv_file_path = os.path.join(LOG_FOLDER, "gaze_tracking_data.csv")
column_names = [
    "Timestamp", "Gaze X", "Gaze Y", "Pitch", "Yaw", "Roll", "Face Position"
]

# Create log file if it doesn't exist
if not os.path.exists(csv_file_path):
    os.makedirs(LOG_FOLDER, exist_ok=True)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        # Blinking Effect Variable
blink_counter = 0
blink_interval = 30  # Frames between blink toggles (adjust as needed)
# Gaze Estimation Logic
def generate_tracking_data():
    global tracking_data
    while tracking_active:
        # Simulate face position changes (replace with actual tracking logic)
        tracking_data["x"] = (tracking_data["x"] + 1) % 640  # x position
        tracking_data["y"] = (tracking_data["y"] + 1) % 480  # y position
        tracking_data["status"] = "Active"
        time.sleep(1)
def estimate_gaze(landmarks, frame_width, frame_height):
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263]
    FACE_ORIENTATION_POINTS = [234, 454, 117, 152, 10]

    left_eye_center = np.mean([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in LEFT_EYE_INDICES], axis=0)
    right_eye_center = np.mean([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in RIGHT_EYE_INDICES], axis=0)
    face_orientation_pts = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in FACE_ORIENTATION_POINTS])
    head_orientation = np.mean(face_orientation_pts, axis=0)

    gaze_x = (left_eye_center[0] + right_eye_center[0] + head_orientation[0]) / 3
    gaze_y = (left_eye_center[1] + right_eye_center[1] + head_orientation[1]) / 3

    screen_width, screen_height = pyautogui.size()  # Get screen size for scaling
    gaze_x = min(max(gaze_x, 0), 1) * screen_width * sensitivity
    gaze_y = min(max(gaze_y, 0), 1) * screen_height * sensitivity

    return gaze_x, gaze_y, left_eye_center, right_eye_center, head_orientation

# Normalize pitch angle
def normalize_pitch(pitch):
    # Normalize pitch to a range, e.g., between -45 to 45 degrees
    normalized_pitch = np.clip(pitch, -45, 45)
    if pitch > 180:
        pitch -= 360
    pitch = -pitch
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch   
    pitch = -pitch
    pitch == normalized_pitch
    return normalized_pitch

# Mouse control based on gaze using pynput
def control_mouse(gaze_x, gaze_y):
    global blink_counter  # Accessing the global blink counter

    try:
        screen_width, screen_height = pyautogui.size()
        current_mouse_position = mouse_controller.position
        gaze_position = (screen_width - np.clip(gaze_x, 0, screen_width),
                         np.clip(gaze_y, 0, screen_height))

        # Move mouse only if there is a significant change (based on sensitivity)
        if abs(gaze_position[0] - current_mouse_position[0]) > 4 or abs(gaze_position[1] - current_mouse_position[1]) > 4:
            # Move mouse smoothly based on sensitivity
            move_x = (gaze_position[0] - current_mouse_position[0]) * sensitivity
            move_y = (gaze_position[1] - current_mouse_position[1]) * sensitivity
            mouse_controller.move(move_x, move_y)

        # Click functionality based on gaze position proximity to screen edge
        if gaze_position[0] > screen_width - click_threshold or gaze_position[1] > screen_height - click_threshold:
            mouse_controller.click(Button.left)

        # Blinking text toggling
        blink_counter += 1
        if blink_counter % blink_interval == 0:
            return True  # Time to "blink" (toggle visibility)
        return False

    except Exception as e:
        print(f"Error in mouse control: {e}")
        return False
    
    # Log data to CSV
def log_data(gaze_x, gaze_y, left_eye_center, right_eye_center, head_orientation):
    # Get timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Head pose estimation (pitch, yaw, roll)
    pitch = np.arctan2(head_orientation[1], head_orientation[0]) * 180 / np.pi
    yaw = np.arctan2(left_eye_center[0] - right_eye_center[0], left_eye_center[1] - right_eye_center[1]) * 180 / np.pi
    roll = 0  # Placeholder, can be calculated more precisely using additional landmarks

    # Normalize the pitch angle
    normalized_pitch = normalize_pitch(pitch)

    # Log to CSV
    row_data = [
        timestamp, gaze_x, gaze_y, normalized_pitch, yaw, roll, "Neutral"
    ]

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)

def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1
def euclidean_distance_3D(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    distance = numerator / denominator
    return distance
def estimate_head_pose(landmarks, image_size):
    scale_factor = USER_FACE_WIDTH / 150.0
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        # Chin
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     # Left eye left corner
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      # Right eye right corner
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    # Left Mouth corner
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      # Right mouth corner
    ])
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )
    dist_coeffs = np.zeros((4,1))
    image_points = np.array([
        landmarks[NOSE_TIP_INDEX],            # Nose tip
        landmarks[CHIN_INDEX],                # Chin
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
        landmarks[LEFT_MOUTH_CORNER_INDEX],      # Left mouth corner
        landmarks[RIGHT_MOUTH_CORNER_INDEX]      # Right mouth corner
    ], dtype="double")
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()[:3]
    pitch = normalize_pitch(pitch)
    return pitch, yaw, roll
def blinking_ratio_left(landmarks):
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])
    left_eye_ratio = (left_eye_ratio + 1) / 2
    return left_eye_ratio
def blinking_ratio_right(landmarks):
    double_blink_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])
    double_blink_ratio = (left_eye_ratio + double_blink_ratio + 1) / 2
    return double_blink_ratio
if PRINT_DATA:
    print("Initializing the face mesh and camera...")
    if PRINT_DATA:
        head_pose_status = "enabled" if ENABLE_HEAD_POSE else "disabled"
        print(f"Head pose estimation is {head_pose_status}.")
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
cam_source = int(args.camSource)
cap = cv.VideoCapture(cam_source)
iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
csv_data = []
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)
column_names = [
    "Timestamp (ms)",
    "Left Eye Center X",
    "Left Eye Center Y",
    "Right Eye Center X",
    "Right Eye Center Y",
    "Left Iris Relative Pos Dx",
    "Left Iris Relative Pos Dy",
    "Right Iris Relative Pos Dx",
    "Right Iris Relative  Pos Dy",
    "Total Blink Count",
]
if ENABLE_HEAD_POSE:
    column_names.extend(["Pitch", "Yaw", "Roll"])
    
if LOG_ALL_FEATURES:
    column_names.extend(
        [f"Landmark_{i}_X" for i in range(468)]
        + [f"Landmark_{i}_Y" for i in range(468)]
    )

    def control_mouse(gaze_x, gaze_y):
     global blink_counter  # Accessing the global blink counter
def generate_frames():
    cap = cv2.VideoCapture(0)
    LEFT_BLINK_COUNT = 0
    RIGHT_BLINK_COUNT = 0
    LEFT_EYES_BLINK_FRAME_COUNTER = 0
    RIGHT_EYES_BLINK_FRAME_COUNTER = 0
    global EYES_BLINK_FRAME_COUNTER
    # Function logic using EYES_BLINK_FRAME_COUNTER
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)  # Adjust size for smoothing
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Estimate gaze and head pose
                gaze_x, gaze_y, left_eye_center, right_eye_center, head_orientation = estimate_gaze(
                    face_landmarks, frame.shape[1], frame.shape[0])

                # Head pose estimation (pitch, yaw, roll)
                pitch = np.arctan2(head_orientation[1], head_orientation[0]) * 180 / np.pi
                yaw = np.arctan2(left_eye_center[0] - right_eye_center[0], left_eye_center[1] - right_eye_center[1]) * 180 / np.pi
                roll = 0  # Placeholder (or can be calculated more precisely if needed)

                # Normalize the pitch and yaw angles
                normalized_pitch = normalize_pitch(pitch)
                normalized_yaw = np.clip(yaw, -45, 45)  # Limit yaw to a reasonable range

                # Blinking effect: toggle visibility every `blink_interval` frames
                is_blinking = control_mouse(gaze_x, gaze_y)

                # Display text for pitch, yaw, gaze, etc. and apply blinking effect
                font_size = 0.5  # Small font size
                thickness = 1  # Thin text
                color = (0, 255, 0)  # Green text

                
                # Gaze coordinates
                cv2.putText(frame, f"Gaze X: {gaze_x:.2f}", (10, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), thickness, cv2.LINE_AA)
                cv2.putText(frame, f"Gaze Y: {gaze_y:.2f}", (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), thickness, cv2.LINE_AA)

                # Log the data
                log_data(gaze_x, gaze_y, left_eye_center, right_eye_center, head_orientation)

            # Encode frame to JPEG for Flask streaming
            ret, jpeg = cv2.imencode('.jpg', frame)  
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )
            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )
            head_pose_points_3D = np.multiply(
                mesh_points_3D[_indices_pose], [img_w, img_h, 1]
            )
            head_pose_points_2D = mesh_points[_indices_pose]
            nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
            nose_2D_point = head_pose_points_2D[0]
            focal_length = 1 * img_w
            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
            head_pose_points_3D = head_pose_points_3D.astype(np.float64)
            head_pose_points_2D = head_pose_points_2D.astype(np.float64)
            success, rot_vec, trans_vec = cv.solvePnP(
                head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
            )
            rotation_matrix, jac = cv.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)
            angle_x = angles[0] * 360
            angle_y = angles[1] * 360
            z = angles[2] * 360
            threshold_angle = 10
            if angle_y < -threshold_angle:
                face_looks = "Right"
                pyautogui.hscroll(100)
            elif angle_y > threshold_angle:
                face_looks = "Left"
                pyautogui.hscroll(-100)
            elif angle_x < -threshold_angle:
                face_looks = "Down"
                pyautogui.scroll(-100)
            elif angle_x > threshold_angle:
                face_looks = "Up"
                pyautogui.scroll(100)
            else:
                face_looks = "Forward"
            if SHOW_ON_SCREEN_DATA:
                cv.putText(
                    frame,
                    f"Face Looking at {face_looks}",
                    (img_w - 400, 80),
                    cv.FONT_HERSHEY_TRIPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
            nose_3d_projection, jacobian = cv.projectPoints(
                nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
            )

            p1 = nose_2D_point
            p2 = (
                int(nose_2D_point[0] + angle_y * 10),
                int(nose_2D_point[1] - angle_x * 10),
            )

            cv.line(frame, p1, p2, (255, 0, 255), 3)
            left_eye_ratio = blinking_ratio_left(mesh_points_3D)
            double_blink_ratio = blinking_ratio_right(mesh_points_3D)
            # Variables to track the state of blinks
            if left_eye_ratio <= BLINK_THRESHOLD and not RIGHT_EYES_BLINK_FRAME_COUNTER:
                LEFT_EYES_BLINK_FRAME_COUNTER += 1
                pyautogui.click(button='left') == LEFT_EYES_BLINK_FRAME_COUNTER
            else:
                if LEFT_EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                 LEFT_BLINK_COUNT += 1
                LEFT_EYES_BLINK_FRAME_COUNTER = 0
            # Check for right eye blink
            if double_blink_ratio <= BLINK_THRESHOLD and  face_looks == "Forward" :
                RIGHT_EYES_BLINK_FRAME_COUNTER += 1 
                pyautogui.click(button='right') == RIGHT_EYES_BLINK_FRAME_COUNTER
            else:
                if RIGHT_EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                    RIGHT_BLINK_COUNT += 1
                RIGHT_EYES_BLINK_FRAME_COUNTER = 0
            if SHOW_ALL_FEATURES:
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(
                frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Left iris
            cv.circle(
                frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Right iris
            cv.circle(
                frame, mesh_points[LEFT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Left eye right corner
            cv.circle(
                frame, mesh_points[LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Left eye left corner
            cv.circle(
                frame, mesh_points[RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Right eye right corner
            cv.circle(
                frame, mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Right eye left corner

            # Calculating relative positions
            l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER], center_left)
            r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)

            # Printing data if enabled
            if PRINT_DATA:
                print(f"Left Eye Blink : {LEFT_BLINK_COUNT}")
                print(f"right Eye Blink : {RIGHT_BLINK_COUNT}")
                print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
                print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
                print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
                print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}\n")
                # Check if head pose estimation is enabled
                if ENABLE_HEAD_POSE:
                    pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                    angle_buffer.add([pitch, yaw, roll])
                    pitch, yaw, roll = angle_buffer.get_average()

                    # Set initial angles on first successful estimation or recalibrate
                    if PRINT_DATA:
                            print("Head pose recalibrated.")

                    # Adjust angles based on initial calibration
                    if calibrated:
                        pitch -= initial_pitch
                        yaw -= initial_yaw
                        roll -= initial_roll
                    if PRINT_DATA:
                        print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
            # Logging data
            if LOG_DATA:
                timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
                log_entry = [
                    timestamp,
                    l_cx,
                    l_cy,
                    r_cx,
                    r_cy,
                    l_dx,
                    l_dy,
                    r_dx,
                    r_dy,
                    LEFT_BLINK_COUNT,  # Left eye blink count
                    RIGHT_BLINK_COUNT,  # Right eye blink count
                ]  # Include blink count in CSV
                log_entry = [timestamp, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy,  LEFT_BLINK_COUNT, 
                    RIGHT_BLINK_COUNT]  # Include blink count in CSV
                
                # Append head pose data if enabled
                if ENABLE_HEAD_POSE:
                    log_entry.extend([pitch, yaw, roll])
                csv_data.append(log_entry)
                if LOG_ALL_FEATURES:
                    log_entry.extend([p for point in mesh_points for p in point])
                csv_data.append(log_entry)

            # Sending data through socket
            timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
            # Create a packet with mixed types (int64 for timestamp and int32 for the rest)
            packet = np.array([timestamp], dtype=np.int64).tobytes() + np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32).tobytes()

            SERVER_ADDRESS = ("127.0.0.1", 7070)
            iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            iris_socket.sendto(packet, SERVER_ADDRESS)

            print(f'Sent UDP packet to {SERVER_ADDRESS}: {packet}')
        # Writing the on screen data on the frame
            if SHOW_ON_SCREEN_DATA:
                if IS_RECORDING:
                    cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle at the top-left corner
                cv.putText(frame,f"Left Eye Blink for Mouse left Click: {LEFT_BLINK_COUNT}",(30, 110),cv.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),2,cv.LINE_AA,)
                cv.putText(frame,f"right Eye Blink for Mouse right click: {RIGHT_BLINK_COUNT}",(30, 140),cv.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2,cv.LINE_AA,)
                if ENABLE_HEAD_POSE:
                    cv.putText(frame, f"Pitch: {int(pitch)}", (30, 240), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 300, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Yaw: {int(yaw)}", (30, 210), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 300, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Roll: {int(roll)}", (30, 180), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 300, 0), 2, cv.LINE_AA)
        key = cv.waitKey(1) & 0xFF
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
                # Encode frame to JPEG and send it to the browser
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Yield frame to be displayed in the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()  # Release the camera when done

# Video feed route to stream frames
@app.route('/video_feed')
def video_feed():
    if tracking_active:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        # Return a blank image while not tracking
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', blank_frame)
        return Response(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Start tracking route
@app.route('/start', methods=['POST'])
def start_tracking():
    global tracking_active
    if not tracking_active:
        tracking_active = False
        threading.Thread(target=generate_frames, daemon=True).start()  # Start tracking in a separate thread
        return jsonify({'status': 'started'}), 200
    else:
        return jsonify({'status': 'already_running'}), 400

# Stop tracking route
@app.route('/stop', methods=['POST'])
def stop_tracking():
    global tracking_active
    tracking_active = True
    return jsonify({'status': 'stopped'}), 200

# Home route with the Start button
@app.route('/')
def index():
    return render_template('index.html')  # This will render the HTML page with the Start button

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

