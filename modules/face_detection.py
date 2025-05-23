import cv2
import os
import numpy as np
from tqdm import tqdm


#LOAD THE MODEL
def load_caffe_model(prototxt_path, caffe_model_path):
    """
    Load the Caffe face detection model.
    """
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)
    return net

# Face detection function 
def detect_faces(frame, net, conf_threshold=0.7, nms_threshold=0.4, margin=0.1):
    """
    Detects multiple faces in an image using a pre-trained DNN model with Non-Maximum Suppression,
    with an optional margin around the bounding boxes.

    Parameters:
    - frame: The input image in BGR format.
    - net: The pre-loaded DNN model.
    - conf_threshold: Confidence threshold to filter weak detections.
    - nms_threshold: Non-Maximum Suppression threshold to eliminate overlapping boxes.
    - margin: Margin percentage to add around the face bounding box (default is 10%).

    Returns:
    - A list of tuples: [((x1, y1), (x2, y2)), ...]
      where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the face.
    """
    (h, w) = frame.shape[:2]

    # Preprocess the frame: resize to 300x300 and perform mean subtraction
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                 1.0, (300, 300), 
                                 (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    boxes = []
    confidences = []
    
    # Iterate over detections and collect boxes and confidences
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding boxes fall within the frame dimensions
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            boxes.append([startX, startY, endX - startX, endY - startY])
            confidences.append(float(confidence))
    
    # Apply Non-Maximum Suppression to suppress overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    faces = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, width, height) = boxes[i]

            # Adjust box coordinates to add margin
            margin_x = int(width * margin)
            margin_y = int(height * margin)
            
            x1 = max(0, x - margin_x)  # Top-left x with margin
            y1 = max(0, y - margin_y)  # Top-left y with margin
            x2 = min(w - 1, x + width + margin_x)  # Bottom-right x with margin
            y2 = min(h - 1, y + height + margin_y)  # Bottom-right y with margin

            faces.append(((x1, y1), (x2, y2)))
    
    return faces

def process_videos_in_folder(folder_path, net, frame_skip=30, conf_threshold=0.7, nms_threshold=0.4):
    """
    Processes all videos in a folder and returns a dictionary of face detections.
    Returns: {video_name: {frame_idx: [(top_left, bottom_right), ...]}}
    """
    detection_results = {}

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(folder_path, file_name)
            video_name = os.path.splitext(file_name)[0]
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open {video_path}")
                continue

            # Get number of frames in video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Total frames in {video_name}: {total_frames}")

            num_of_iterations = total_frames // frame_skip + 1
            print(f"Total frames to process in {video_name}: {num_of_iterations}")

            print(f"Processing video: {video_path}")
            frame_results = {}
            frame_idx = 0

            with tqdm(total=num_of_iterations, desc=f"Searching for 'face-detected' frames in {video_name}", unit="frame") as pbar:
                while True:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break

                    try:
                        faces = detect_faces(frame, net, conf_threshold, nms_threshold)
                        if faces:  # Only store frames with faces
                            frame_results[frame_idx] = faces
                    except Exception as e:
                        print(f"Error processing frame {frame_idx}: {e}")
                    
                    frame_idx += frame_skip
                    # Update tqdm progress bar
                    pbar.update(1)

            cap.release()
            if frame_results:  # Only store videos with detected faces
                detection_results[video_name] = frame_results

    return detection_results

