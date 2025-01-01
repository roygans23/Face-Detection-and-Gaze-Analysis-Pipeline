import cv2
import os
import json
import numpy as np


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

# Creating a json from the video in the format of {FRAME_INDEX : LIST OF FACES AS TUPLES}
def process_video(cap, output_filename, net, frame_skip=30, conf_threshold=0.7, nms_threshold=0.4):
    """
    Processes a video capture to detect faces every `frame_skip` frames and saves the results to a JSON file.
    Only frames with detected faces will be included in the JSON file.

    Parameters:
    - cap: The cv2.VideoCapture object.
    - output_filename: The name of the JSON file to save the results. 
    - net: The pre-loaded DNN model.
    - frame_skip: Process every `frame_skip` frames.
    - conf_threshold: Confidence threshold for face detection.
    - nms_threshold: Non-Maximum Suppression threshold.
    """
    face_data = {}
    frame_index = 0  # To track the actual frame number

    while True:
        # Skip directly to the next frame to be processed
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        # If the frame is not read correctly, break the loop
        if not ret:
            break

        try:
            # Detect faces in the current frame
            faces = detect_faces(frame, net, conf_threshold=conf_threshold, nms_threshold=nms_threshold)
            
            if faces:
                # Store results as a list of tuples (top-left and bottom-right coordinates)
                face_data[frame_index] = [
                    ((int(f[0][0]), int(f[0][1])), (int(f[1][0]), int(f[1][1]))) for f in faces
                ]
        except Exception as e:
            print(f"Error processing frame {frame_index}: {e}")
            pass
        
        # Move to the next frame to be processed
        frame_index += frame_skip

    # Release the video capture object
    cap.release()

    # Save the face detection data to a JSON file
    with open(output_filename, "w") as f:
        json.dump(face_data, f, indent=4)


def process_videos_in_folder(folder_path, output_folder, net, frame_skip=30, conf_threshold=0.7, nms_threshold=0.4):
    """
    Processes all videos in a folder to create a JSON for each video with face detection data.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(folder_path, file_name)
            base_name, _ = os.path.splitext(file_name)
            output_filename = os.path.join(output_folder, base_name + ".json")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open {video_path}")
                continue

            print(f"Processing video: {video_path}")
            process_video(
                cap,
                output_filename,
                net,
                frame_skip=frame_skip,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold
            )
            cap.release()

