# feature_extraction.py

import os
import shutil
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np


# Initialize device, FaceNet, MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, device=device)


def extract_aligned_face(image_path):
    """
    Extract and align a face from an image file using MTCNN.
    Returns a torch.Tensor or None if extraction fails.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')

            # Optionally resize if extremely large
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024))

            face = mtcnn(img)
            if face is not None:
                return face
            else:
                print(f"No face detected in {image_path}.")
                return None

    except (OSError, IOError) as e:
        print(f"File error with {image_path}: {e}")
        return None
    except MemoryError:
        print(f"MemoryError: Unable to process {image_path}. Skipping file.")
        return None
    except Exception as e:
        print(f"Unexpected error processing {image_path}: {e}")
        return None


def generate_embedding(face):
    """
    Generates a 1D embedding tensor from an aligned face using FaceNet.
    """
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = facenet(face)
    return embedding.squeeze(0)


def generate_actor_embeddings(actors_ready_folder):
    """
    Generates face embeddings for each actor in `actors_ready_folder`.
    We assume there's at least one image in each actor's folder.
    Returns a dictionary: {actor_name: embedding_tensor}
    """
    actor_embeddings = {}
    for actor_name in os.listdir(actors_ready_folder):
        actor_dir = os.path.join(actors_ready_folder, actor_name)
        if not os.path.isdir(actor_dir):
            continue

        # Take the first image that ends with .png, .jpg, etc.
        images = [f for f in os.listdir(actor_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            continue

        first_image = images[0]
        image_path = os.path.join(actor_dir, first_image)
        face = extract_aligned_face(image_path)
        if face is not None:
            embedding = generate_embedding(face)
            actor_embeddings[actor_name] = embedding

    return actor_embeddings


def find_best_match(face_embedding, actor_embeddings, threshold=0.9):
    """
    Finds the best-matching actor. Returns (best_actor, best_similarity) or (None, best_similarity) if below threshold.
    """
    best_similarity = -1.0
    best_actor = None

    for actor_name, embedding in actor_embeddings.items():
        sim = cosine_similarity(
            face_embedding.unsqueeze(0),
            embedding.unsqueeze(0),
            dim=1
        ).item()

        if sim > best_similarity:
            best_similarity = sim
            best_actor = actor_name

    if best_similarity >= threshold:
        return best_actor, best_similarity
    else:
        return None, best_similarity

def match_and_move_faces(faces_folder, actors_ready_folder, actor_embeddings, threshold=0.9):
    """
    Matches each face in `faces_folder` to the best actor (above `threshold`) and moves it to that actor's folder.
    If no match is found, moves the file to an "unknown" folder in the same directory as faces_folder.
    """

    # Create the "unknown" folder in the same directory as faces_folder
    unknown_dir = os.path.join(faces_folder, 'unknown')
    os.makedirs(unknown_dir, exist_ok=True)

    for face_image in os.listdir(faces_folder):
        face_image_path = os.path.join(faces_folder, face_image)
        
        # Skip if it's not a file (could be subfolder, etc.)
        if not os.path.isfile(face_image_path):
            continue

        # Extract face from the image
        face = extract_aligned_face(face_image_path)
        if face is None:
            # If no face was extracted, skip
            continue

        # Generate face embedding
        face_embedding = generate_embedding(face)

        # Find best match from actor embeddings
        best_actor, best_similarity = find_best_match(face_embedding, actor_embeddings, threshold)
        
        if best_actor is not None:
            # We have a match above the threshold; move the file to best_actor's folder
            actor_dir = os.path.join(actors_ready_folder, best_actor)
            destination_path = os.path.join(actor_dir, face_image)
            shutil.move(face_image_path, destination_path)
            print(f"Moved {face_image} to {best_actor} (similarity: {best_similarity:.4f})")
        else:
            # No match found above threshold, move image to "unknown" folder
            destination_path = os.path.join(unknown_dir, face_image)
            shutil.move(face_image_path, destination_path)
            print(f"No match found for {face_image} (best similarity: {best_similarity:.4f}). Moved to 'unknown'.")
