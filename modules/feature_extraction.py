# feature_extraction.py

import os
import shutil
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import time
from modules.model_training import create_model
from tqdm import tqdm
from modules.representation_layer_extraction import RepresentationLayerExtraction

class FeatureExtraction:
    def __init__(self, face_recognition_arch, embedding_layer_name, face_recognition_model_checkpoint=None, pretrained=False, data_parallel_patch=False, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # if face_recognition_model_checkpoint:
        # print(f"Loading model checkpoint from {face_recognition_model_checkpoint}")
        # else:
        #     # Load the pre-trained FaceNet model
        #     self.face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        #     print("Using pre-trained ResNet FaceNet model (InceptionResnetV1)")

        self.mtcnn = MTCNN(image_size=160, margin=20, device=self.device)

        face_recognition_model = create_model(device=self.device, model_arch=face_recognition_arch, model_checkpoint_path=face_recognition_model_checkpoint,
         pretrained=pretrained, train_mode=False, data_parallel_patch=data_parallel_patch).eval().to(self.device)
        self.representationLayerExtractor = RepresentationLayerExtraction(face_recognition_model, target_layer_name=embedding_layer_name)

    def extract_aligned_face(self, image_path):
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

                face = self.mtcnn(img)
                if face is not None:
                    return face
                else:
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


    def generate_embedding(self, face):
        """
        Generates a 1D embedding tensor from an aligned face using model_checkpoint_path passed.
        If None, uses FaceNet pretrained model (on VGGFace2 dataset).
        """

        face = face.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.representationLayerExtractor.extract_representation(face)
        return embedding.squeeze(0)


    def generate_actor_embeddings(self, actors_ready_folder):
        """
        Generates face embeddings for each actor in `actors_ready_folder`.
        Returns a dictionary: {actor_name: mean_embedding_tensor}
        The embedding for each actor is the mean of embeddings from all their images.
        """
        actor_embeddings = {}
        
        for actor_name in os.listdir(actors_ready_folder):
            actor_dir = os.path.join(actors_ready_folder, actor_name)
            if not os.path.isdir(actor_dir):
                continue

            # Get all images for this actor
            images = [f for f in os.listdir(actor_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                print(f"No images found for actor: {actor_name}")
                continue

            print(f"Processing {len(images)} images for {actor_name}")
            embeddings_list = []

            # Generate embedding for each image
            for image_name in images:
                image_path = os.path.join(actor_dir, image_name)
                face = self.extract_aligned_face(image_path)
                
                if face is not None:
                    embedding = self.generate_embedding(face)
                    embeddings_list.append(embedding)

            if embeddings_list:
                # Calculate mean embedding
                mean_embedding = torch.stack(embeddings_list).mean(dim=0)
                actor_embeddings[actor_name] = mean_embedding
                print(f"Generated mean embedding for {actor_name} from {len(embeddings_list)} images")
            else:
                print(f"No valid embeddings generated for {actor_name}")

        return actor_embeddings


    def find_best_match(self, face_embedding, actor_embeddings, threshold=0.9):
        """
        Finds the best-matching actor. Always returns (best_actor, best_similarity).
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

        # Always return the best actor, even if similarity is below threshold
        return best_actor, best_similarity

    def match_and_move_faces(self, faces_folder, actors_ready_folder, actor_embeddings, threshold=0.9):
        """
        Recursively matches each face in `faces_folder` (including subfolders) to the best actor.
        Only moves images that meet the similarity threshold requirement.
        Images below threshold are left in their original location.
        """
        for root, dirs, files in os.walk(faces_folder):
            dirs[:] = [d for d in dirs if d.lower() != 'unknown']
            
            for file in tqdm(files, desc=f"Processing {root}", unit="face frame .PNG"):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    tqdm.write(f"Skipping non-image file: {file}")
                    continue
                
                face_image_path = os.path.join(root, file)
                
                face = self.extract_aligned_face(face_image_path)
                if face is None:
                    tqdm.write(f"No face extracted from: {face_image_path}. Skipping.")
                    continue
                
                face_embedding = self.generate_embedding(face)
                if face_embedding is None:
                    tqdm.write(f"Failed to generate embedding for: {face_image_path}. Skipping.")
                    continue
                
                # Get best match and check against threshold
                best_actor, best_similarity = self.find_best_match(face_embedding, actor_embeddings)
                
                # Only move files that meet the threshold requirement
                if best_similarity >= threshold:
                    actor_dir = os.path.join(actors_ready_folder, best_actor)
                    os.makedirs(actor_dir, exist_ok=True)
                    
                    timestamp = int(time.time())
                    new_filename = f"{os.path.splitext(file)[0]}_{timestamp}{os.path.splitext(file)[1]}"
                    destination_path = os.path.join(actor_dir, new_filename)
                    
                    shutil.move(face_image_path, destination_path)
                    tqdm.write(f"Moved {file} to {best_actor} (similarity: {best_similarity:.3f})")
                else:
                    tqdm.write(f"Skipping {file}: Best match {best_actor} below threshold (similarity: {best_similarity:.3f})")
        
        print("[INFO] Completed matching and moving faces.")

    def find_best_matches(self, embeddings_dict, query_embedding, top_k=1):
        """Find the best matching embeddings based on cosine similarity."""
        similarities = {}
        for name, stored_embedding in embeddings_dict.items():
            similarity = cosine_similarity(query_embedding, stored_embedding)
            similarities[name] = similarity
        
        # Sort by similarity score and get top k matches
        sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_matches

    def process_embeddings(self, embeddings_folder, query_embeddings):
        """Process embeddings and find best matches."""
        # Load all stored embeddings
        stored_embeddings = {}
        for embedding_file in os.listdir(embeddings_folder):
            if embedding_file.endswith('.npy'):
                name = os.path.splitext(embedding_file)[0]
                embedding_path = os.path.join(embeddings_folder, embedding_file)
                stored_embeddings[name] = np.load(embedding_path)
        
        # Find best match for each query embedding
        results = []
        for query_embedding in query_embeddings:
            matches = self.find_best_matches(stored_embeddings, query_embedding)
            if matches:
                best_match, similarity = matches[0]
                results.append((best_match, similarity))
        
        return results