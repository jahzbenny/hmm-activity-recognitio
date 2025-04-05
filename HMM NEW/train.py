import cv2
import numpy as np
from sklearn.cluster import KMeans
from hmmlearn import hmm
import argparse
import os

# Step 1: Load video frames
def load_video_frames(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        count += 1
    cap.release()
    print(f"Loaded {len(frames)} frames.")
    return frames

# Step 2: Extract basic motion features (frame difference)
def extract_motion_features(frames):
    features = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        features.append(np.mean(diff))  # simple motion intensity
    return np.array(features).reshape(-1, 1)

# Step 3: Cluster features into discrete observations
def cluster_features(features, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    obs_seq = kmeans.fit_predict(features)
    return obs_seq, kmeans

# Step 4: Train an HMM
def train_hmm(obs_seq, n_states=4):
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=42)
    model.fit(obs_seq.reshape(-1, 1))
    return model

# Step 5: Predict state sequence
def predict_states(model, obs_seq):
    log_prob, states = model.decode(obs_seq.reshape(-1, 1), algorithm="viterbi")
    return states

# Save model (optional)
def save_model(model, filename="hmm_model.npz"):
    np.savez(filename, startprob=model.startprob_, transmat=model.transmat_)

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HMM on video for activity recognition.")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--clusters", type=int, default=4, help="Number of feature clusters (observation symbols)")
    parser.add_argument("--states", type=int, default=4, help="Number of hidden states in HMM")
    parser.add_argument("--output", type=str, default="hmm_model.npz", help="Where to save the HMM model")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    # Run the training pipeline
    frames = load_video_frames(args.video_path)
    features = extract_motion_features(frames)
    obs_seq, kmeans = cluster_features(features, n_clusters=args.clusters)
    model = train_hmm(obs_seq, n_states=args.states)
    predicted_states = predict_states(model, obs_seq)

    # Output results
    print("Predicted hidden states:", predicted_states)
    print("HMM transition matrix:\n", model.transmat_)
    save_model(model, args.output)
    print(f"Model saved to {args.output}")
