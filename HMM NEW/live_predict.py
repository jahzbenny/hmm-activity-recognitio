import cv2
import numpy as np
from sklearn.cluster import KMeans
from hmmlearn import hmm
from object_detection_features import load_person_detector, extract_object_features

# Load HMM model
def load_model(npz_path):
    data = np.load(npz_path)
    model = hmm.MultinomialHMM(n_components=len(data['startprob']), n_iter=100)
    model.startprob_ = data['startprob']
    model.transmat_ = data['transmat']
    return model

# Live webcam prediction
def live_predict(model, kmeans):
    net = load_person_detector()
    cap = cv2.VideoCapture(0)

    recent_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_small = cv2.resize(frame, (300, 300))
        feature = extract_object_features([frame_small], net)[0]
        recent_features.append(feature)

        if len(recent_features) > 10:  # Run prediction on last 10 frames
            X = np.array(recent_features[-10:])
            obs_seq = kmeans.predict(X)
            _, states = model.decode(obs_seq.reshape(-1, 1), algorithm="viterbi")
            current_state = states[-1]
            cv2.putText(frame, f"Predicted State: {current_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live Activity Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# TODO: Add live prediction code here
