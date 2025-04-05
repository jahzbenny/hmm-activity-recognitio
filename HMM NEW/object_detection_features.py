import numpy as np

# Load pre-trained MobileNet SSD (COCO-trained)
def load_person_detector():
    net = cv2.dnn.readNetFromCaffe(
        'models/MobileNetSSD_deploy.prototxt',
        'models/MobileNetSSD_deploy.caffemodel'
    )
    return net

# Extract person bounding box features (centroids and sizes)
def extract_object_features(frames, net, conf_threshold=0.4):
    features = []
    for frame in frames:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # For each frame, compute average person location and size
        positions = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            if confidence > conf_threshold and class_id == 15:  # person class
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                positions.append([center_x, center_y, area])

        if positions:
            avg_pos = np.mean(positions, axis=0)
        else:
            avg_pos = [0, 0, 0]  # No detection fallback

        features.append(avg_pos)
    return np.array(features)
 feature extractor here
