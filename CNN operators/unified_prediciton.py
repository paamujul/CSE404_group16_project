import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

base_path = os.path.dirname(os.path.abspath(__file__))
expr_path = os.path.join(base_path, "expression_dataset")
csv_path = os.path.join(expr_path, "expression_labels.csv")
model_path = os.path.join(base_path, "unified_classifier.h5")

model = load_model(model_path)

label_map = {i: str(i) for i in range(10)}
label_map.update({10: '+', 11: '-', 12: '*', 13: '/'})

def merge_boxes(boxes, threshold=12):
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        box_i = [x1, y1, x1 + w1, y1 + h1]
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            x2, y2, w2, h2 = boxes[j]
            box_j = [x2, y2, x2 + w2, y2 + h2]
            if abs(x1 - x2) < threshold or abs((x2 + w2) - (x1 + w1)) < threshold:
                if abs(y1 - y2) < 14:
                    new_x = min(box_i[0], box_j[0])
                    new_y = min(box_i[1], box_j[1])
                    new_x2 = max(box_i[2], box_j[2])
                    new_y2 = max(box_i[3], box_j[3])
                    box_i = [new_x, new_y, new_x2, new_y2]
                    used[j] = True
        used[i] = True
        merged.append((box_i[0], box_i[1], box_i[2] - box_i[0], box_i[3] - box_i[1]))
    return merged

def segment_expression(image_path, margin=2):
    img = cv2.imread(image_path)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: b[0])
    merged_boxes = merge_boxes(boxes, threshold=15)
    segments = []
    for x, y, w, h in merged_boxes:
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray.shape[1], x + w + margin)
        y2 = min(gray.shape[0], y + h + margin)
        char_img = gray[y1:y2, x1:x2]
        segments.append(char_img)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_copy, segments

def preprocess_segment(seg):
    resized = cv2.resize(seg, (28, 28)).astype('float32') / 255.0
    return np.expand_dims(resized, axis=(0, -1))

df = pd.read_csv(csv_path)
correct = 0
total = 0

for idx, row in df.iterrows():
    fname = row['image']
    true_expr = row['true_expr']
    img_path = os.path.join(expr_path, fname)

    try:
        boxed_img, segments = segment_expression(img_path)
        prediction = ''
        for seg in segments:
            x = preprocess_segment(seg)
            pred = model.predict(x, verbose=0)
            label = np.argmax(pred)
            prediction += label_map[label]

        total += 1
        if prediction == true_expr:
            correct += 1
        else:
            print(f"{fname}: predicted = {prediction}, true = {true_expr}")

    except Exception as e:
        print(f"Error in {fname}: {e}")

print(f"\nFinal Accuracy: {correct}/{total} = {100 * correct / total:.2f}%")
