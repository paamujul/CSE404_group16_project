import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd

csv_path = 'expression_dataset/expression_labels.csv'
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
    h_img, w_img = gray.shape
    for x, y, w, h in merged_boxes:
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_img, x + w + margin)
        y2 = min(h_img, y + h + margin)
        char_img = gray[y1:y2, x1:x2]
        segments.append(char_img)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_copy, segments

base_path = os.path.dirname(os.path.abspath(__file__))
expr_path = os.path.join(base_path, "expression_dataset")

for i in range(2):  # only for visualization purposes
    fname = f"expr_{i:03d}.png"
    img_path = os.path.join(expr_path, fname)
    boxed_img, segments = segment_expression(img_path)

    print(f"{fname}: {len(segments)} segments")

    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{fname} with bounding boxes")
    plt.axis('off')
    plt.show()

    for idx, seg in enumerate(segments):
        plt.figure()
        plt.imshow(seg, cmap='gray')
        plt.title(f"{fname} - Segment {idx}")
        plt.axis('off')
        plt.show()

model_bin = load_model(os.path.join(base_path, 'binary_digit_operator_model.h5'))
model_digit = load_model(os.path.join(base_path, 'digit_model.h5'))
model_op = load_model(os.path.join(base_path, 'operator_classifier_optimized.h5'))

def preprocess_gray(img):
    resized = cv2.resize(img, (28, 28)).astype('float32') / 255.0
    return np.expand_dims(resized, axis=(0, -1))

def preprocess_rgb(img):
    resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (64, 64)).astype('float32') / 255.0
    return np.expand_dims(resized, axis=0)

class_labels_op = {
    'add': '+',
    'sub': '-',
    'mul': '*',
    'div': '/',
    'eq': '='
}

for i in range(2):  
    fname = f"expr_{i:03d}.png"
    img_path = os.path.join(expr_path, fname)
    boxed_img, segments = segment_expression(img_path)
    result = ''
    for i in segments:
        x_bin = preprocess_gray(i)
        is_op = model_bin.predict(x_bin)[0][0]
        print(is_op)
        if is_op >= 0.5:
            x_bin = preprocess_rgb(i)
            op = model_op.predict(x_bin)[0]
            label_raw = list(class_labels_op.keys())[np.argmax(op)]
            result += class_labels_op[label_raw]
        else:
            digit = model_digit.predict(x_bin)[0]
            result += str(np.argmax(digit))
    print(f"{fname}: {result}")


correct = 0
total = 0

df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    fname = row['image']
    true_expr = row['true_expr']
    img_path = os.path.join(expr_path, fname)

    try:
        boxed_img, segments = segment_expression(img_path)
        predicted = ''
        for seg in segments:
            x_bin = preprocess_gray(seg)
            is_op = model_bin.predict(x_bin, verbose=0)[0][0]
            if is_op >= 0.5:
                x_rgb = preprocess_rgb(seg)
                op = model_op.predict(x_rgb, verbose=0)[0]
                label_raw = list(class_labels_op.keys())[np.argmax(op)]
                predicted += class_labels_op[label_raw]
            else:
                digit = model_digit.predict(x_bin, verbose=0)[0]
                predicted += str(np.argmax(digit))

        total += 1
        if predicted == true_expr:
            correct += 1
        else:
            print(f"Wrong: {fname} | Predicted: {predicted} | True: {true_expr}")

    except Exception as e:
        print(f"Error processing {fname}: {e}")

print(f"\nTotal: {total} | Correct: {correct} | Accuracy: {100 * correct / total:.2f}%")