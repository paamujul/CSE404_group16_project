import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
mnist_train_path = os.path.join(base_path, 'mnist_train.csv')
op_train_path = os.path.join(base_path, 'dataset_train')
output_dir = os.path.join(base_path, 'expression_dataset')
os.makedirs(output_dir, exist_ok=True)

digit_data = pd.read_csv(mnist_train_path)
X_digits = digit_data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
X_digits = 255 - X_digits
y_digits = digit_data.iloc[:, 0].values

operator_images = []
operator_labels = []
for label in os.listdir(op_train_path):
    class_dir = os.path.join(op_train_path, label)
    for fname in os.listdir(class_dir):
        path = os.path.join(class_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (28, 28))
            operator_images.append(img)
            operator_labels.append(label)

operator_symbol_map = {
    'add': '+',
    'sub': '-',
    'mul': '*',
    'div': '/',
    'eq': '='
}

expression_images = []
expression_labels = []
true_expressions = []

def get_digit():
    idx = random.randint(0, len(X_digits) - 1)
    return X_digits[idx], str(y_digits[idx])

for i in range(100):
    left_img, left_str = get_digit()
    right_img, right_str = get_digit()

    op_idx = random.randint(0, len(operator_images) - 1)
    op_img = operator_images[op_idx]
    op_label = operator_labels[op_idx]
    op_symbol = operator_symbol_map.get(op_label, op_label)

    expr_img = np.hstack([left_img, op_img, right_img])
    expression_images.append(expr_img)

    readable = f"{left_str} {op_label} {right_str}"
    compact = f"{left_str}{op_symbol}{right_str}"
    expression_labels.append(readable)
    true_expressions.append(compact)

    cv2.imwrite(os.path.join(output_dir, f"expr_{i:03d}.png"), expr_img)

df = pd.DataFrame({
    'image': [f"expr_{i:03d}.png" for i in range(100)],
    'label': expression_labels,
    'true_expr': true_expressions
})
df.to_csv(os.path.join(output_dir, "expression_labels.csv"), index=False)

plt.figure(figsize=(20, 4))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(expression_images[i], cmap='gray')
    plt.title(expression_labels[i], fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()
