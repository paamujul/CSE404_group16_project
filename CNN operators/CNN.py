import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy

base_path = os.path.dirname(os.path.abspath(__file__))
mnist_train_path = os.path.join(base_path, 'mnist_train.csv')
mnist_test_path = os.path.join(base_path, 'mnist_test.csv')
train_dir = os.path.join(base_path, 'dataset_train')
test_dir = os.path.join(base_path, 'dataset_test')

def load_mnist_from_csv(csv_path, per_class):
    data = pd.read_csv(csv_path)
    X = data.iloc[:, 1:].values / 255.0
    y = data.iloc[:, 0].values
    X = X.reshape(-1, 28, 28, 1)
    X = 1.0 - X
    buckets = defaultdict(list)
    for i in range(len(y)):
        label = y[i]
        if len(buckets[label]) < per_class:
            buckets[label].append(X[i])
        if all(len(buckets[d]) == per_class for d in range(10)):
            break
    X_out, y_out = [], []
    for d in range(10):
        X_out.extend(buckets[d])
        y_out.extend([d] * per_class)
    return np.array(X_out), np.array(y_out)

def load_operator_images(folder_path, per_class_limit=None):
    class_map = {'add': 10, 'sub': 11, 'mul': 12, 'div': 13}
    X, y = [], []
    for op_label, class_idx in class_map.items():
        class_folder = os.path.join(folder_path, op_label)
        count = 0
        for fname in os.listdir(class_folder):
            if per_class_limit and count >= per_class_limit:
                break
            img_path = os.path.join(class_folder, fname)
            img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
            arr = img_to_array(img) / 255.0
            X.append(arr)
            y.append(class_idx)
            count += 1
    return np.array(X), np.array(y)

X_digit_train, y_digit_train = load_mnist_from_csv(mnist_train_path, per_class=500)
X_digit_test, y_digit_test = load_mnist_from_csv(mnist_test_path, per_class=100)

X_op_train, y_op_train = load_operator_images(train_dir, per_class_limit=500)
X_op_test, y_op_test = load_operator_images(test_dir)

X_all_train = np.concatenate([X_digit_train, X_op_train])
y_all_train = np.concatenate([y_digit_train, y_op_train])
X_all_test = np.concatenate([X_digit_test, X_op_test])
y_all_test = np.concatenate([y_digit_test, y_op_test])

X_all_train, y_all_train = shuffle(X_all_train, y_all_train, random_state=42)
X_all_test, y_all_test = shuffle(X_all_test, y_all_test, random_state=42)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_all_train, y_all_train, test_size=0.1, random_state=42
)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    shear_range=0.15
)
datagen.fit(X_train_split)

model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(), LeakyReLU(),

    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(), LeakyReLU(),
    MaxPooling2D(),

    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(), LeakyReLU(),

    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(), LeakyReLU(),
    MaxPooling2D(),

    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(), LeakyReLU(),
    MaxPooling2D(),

    GlobalAveragePooling2D(),
    Dense(256), LeakyReLU(),
    Dropout(0.5),
    Dense(14, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    datagen.flow(X_train_split, y_train_split, batch_size=32),
    validation_data=(X_val_split, y_val_split),
    epochs=35,
    callbacks=[early_stop]
)

model.save("unified_classifier.h5")

loss, acc = model.evaluate(X_all_test, y_all_test)
print(f"Unified Model Accuracy: {acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_all_test), axis=1)
class_labels = [str(i) for i in range(10)] + ['add', 'sub', 'mul', 'div']
print(classification_report(y_all_test, y_pred, target_names=class_labels))

cm = confusion_matrix(y_all_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=class_labels, yticklabels=class_labels)
plt.title("14-Class Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
