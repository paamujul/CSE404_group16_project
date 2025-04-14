import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.utils import shuffle

def load_mnist_from_csv(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.iloc[:, 1:].values / 255.0
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values / 255.0
    y_test = test_data.iloc[:, 0].values
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_train = 1.0 - X_train
    X_test = 1.0 - X_test
    noise_factor = 0.2
    X_train = np.clip(X_train + noise_factor * np.random.normal(0.0, 1.0, X_train.shape), 0.0, 1.0)
    X_test = np.clip(X_test + noise_factor * np.random.normal(0.0, 1.0, X_test.shape), 0.0, 1.0)
    return X_train, y_train, X_test, y_test

def select_balanced_digits(X, y, per_class=58):
    buckets = defaultdict(list)
    for i in range(len(y)):
        label = y[i]
        if len(buckets[label]) < per_class:
            buckets[label].append(X[i])
        if all(len(v) == per_class for v in buckets.values()):
            break
    X_selected = []
    y_selected = []
    for d in range(10):
        X_selected.extend(buckets[d])
        y_selected.extend([0] * per_class)
    return np.array(X_selected), np.array(y_selected)

base_path = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_path, 'dataset_train')
test_dir = os.path.join(base_path, 'dataset_test')

binary_datagen = ImageDataGenerator(rescale=1./255)

op_gen = binary_datagen.flow_from_directory(
    train_dir, target_size=(28, 28), color_mode='grayscale',
    batch_size=32, class_mode='categorical', shuffle=True
)

op_test_gen = binary_datagen.flow_from_directory(
    test_dir, target_size=(28, 28), color_mode='grayscale',
    batch_size=32, class_mode='categorical', shuffle=False
)

X_op, y_op = [], []
X_op_test, y_op_test = [], []
for i in range(len(op_gen)):
    xb, yb = op_gen[i]
    X_op.append(xb)
    y_op.append(np.ones(len(yb)))
X_op = np.concatenate(X_op)[:580]
y_op = np.concatenate(y_op)[:580]

for i in range(len(op_test_gen)):
    xb, yb = op_test_gen[i]
    X_op_test.append(xb)
    y_op_test.append(np.ones(len(yb)))
X_op_test = np.concatenate(X_op_test)[:100]
y_op_test = np.concatenate(y_op_test)[:100]

X_digit_train, y_digit_train, X_digit_test, y_digit_test = load_mnist_from_csv(
    os.path.join(base_path, 'mnist_train.csv'),
    os.path.join(base_path, 'mnist_test.csv')
)

X_digit_selected, y_digit_selected = select_balanced_digits(X_digit_train, y_digit_train, per_class=58)
X_digit_test_selected, y_digit_test_selected = select_balanced_digits(X_digit_test, y_digit_test, per_class=10)

X_bin = np.concatenate([X_op, X_digit_selected])
y_bin = np.concatenate([y_op, y_digit_selected])
X_bin_test = np.concatenate([X_op_test, X_digit_test_selected])
y_bin_test = np.concatenate([y_op_test, np.zeros(len(X_digit_test_selected))])

X_bin, y_bin = shuffle(X_bin, y_bin, random_state=42)
X_bin_test, y_bin_test = shuffle(X_bin_test, y_bin_test, random_state=42)

model_bin = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model_bin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_bin.fit(X_bin, y_bin, epochs=15, batch_size=32, validation_split=0.1)
model_bin.save("binary_digit_operator_model.h5")

loss_bin, acc_bin = model_bin.evaluate(X_bin_test, y_bin_test)
print(f"Binary Model Accuracy: {acc_bin * 100:.2f}%")
y_pred_bin = model_bin.predict(X_bin_test).ravel()
y_pred_bin_class = (y_pred_bin > 0.5).astype(int)

cm_bin = confusion_matrix(y_bin_test, y_pred_bin_class)
sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues", xticklabels=["Digit", "Operator"], yticklabels=["Digit", "Operator"])
plt.title("Binary Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

model_digit = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model_digit.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_digit.fit(X_digit_train, y_digit_train, epochs=10, batch_size=32, validation_split=0.1)
model_digit.save("digit_model.h5")

loss_digit, acc_digit = model_digit.evaluate(X_digit_test, y_digit_test)
print(f"Digit Model Accuracy: {acc_digit * 100:.2f}%")
y_pred_digit = np.argmax(model_digit.predict(X_digit_test), axis=1)
cm_digit = confusion_matrix(y_digit_test, y_pred_digit)
sns.heatmap(cm_digit, annot=True, fmt="d", cmap="Greens", xticklabels=list(range(10)), yticklabels=list(range(10)))
plt.title("Digit Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

op_train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
op_test_aug = ImageDataGenerator(rescale=1./255)

op_train_64 = op_train_aug.flow_from_directory(
    train_dir, target_size=(64, 64), color_mode='rgb',
    batch_size=32, class_mode='categorical', shuffle=True
)
op_test_64 = op_test_aug.flow_from_directory(
    test_dir, target_size=(64, 64), color_mode='rgb',
    batch_size=32, class_mode='categorical', shuffle=False
)

class_labels_op = list(op_train_64.class_indices.keys())
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(op_train_64.classes),
    y=op_train_64.classes
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

model_op = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels_op), activation='softmax')
])

model_op.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_op.fit(
    op_train_64,
    epochs=10,
    validation_data=op_test_64,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    workers=1,
    use_multiprocessing=False
)
model_op.save('operator_classifier_optimized.h5')

loss_op, acc_op = model_op.evaluate(op_test_64)
print(f"Operator Model Accuracy: {acc_op * 100:.2f}%")
y_pred_op = model_op.predict(op_test_64)
y_pred_op_classes = np.argmax(y_pred_op, axis=1)
y_true_op = op_test_64.classes
cm_op = confusion_matrix(y_true_op, y_pred_op_classes)
sns.heatmap(cm_op, annot=True, fmt="d", cmap="Purples", xticklabels=class_labels_op, yticklabels=class_labels_op)
plt.title("Operator Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
