import os
import random
import shutil


src_dir = './'
train_dir = './dataset_train'
test_dir = './dataset_test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = ['add', 'sub', 'mul', 'eq', 'div']
for class_name in classes:

    class_path = os.path.join(src_dir, class_name)
    image_files = os.listdir(class_path)

    random.shuffle(image_files)
    
    train_files = image_files[:500]
    test_files = image_files[500:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(train_dir, class_name, file))

    for file in test_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(test_dir, class_name, file))
