# classifier_app/train_model.py
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Пути к данным и папке для сохранения обученной модели
data_dir = 'dataset/animals'
model_save_path = 'trained_model.h5'  # Путь для сохранения обученной модели
labels_path = 'dataset/name_of_the_animals.txt'  # Путь к файлу с именами классов

# Чтение имен классов из файла
with open(labels_path, 'r') as f:
    class_names = f.read().splitlines()

# Параметры модели
input_shape = (224, 224, 3)
batch_size = 32
num_classes = len(class_names)

# Создание генератора изображений
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Загрузка данных из папки и подготовка тренировочного и валидационного наборов
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    classes=class_names  # Указываем имена классов для генератора
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    classes=class_names  # Указываем имена классов для генератора
)

# Создание и обучение модели
base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Сохранение обученной модели
model.save(model_save_path)
