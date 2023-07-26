import os
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from .models import load_and_preprocess_image
import tensorflow as tf

def classify_animal(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(image_path, 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)

        img_array = load_and_preprocess_image(image_path)

        model_save_path = os.path.join(settings.BASE_DIR, 'trained_model.h5')
        labels_path = os.path.join(settings.BASE_DIR, 'classifier_app', 'dataset', 'name_of_the_animals.txt')

        model = tf.keras.models.load_model(model_save_path)
        with open(labels_path) as f:
            labels = f.readlines()

        predictions = model.predict(img_array)
        predicted_class = labels[predictions.argmax()]

        top_classes = get_top_classes(predictions, labels)

        context = {'predicted_class': predicted_class, 'top_classes': top_classes}
        return render(request, 'classifier_app/result.html', context)

    return render(request, 'classifier_app/classify.html')


def home(request):
    return render(request, 'classifier_app/home.html')


def get_top_classes(predictions, labels, top_n=5):
    top_classes = []
    top_indices = tf.math.top_k(predictions, k=top_n).indices.numpy()[0]
    for index in top_indices:
        class_probability = predictions[0, index]
        class_name = labels[index]
        top_classes.append({'label': class_name, 'probability': class_probability * 100})
    return top_classes
