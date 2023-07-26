from django.urls import path
from . import views

urlpatterns = [
    path('classify/', views.classify_animal, name='classify_animal'),
    path('', views.home, name='home'),
]
