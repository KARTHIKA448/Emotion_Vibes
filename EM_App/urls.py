# EM_App/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='home'),
    path('predict/', views.predict_emotion, name='predict_emotion'),
    
]      