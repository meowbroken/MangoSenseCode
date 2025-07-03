from django.urls import path
from . import views

app_name = 'mangosense'

urlpatterns = [
    # Authentication API endpoints
    path('register/', views.register_api, name='register_api'),
    path('login/', views.login_api, name='login_api'),
    path('logout/', views.logout_api, name='logout_api'),
    path('predict/', views.predict_image, name='predict_image'),
    path('test-model/', views.test_model_status, name='test_model_status'),  # Debug endpoint
]