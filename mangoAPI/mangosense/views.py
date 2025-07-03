from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from django.conf import settings
from django.utils import timezone
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import re
import gc

# ML Configuration
IMG_SIZE = (224, 224)
class_names = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge',
    'Healthy', 'Powdery Mildew', 'Sooty Mold', 'Black Mold Rot', 'Stem End Rot'
]

# Load ML model
model_path = str(settings.MODEL_PATH)
model = tf.keras.models.load_model(model_path)

# Treatment suggestions
treatment_suggestions = {
    'Anthracnose': 'The diseased twigs should be pruned and burnt along with fallen leaves. Spraying twice with Carbendazim (Bavistin 0.1%) at 15 days interval during flowering controls blossom infection.',
    'Bacterial Canker': 'Three sprays of Streptocycline (0.01%) or Agrimycin-100 (0.01%) after first visual symptom at 10 day intervals are effective in controlling the disease.',
    'Cutting Weevil': 'Use recommended insecticides and remove infested plant material.',
    'Die Back': 'Pruning of the diseased twigs 2-3 inches below the affected portion and spraying Copper Oxychloride (0.3%) on infected trees controls the disease.',
    'Gall Midge': 'Remove and destroy infested fruits; use appropriate insecticides.',
    'Healthy': 'No treatment needed. Maintain good agricultural practices.',
    'Powdery Mildew': 'Alternate spraying of Wettable sulphur 0.2 per cent at 15 days interval are recommended for effective control of the disease.',
    'Sooty Mold': 'Pruning of affected branches and their prompt destruction followed by spraying of Wettasulf (0.2%) helps to control the disease.'
}


# Validation Functions
def validate_password_strength(password):
    """Validate password strength - minimum 8 characters"""
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    return errors

def validate_name(name, field_name):
    """Validate first name and last name"""
    errors = []
    if not name or len(name.strip()) < 2:
        errors.append(f"{field_name} must be at least 2 characters long.")
    return errors

def validate_address(address):
    """Validate address field"""
    errors = []
    if not address or len(address.strip()) < 5:
        errors.append("Address must be at least 5 characters long.")
    if len(address) > 200:
        errors.append("Address cannot exceed 200 characters.")
    return errors

def validate_mobile_image(image_file):
    """Validate image uploaded from mobile app"""
    # Check file size (max 5MB for mobile)
    if image_file.size > 5 * 1024 * 1024:
        return False, "Image size must be less than 5MB"
    
    # Check file type - common mobile formats
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
    if image_file.content_type not in allowed_types:
        return False, "Only JPEG and PNG images are allowed"
    
    try:
        # Verify it's a valid image
        img = Image.open(image_file)
        img.verify()
        return True, "Valid image"
    except Exception:
        return False, "Invalid image file"

def preprocess_image(image_file):
    """Preprocess image for ML model prediction"""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img.close()
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

# Authentication Views
def register_view(request):
    if request.method == 'GET':
        return render(request, 'mangosense/register.html')

@csrf_exempt
@require_http_methods(["POST"])
def register_api(request):
    try:
        data = json.loads(request.body)
        
        # Handle both camelCase (Ionic) and snake_case (Django) field names
        first_name = (data.get('first_name') or data.get('firstName', '')).strip()
        last_name = (data.get('last_name') or data.get('lastName', '')).strip()
        address = data.get('address', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password') or data.get('confirmPassword') or password  # Handle missing confirm_password
        
        errors = []

        # Update required fields validation
        if not all([first_name, last_name, address, email, password]):
            return JsonResponse({
                'success': False,
                'error': 'All required fields must be provided.'
            }, status=400)
        
        # First name validation
        name_errors = validate_name(first_name, "First name")
        errors.extend(name_errors)

        # Last name validation
        name_errors = validate_name(last_name, "Last name")
        errors.extend(name_errors)

        # Address validation
        address_errors = validate_address(address)
        errors.extend(address_errors)

        # Email validation
        try:
            validate_email(email)
        except ValidationError:
            errors.append("Invalid email format.")

        # Check for existing email
        if User.objects.filter(email=email).exists():
            errors.append("An account with this email already exists.")

        # Only check password confirmation if confirm_password is provided
        if confirm_password and password != confirm_password:
            errors.append("Passwords do not match.")
        
        password_errors = validate_password_strength(password)
        errors.extend(password_errors)

        if errors:
            return JsonResponse({
                'success': False,
                'errors': errors
            }, status=400)
        
        # Create user (Note: Django User model doesn't have address field by default)
        user = User.objects.create_user(
            username=email,
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password
        )

        # Store address in user profile (you'll need to create a profile model for this)
        # For now, we'll return success but note that address isn't stored
        return JsonResponse({
            'success': True,
            'message': 'Account created successfully! You may now log in',
            'user_id': user.id,
            'note': 'Address received but not stored (requires profile model)'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid data format.'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred: {}'.format(str(e))
        }, status=500)
    
@csrf_exempt
@require_http_methods(["POST"])
def login_api(request):
    try:
        data = json.loads(request.body)
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return JsonResponse({
                'success': False,
                'error': 'Email and password are required.'
            }, status=400)
        
        try:
            validate_email(email)
        except ValidationError:
            return JsonResponse({
                'success': False,
                'error': 'Please enter a valid email address.'
            }, status=400)
        
        user = authenticate(request, username=email, password=password)
        if user is not None:
            if user.is_active:
                login(request, user)
                return JsonResponse({
                    'success': True,
                    'message': 'Login successful.',
                    'user': {
                        'id': user.id,
                        'email': user.email,
                        'firstName': user.first_name,  # Fix: use first_name (Django field)
                        'lastName': user.last_name,    # Fix: use last_name (Django field)
                        'full_name': f"{user.first_name} {user.last_name}"  # Fix: use Django fields
                    }
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Your account is deactivated. Please contact support.'
                }, status=401)
        else:
            return JsonResponse({
                'success': False,
                'error': 'Invalid email or password.'
            }, status=401)
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid data format.'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred: {}'.format(str(e))
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def logout_api(request):
    if request.user.is_authenticated:
        logout(request)
        return JsonResponse({
            'success': True,
            'message': 'Logout successful.'
        })
    else:
        return JsonResponse({
            'success': False,
            'error': 'You are not logged in.'
        }, status=401)

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

# ML Detection Views
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def predict_image(request):
    """Handle image prediction from mobile Ionic app"""
    if 'image' not in request.FILES:
        return JsonResponse({
            'success': False,
            'error': 'No image uploaded'
        }, status=400)
    
    try:
        image_file = request.FILES['image']
        
        # Validate mobile image
        is_valid, validation_message = validate_mobile_image(image_file)
        if not is_valid:
            return JsonResponse({
                'success': False,
                'error': validation_message
            }, status=400)
        
        # Check if model is loaded
        if model is None:
            return JsonResponse({
                'success': False,
                'error': 'ML model not loaded properly'
            }, status=500)
        
        # Process image for prediction
        img_array = preprocess_image(image_file)
        prediction = model.predict(img_array)
        
        # Debug: Print prediction values
        print(f"Raw prediction: {prediction[0]}")
        print(f"Max prediction: {np.max(prediction[0])}")
        print(f"Prediction shape: {prediction.shape}")
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        
        top_3_predictions = []
        for i, idx in enumerate(top_3_indices):
            confidence = float(prediction[0][idx]) * 100
            disease_name = class_names[idx] if idx < len(class_names) else f"Unknown_{idx}"
            
            top_3_predictions.append({
                'rank': i + 1,
                'disease': disease_name,
                'confidence': round(confidence, 2),
                'confidence_formatted': f"{confidence:.2f}%",
                'treatment': treatment_suggestions.get(disease_name, "No treatment information available.")
            })
        
        # Primary prediction (highest confidence)
        primary_prediction = top_3_predictions[0]
        predicted_class = primary_prediction['disease']
        
        # Debug: Print final results
        print(f"Predicted class: {predicted_class}")
        print(f"Top 3: {[p['disease'] for p in top_3_predictions]}")
        
        # Clear memory
        gc.collect()
        
        # Return mobile-optimized response with top 3 predictions
        response_data = {
            'success': True,
            'primary_prediction': {
                'disease': primary_prediction['disease'],
                'confidence': primary_prediction['confidence_formatted'],
                'confidence_score': primary_prediction['confidence'],
                'treatment': primary_prediction['treatment']
            },
            'top_3_predictions': top_3_predictions,
            'prediction_summary': {
                'most_likely': primary_prediction['disease'],
                'confidence_level': 'High' if primary_prediction['confidence'] > 80 else 'Medium' if primary_prediction['confidence'] > 60 else 'Low',
                'total_diseases_checked': len(class_names)
            },
            'timestamp': timezone.now().isoformat(),
            'debug_info': {
                'model_loaded': model is not None,
                'class_names_count': len(class_names),
                'image_size': img_array.shape
            }
        }
        
        print(f"Response data: {response_data}")
        return JsonResponse(response_data)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }, status=500)

@api_view(['GET'])
def test_model_status(request):
    """Test endpoint to check if model and class names are loaded properly"""
    try:
        return JsonResponse({
            'success': True,
            'model_status': {
                'model_loaded': model is not None,
                'model_path': str(settings.MODEL_PATH) if hasattr(settings, 'MODEL_PATH') else 'Not set',
                'class_names': class_names,
                'class_names_count': len(class_names),
                'treatment_suggestions_count': len(treatment_suggestions)
            },
            'available_diseases': class_names,
            'img_size': IMG_SIZE
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

