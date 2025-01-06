from django import forms
from .models import Image

MODEL_CHOICES = [
    ("yolo11x", "yolo11x"),
    ("yolo11s", "yolo11s"),
    ("yolo11m", "yolo11m"),
    ("yolo11l", "yolo11l"),
    ("yolov8x", "yolov8x"),
    ("yolov5s", "yolov5s"),
]
class ImageForm(forms.ModelForm):
    
    model_choice = forms.ChoiceField(choices=MODEL_CHOICES, label='Chọn Model')
    image = forms.ImageField(label='Ảnh')
    class Meta:
        model = Image
        fields = ['image', 'model_choice', 'detection_type', 'specific_objects']
        
    DETECTION_CHOICES = [
        ("all", "Detect All Objects"),
        ("specific", "Segmnent Specific Objects"),
    ]
    
    detection_type = forms.ChoiceField(
        choices=DETECTION_CHOICES, 
        label='Detection Type', 
        widget=forms.RadioSelect
    )
    specific_objects = forms.CharField(
        label='Specific Objects',
        max_length=400,
        required=False, 
        widget=forms.TextInput(attrs={'placeholder': 'Enter labels separated by commas'})
    )
        
    
