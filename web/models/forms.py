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
    model_choice = forms.ChoiceField(choices=MODEL_CHOICES, label='Select Model')
    class Meta:
        model = Image
        fields = ['image']