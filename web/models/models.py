from django.db import models

# Create your models here.
class Image(models.Model):
    image = models.ImageField(upload_to='images/', blank=True)
    predicted_image = models.ImageField(upload_to='predicted_images/', blank=True, null=True)
    cropped_images = models.JSONField(default=list, blank=True, null=True)
    segmented_images = models.JSONField(default=list, blank=True, null=True)
    masked_images = models.JSONField(default=list, blank=True, null=True)
    model_choice = models.CharField(max_length=50, blank=True, null=True)
    detection_type = models.CharField(max_length=50, blank=True, null=True)
    specific_objects = models.CharField(max_length=400, blank=True, null=True)
    # uploaded_at = models.DateTimeField(auto_now_add=True)
    # description = models.TextField(blank=True, null=True)

