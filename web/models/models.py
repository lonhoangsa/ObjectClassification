from django.db import models

# Create your models here.
class Image(models.Model):
    image = models.ImageField(upload_to='images/', blank=True)

    # uploaded_at = models.DateTimeField(auto_now_add=True)
    predicted_image = models.ImageField(upload_to='predicted_images/', blank=True, null=True)
    # description = models.TextField(blank=True, null=True)