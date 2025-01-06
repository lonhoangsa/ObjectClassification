from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload_image, name="upload_image"),
    # path("image/<int:pk>/detect_and_segment/", views.detect_object, name="detect_and_segment"),
    path("image/<int:pk>/", views.image_detail, name="image_detail"),
]