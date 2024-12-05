import os
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import ImageForm
from .models import Image
from ultralytics import YOLO
import cv2



def upload_image(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            uploaded_image = form.save()
            model_choice = form.cleaned_data["model_choice"]

            model = YOLO(model_choice + ".pt")

            # Run YOLO prediction
            original_path = os.path.join(settings.MEDIA_ROOT, str(uploaded_image.image))
            results = model(original_path)

            # Save the predicted image
            for result in results:
                result_image = result.plot()
                predicted_path = os.path.join(settings.MEDIA_ROOT, "predictions", f"predicted_{uploaded_image.id}.jpg")
                cv2.imwrite(predicted_path, result_image)
                uploaded_image.predicted_image = f"predictions/predicted_{uploaded_image.id}.jpg"
                uploaded_image.save()

            return redirect("image_detail", pk=uploaded_image.id)
    else:
        form = ImageForm()
    return render(request, "models/upload_image.html", {"form": form})


def image_detail(request, pk):
    image = Image.objects.get(pk=pk)
    return render(request, "models/image_detail.html", {"image": image})