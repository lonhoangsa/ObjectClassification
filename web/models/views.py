import os

import cv2
import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology
from django.conf import settings
from django.shortcuts import render, redirect
from ultralytics import YOLO

from autodistill.utils import plot
from autodistill_grounding_dino import GroundingDINO
from autodistill_grounded_sam_2 import GroundedSAM2
from .forms import ImageForm
from .models import Image


def upload_image(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            uploaded_image = form.save()
            
            model_choice = form.cleaned_data["model_choice"]
            detection_type = form.cleaned_data["detection_type"]
            detected_objects = form.cleaned_data["specific_objects"]
            label_dict = {}
            label_list = []
            segmented_images = []
            cropped_images = []
            original_path = os.path.join(settings.MEDIA_ROOT, str(uploaded_image.image))
            
            cropped_dir = os.path.join(settings.MEDIA_ROOT, "cropped", str(uploaded_image.id))
            os.makedirs(cropped_dir, exist_ok=True)
            print(cropped_dir)
            segmented_dir = os.path.join(settings.MEDIA_ROOT, "segmented", str(uploaded_image.id))
            os.makedirs(segmented_dir, exist_ok=True)
            print(segmented_dir)
            if detection_type == "all":
                detected_objects = None
                model = YOLO(model_choice + ".pt")

                # Run YOLO prediction
                results = model(original_path)
                
                # Save the predicted image
                for result in results:
                    result_image = result.plot()
                    predicted_path = os.path.join(settings.MEDIA_ROOT, "predictions", f"predicted_{uploaded_image.id}.jpg")
                    cv2.imwrite(predicted_path, result_image)
                    uploaded_image.predicted_image = f"predictions/predicted_{uploaded_image.id}.jpg"
                    uploaded_image.save()

                img = cv2.imread(original_path)
                
                
                for i, result in enumerate(results):
                    for j, (box, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
                        xmin, ymin, xmax, ymax = map(int, box)
                        cropped_img = img[ymin:ymax, xmin:xmax]
                        
                        label = result.names[int(cls)]
                        # cv2.imshow(label, cropped_img)
                        # Save the cropped image
                        cropped_img_path = os.path.join(settings.MEDIA_ROOT, "cropped", str(uploaded_image.id), f"{label}_{i}_{j}.jpg")
                        cv2.imwrite(cropped_img_path, cropped_img)
                        cropped_images.append(f"{settings.MEDIA_URL}cropped/{uploaded_image.id}/{label}_{i}_{j}.jpg")
                        # Add the cropped image to the dictionary
                        if label not in label_dict:
                            label_dict[label] = []
                            label_list.append(label)
                        label_dict[label].append(cropped_img_path)   
                         
                # uploaded_image.cropped_images = cropped_images
                # uploaded_image.save()
                
                img_ontology_dict = {label: label for label in label_list}
                
                # Create a directory to save the segmented images

                
                base_model = GroundedSAM2(ontology=CaptionOntology(img_ontology_dict), model="Grounding DINO")
                
                results = base_model.predict(original_path)
                        
                annotated_image = plot(
                    image = cv2.imread(original_path),
                    classes = base_model.ontology.classes(),
                    detections= results.with_nms(),
                    raw = True
                )
                
                segmented_path = os.path.join(segmented_dir, 'original.jpg')
                # Save the annotated image
                cv2.imwrite(segmented_path, annotated_image)
                segmented_images.append(f"{settings.MEDIA_URL}segmented/{uploaded_image.id}/original.jpg")
                
                
                # for filename in os.listdir(cropped_dir):
                #     if filename.endswith(".jpg"):
                #         # Full path to the image
                #         image_path = os.path.join(cropped_dir, filename)
                #         label = filename.split("_")[0]
                #         new_model = GroundedSAM2(ontology=CaptionOntology({label:label}), model="Grounding DINO")
                #         # Run inference on the image
                #         results = new_model.predict(image_path)
                        
                #         annotated_image = plot(
                #             image = cv2.imread(image_path),
                #             classes = new_model.ontology.classes(),
                #             detections= results.with_nms(),
                #             raw = True
                #         )
                        
                #         segmented_path = os.path.join(segmented_dir, filename)
                #         # Save the annotated image
                #         cv2.imwrite(segmented_path, annotated_image)
                #         segmented_images.append(f"{settings.MEDIA_URL}segmented/{uploaded_image.id}/{filename}")
                        
                uploaded_image.segmented_images = segmented_images
                uploaded_image.save()
                        
            else:
                detected_objects = detected_objects.split(",")
                img_ontology_dict = {label: label for label in detected_objects}
                
                model = GroundingDINO(ontology=CaptionOntology(img_ontology_dict))
                
                # Run YOLO prediction
                results = model.predict(original_path)
                
                # Read the input image
                img = plot(
                    image = cv2.imread(original_path),
                    classes = model.ontology.classes(),
                    detections= results.with_nms(),
                    raw = True
                )
                
                print(results)
                
                
                # Save the predicted image with bounding boxes
                predicted_path = os.path.join(settings.MEDIA_ROOT, "predictions", f"predicted_{uploaded_image.id}.jpg")
                cv2.imwrite(predicted_path, img)
                uploaded_image.predicted_image = f"predictions/predicted_{uploaded_image.id}.jpg"
                uploaded_image.save()
                
                # # print(uploaded_image.predicted_image)
                # model = YOLO(model_choice + ".pt")
                img = cv2.imread(original_path)
                # results = model(original_path)
                # # Save each detected object separately
                # for i, result in enumerate(results):
                #     for j, (box, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
                #         xmin, ymin, xmax, ymax = map(int, box)
                #         cropped_img = img[ymin:ymax, xmin:xmax]
                        
                #         label = result.names[int(cls)]
                #         if label not in detected_objects:
                #             continue
                        
                #         cropped_img_path = os.path.join(settings.MEDIA_ROOT, "cropped", str(uploaded_image.id), f"{label}_{i}_{j}.jpg")
                #         cv2.imwrite(cropped_img_path, cropped_img)
                #         cropped_images.append(f"{settings.MEDIA_URL}cropped/{uploaded_image.id}/{label}_{i}_{j}.jpg")
                for i, box in enumerate(results.xyxy):
                    
                    xmin, ymin, xmax, ymax = map(int, box)
                    cropped_img = img[ymin:ymax, xmin:xmax]
                    
                    label = detected_objects[results.class_id[i]]
                    cropped_img_path = os.path.join(settings.MEDIA_ROOT, "cropped", str(uploaded_image.id), f"{label}_{xmin}_{ymin}_{xmax}_{ymax}.jpg")
                    cv2.imwrite(cropped_img_path, cropped_img)
                    cropped_images.append(f"{settings.MEDIA_URL}cropped/{uploaded_image.id}/{label}_{xmin}_{ymin}_{xmax}_{ymax}.jpg")
                
                # print((cropped_images))
                uploaded_image.cropped_images = cropped_images
                uploaded_image.save()
                
                segmented_images = []
                segmented_dir = os.path.join(settings.MEDIA_ROOT, "segmented", str(uploaded_image.id))
                os.makedirs(segmented_dir, exist_ok=True)
                masked_images = []
                masked_dir = os.path.join(settings.MEDIA_ROOT, "masked", str(uploaded_image.id))
                os.makedirs(masked_dir, exist_ok=True)
                
                for filename in os.listdir(cropped_dir):
                    if filename.endswith(".jpg"):
                        # Full path to the image
                        image_path = os.path.join(cropped_dir, filename)
                        label = filename.split("_")[0]
                        new_model = GroundedSAM2(ontology=CaptionOntology({label:label}), model="Grounding DINO")
                        # Run inference on the image
                        results = new_model.predict(image_path)
                        
                        annotated_image = plot(
                            image = cv2.imread(image_path),
                            classes = new_model.ontology.classes(),
                            detections= results.with_nms(),
                            raw = True
                        )
                        
                        segmented_path = os.path.join(segmented_dir, filename)
                        # Save the annotated image
                        cv2.imwrite(segmented_path, annotated_image)
                        segmented_images.append(f"{settings.MEDIA_URL}segmented/{uploaded_image.id}/{filename}")
                        
                        masked = []
                        image = cv2.imread(image_path)
                        mask_img = np.zeros_like(image, dtype=np.uint8)
                        # Draw the detections on the image
                        mask_annotator = sv.MaskAnnotator(color=sv.Color(255,255,255))
                        
                        # Define the path to save the annotated image
                        annotated_image = mask_annotator.annotate(
                            mask_img.copy(), detections=results, 
                        )
                        # Add a suffix to the filename to avoid duplication
                        base_filename, ext = os.path.splitext(filename)
                        new_filename = f"{base_filename}_masked{ext}"
                        masked_path = os.path.join(masked_dir, new_filename)
                        # Save the annotated image
                        cv2.imwrite(masked_path, annotated_image)
                        masked_images.append(f"{settings.MEDIA_URL}masked/{uploaded_image.id}/{new_filename}")
                        
                uploaded_image.segmented_images = segmented_images
                uploaded_image.masked_images = masked_images
                uploaded_image.save()
                        
            return redirect("image_detail", pk=uploaded_image.id)
    else:
        form = ImageForm()
    return render(request, "models/upload_image.html", {"form": form})


def image_detail(request, pk):
    image = Image.objects.get(pk=pk)
    return render(request, "models/image_detail.html", {"image": image})