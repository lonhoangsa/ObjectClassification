import os
from dataclasses import dataclass
import os
import subprocess
import sys
import urllib.request
from groundingdino.util.inference import Model

import torch
import numpy as np
import supervision as sv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("WARNING: CUDA not available. GroundingDINO will run very slowly.")

def load_grounding_dino():
    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")

    GROUDNING_DINO_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "groundingdino")

    GROUNDING_DINO_CONFIG_PATH = os.path.join(
        GROUDNING_DINO_CACHE_DIR, "GroundingDINO_SwinT_OGC.py"
    )
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
        GROUDNING_DINO_CACHE_DIR, "groundingdino_swint_ogc.pth"
    )

    try:
        print("trying to load grounding dino directly")
        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE,
        )
        return grounding_dino_model
    except Exception:
        print("downloading dino model weights")
        if not os.path.exists(GROUDNING_DINO_CACHE_DIR):
            os.makedirs(GROUDNING_DINO_CACHE_DIR)

        if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CHECKPOINT_PATH)

        if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
            url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CONFIG_PATH)

        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE,
        )

        # grounding_dino_model.to(DEVICE)

        return grounding_dino_model

def load_SAM():
    cur_dir = os.getcwd()

    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    SAM_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "segment_anything_2")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "sam2_hiera_base_plus.pth")

    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

    os.chdir(SAM_CACHE_DIR)

    if not os.path.isdir("~/.cache/autodistill/segment_anything_2/segment-anything-2"):
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/segment-anything-2.git",
            ]
        )

        os.chdir("segment-anything-2")

        subprocess.run(["pip", "install", "-e", "."])

    sys.path.append("~/.cache/autodistill/segment_anything_2/segment-anything-2")

    # Download the file if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "~/.cache/autodistill/segment_anything_2/sam2_hiera_base_plus.pth"
    checkpoint = os.path.expanduser(checkpoint)
    model_cfg = "sam2_hiera_b+.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    os.chdir(cur_dir)

    return predictor



def combine_detections(detections_list, overwrite_class_ids):
    if len(detections_list) == 0:
        return sv.Detections.empty()

    if overwrite_class_ids is not None and len(overwrite_class_ids) != len(
        detections_list
    ):
        raise ValueError(
            "Length of overwrite_class_ids must match the length of detections_list."
        )

    xyxy = []
    mask = []
    confidence = []
    class_id = []
    tracker_id = []

    for idx, detection in enumerate(detections_list):
        xyxy.append(detection.xyxy)

        if detection.mask is not None:
            mask.append(detection.mask)

        if detection.confidence is not None:
            confidence.append(detection.confidence)

        if detection.class_id is not None:
            if overwrite_class_ids is not None:
                # Overwrite the class IDs for the current Detections object
                class_id.append(
                    np.full_like(
                        detection.class_id, overwrite_class_ids[idx], dtype=np.int64
                    )
                )
            else:
                class_id.append(detection.class_id)

        if detection.tracker_id is not None:
            tracker_id.append(detection.tracker_id)

    xyxy = np.vstack(xyxy)
    mask = np.vstack(mask) if mask else None
    confidence = np.hstack(confidence) if confidence else None
    class_id = np.hstack(class_id) if class_id else None
    tracker_id = np.hstack(tracker_id) if tracker_id else None

    return sv.Detections(
        xyxy=xyxy,
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SamPredictor = load_SAM()

SUPPORTED_GROUNDING_MODELS = ["Grounding DINO"]


@dataclass
class GroundedSAM2(DetectionBaseModel):
    ontology: CaptionOntology
    box_threshold: float
    text_threshold: float

    def __init__(self, ontology: CaptionOntology, model="Grounding DINO", grounding_dino_box_threshold=0.35,
                 grounding_dino_text_threshold=0.25):
        super().__init__(ontology)
        if model not in SUPPORTED_GROUNDING_MODELS:
            raise ValueError(
                f"Grounding model {model} is not supported. Supported models are {SUPPORTED_GROUNDING_MODELS}")

        self.ontology = ontology
        # if model == "Florence 2":
        #     self.florence_2_predictor = Florence2(ontology=ontology)
        if model == "Grounding DINO":
            self.grounding_dino_model = load_grounding_dino()
        self.sam_2_predictor = SamPredictor
        self.model = model
        self.grounding_dino_box_threshold = grounding_dino_box_threshold
        self.grounding_dino_text_threshold = grounding_dino_text_threshold

    def predict(self, input: Any) -> sv.Detections:
        image = load_image(input, return_format="cv2")

        # if self.model == "Florence 2":
        #     detections = self.florence_2_predictor.predict(image)
        if self.model == "Grounding DINO":
            # GroundingDINO predictions
            detections_list = []

            for i, description in enumerate(self.ontology.prompts()):
                # detect objects
                detections = self.grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=[description],
                    box_threshold=self.grounding_dino_box_threshold,
                    text_threshold=self.grounding_dino_text_threshold,
                )

                detections_list.append(detections)

            detections = combine_detections(
                detections_list, overwrite_class_ids=range(len(detections_list))
            )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_2_predictor.set_image(image)
            result_masks = []
            for box in detections.xyxy:
                masks, scores, _ = self.sam_2_predictor.predict(
                    box=box, multimask_output=False
                )
                index = np.argmax(scores)
                masks = masks.astype(bool)
                result_masks.append(masks[index])

        detections.mask = np.array(result_masks)

        return detections

