import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from _helpers import (
    draw_fitted_ellipse,
    draw_oriented_box,
    draw_polyline,
    get_region_color,
    read_images,
    write_image,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FACE_DETECTOR_MODEL = "src/models/blaze_face_short_range.tflite"
# FACE_DETECTOR_MODEL = "src/models/blaze_face_full_range.tflite"
FACE_LANDMARKER_MODEL = "src/models/face_landmarker.task"

MAX_FACES = 20
MIN_FACE_DETECTION_CONFIDENCE = 0.2

# Padding added around each detected face, as a fraction
# of the detected face's width and height.
FACE_PADDING = 0.25

DRAW_LANDMARKS = False
DRAW_FACE_BOXES = True
DRAW_REGION_LABELS = True


# ---------------------------------------------------------
# Facial regions
# MediaPipe Face Landmarker landmark indices
# ---------------------------------------------------------

RAW_REGIONS = json.load(Path("src/models/landmarks.json").open())
DRAW_ORDER = [
    # "nose",
    "outer_mouth",
    "inner_mouth",
    "left_eye",
    "right_eye",
    "left_inner_eye",
    "right_inner_eye",
    "left_iris",
    "right_iris",
]
FACE_REGIONS = {key: RAW_REGIONS[key] for key in DRAW_ORDER}

REGION_COLORS = {name: get_region_color(name) for name in FACE_REGIONS}


# ---------------------------------------------------------
# MediaPipe model configuration
# ---------------------------------------------------------

face_detector_options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=FACE_DETECTOR_MODEL),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    min_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
)

face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------


def create_mp_image(image):
    """Convert an OpenCV BGR image to a MediaPipe RGB image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )


def get_padded_face_crop(image, bbox, padding=FACE_PADDING):
    """
    Crop a detected face with padding.

    Returns:
        crop: Cropped BGR image.
        origin_x: X coordinate of the crop in the original image.
        origin_y: Y coordinate of the crop in the original image.
    """
    height, width = image.shape[:2]

    x1 = bbox.origin_x
    y1 = bbox.origin_y
    x2 = x1 + bbox.width
    y2 = y1 + bbox.height

    pad_x = int(bbox.width * padding)
    pad_y = int(bbox.height * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None, 0, 0

    crop = image[y1:y2, x1:x2].copy()

    return crop, x1, y1


# ---------------------------------------------------------
# Main processing function
# ---------------------------------------------------------


def draw_face_regions(
    image,
    face_detector,
    face_landmarker,
    type_of_overlay: str,
):
    """
    Detect faces, crop them, detect facial landmarks on each crop,
    and draw facial-region boxes on the original image.

    Args:
        image: OpenCV BGR image.
        face_detector: Initialized MediaPipe FaceDetector.
        face_landmarker: Initialized MediaPipe FaceLandmarker.

    Returns:
        Annotated copy of the original image.
    """
    if image is None:
        raise ValueError("Input image is None")

    output = image.copy()

    height, width = image.shape[:2]

    # Step 1: Detect faces in the original image.
    detection_result = face_detector.detect(create_mp_image(image))

    detections = detection_result.detections[:MAX_FACES]

    print(f"Detected {len(detections)} face(s)")

    # Step 2: Process each detected face independently.
    for face_number, detection in enumerate(detections, start=1):
        bbox = detection.bounding_box

        if DRAW_FACE_BOXES:
            x1 = max(0, bbox.origin_x)
            y1 = max(0, bbox.origin_y)
            x2 = min(width - 1, bbox.origin_x + bbox.width)
            y2 = min(height - 1, bbox.origin_y + bbox.height)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                1,
            )

        # Step 3: Crop the face with padding.
        crop, offset_x, offset_y = get_padded_face_crop(
            image,
            bbox,
        )

        if crop is None or crop.size == 0:
            continue

        # Step 4: Run Face Landmarker on the face crop.
        landmark_result = face_landmarker.detect(create_mp_image(crop))

        if not landmark_result.face_landmarks:
            print(
                f"Face {face_number}: detector found a face, "
                "but the landmarker found no landmarks."
            )
            continue

        crop_height, crop_width = crop.shape[:2]

        # Step 5: Convert crop-relative landmarks into
        # original-image pixel coordinates.
        for face_landmarks in landmark_result.face_landmarks:
            points = np.array(
                [
                    [
                        landmark.x * crop_width + offset_x,
                        landmark.y * crop_height + offset_y,
                    ]
                    for landmark in face_landmarks
                ],
                dtype=np.float32,
            )

            # Draw facial landmarks.
            if DRAW_LANDMARKS:
                for px, py in points.astype(np.int32):
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(
                            output,
                            (int(px), int(py)),
                            1,
                            (0, 255, 0),
                            -1,
                        )

            # Step 6: Draw oriented boxes around facial regions.
            for region_name, indices in FACE_REGIONS.items():
                if max(indices) >= len(points):
                    continue

                region_points = points[indices]

                if type_of_overlay == "ellipse":
                    output = draw_fitted_ellipse(
                        output,
                        region_points,
                        REGION_COLORS[region_name],
                    )
                elif type_of_overlay == "polylines":
                    output = draw_polyline(
                        output,
                        region_points,
                        REGION_COLORS[region_name],
                    )
                else:
                    draw_oriented_box(
                        output,
                        region_points,
                        REGION_COLORS[region_name],
                        label=region_name,
                    )

    return output


# ---------------------------------------------------------
# Run on images
# ---------------------------------------------------------


def main():
    image_paths = read_images()

    with (
        mp.tasks.vision.FaceDetector.create_from_options(
            face_detector_options
        ) as face_detector,
        mp.tasks.vision.FaceLandmarker.create_from_options(
            face_landmarker_options
        ) as face_landmarker,
    ):
        for image_path in image_paths:
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            for overlay in ["bbox", "polylines", "ellipse"]:
                output = draw_face_regions(
                    image,
                    face_detector,
                    face_landmarker,
                    overlay,
                )

                write_image(
                    image_path,
                    output,
                    f"mediapipe/face/{overlay}",
                )


if __name__ == "__main__":
    main()
