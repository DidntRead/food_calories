#%%
import cv2
import numpy as np
import statistics
import requests
import argparse
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

yolo_plate_model = YOLO("yolo11m-obb-plate.pt")
yolo_model = YOLO("yolo11m.pt")

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").cuda()
predictor = SamPredictor(sam)

reference_sizes = {
    "Fork": 18.5,
    "Knife": 21.0,
    "Spoon": 17.0,
    "Plate": 25.0,
}

food_reference = {
    'apple':     {'density': 0.61, 'calories_per_100g': 52},
    'banana':    {'density': 0.94, 'calories_per_100g': 89},
    'chicken':   {'density': 1.05, 'calories_per_100g': 165},
    'rice':      {'density': 0.85, 'calories_per_100g': 130},
    'broccoli':  {'density': 0.65, 'calories_per_100g': 35},
    'pasta':     {'density': 0.85, 'calories_per_100g': 158},
    'egg':       {'density': 1.03, 'calories_per_100g': 155},
    'cheese':    {'density': 1.11, 'calories_per_100g': 402},
    'cake':      {'density': 0.57, 'calories_per_100g': 390},
}
#%%
def rotated_box_to_length(box):
    """Compute the longest side of an oriented box."""
    pts = np.array(box).reshape(4, 2)
    d1 = np.linalg.norm(pts[0] - pts[1])
    d2 = np.linalg.norm(pts[1] - pts[2])
    return max(d1, d2)

def calculate_scale_obb(detections, ref_sizes):
    """Estimate pixel/cm scale from oriented reference object boxes."""
    ratios = []
    # print(detections)
    for det in detections:
        name = det['name']
        if name in ref_sizes:
            length_px = rotated_box_to_length(det['obb'])
            length_cm = ref_sizes[name]
            ratios.append(length_px / length_cm)
    return statistics.median(ratios) if ratios else None

def expand_obb(obb, scale=1.1):
    """
    Expand an OBB outward by scaling its width and height while preserving rotation.

    Args:
        obb: 4x2 array of corner points
        scale: float >1 to expand, <1 to shrink
    Returns:
        Expanded 4x2 OBB
    """
    rect = cv2.minAreaRect(obb.astype(np.float32))  # (center(x, y), (w, h), angle)
    (cx, cy), (w, h), angle = rect
    w *= scale
    h *= scale
    expanded_box = cv2.boxPoints(((cx, cy), (w, h), angle))
    return expanded_box

def get_rotated_crop(image, points):
    """Crop a rotated rectangle from image using perspective transform."""
    pts = np.array(points, dtype="float32")
    width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0])))

    dst = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, width, height

#%%
def detect_objects(image):
    results = yolo_plate_model(image)[0]
    detections = []

    for box in results.obb:
        cls_id = int(box.cls)
        name = yolo_plate_model.names[cls_id]
        obb = box.xyxyxyxy.cpu().numpy().reshape(4, 2)
        detections.append({
            'name': name,
            'obb': obb
        })
    return detections

def debug_segmentation(image, plate_obb, mask, cropped_mask):
    """Plot intermediate debug outputs."""
    debug_img = image.copy()
    pts = plate_obb.astype(np.int32)
    cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")

    axs[1].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Image with Plate OBB")

    axs[2].imshow(mask, cmap='gray')
    axs[2].set_title("Full SAM Mask")

    axs[3].imshow(cropped_mask, cmap='gray')
    axs[3].set_title("Cropped Plate Mask")

    for ax in axs:
        ax.axis('off')
    plt.show()

def get_center_and_corners(obb):
    pts = np.array(obb, dtype=np.float32)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    corners = pts
    midpoints = [(pts[i] + pts[(i + 1) % 4]) / 2 for i in range(4)]
    return np.array([[cx, cy]] + midpoints)

def segment_plate_fill(image, plate_obb):
    predictor.set_image(image)

    input_points = get_center_and_corners(plate_obb)
    input_labels = np.ones(len(input_points), dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    best_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    plate_polygon_mask = np.zeros_like(best_mask)
    cv2.fillPoly(plate_polygon_mask, [plate_obb.astype(np.int32)], 255)
    clipped_mask = cv2.bitwise_and(best_mask, plate_polygon_mask)

    cropped_mask, w, h = get_rotated_crop(clipped_mask, plate_obb)
    mask_pixels_in_plate = np.count_nonzero(cropped_mask)
    plate_area_px = w * h
    fill_percent = (mask_pixels_in_plate / plate_area_px) * 100 if plate_area_px > 0 else 0

    return {
        "fill_percent": fill_percent,
        "mask_pixels_in_plate": mask_pixels_in_plate,
        "plate_area_px": plate_area_px,
        "full_mask": best_mask,
        "cropped_mask": cropped_mask,
        "plate_obb": plate_obb
    }
#%%
def load_image_from_file(path):
    """Load an image from a local file."""
    return cv2.imread(path)

def load_image_from_url(url):
    """Load an image from a URL."""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#%%
def get_masked_plate_image(image, plate_obb, mask):
    """
    Crop the plate region and apply the SAM mask as a mask over the plate.

    Args:
        image: Original BGR image
        plate_obb: 4x2 array of float32
        mask: Full-size SAM mask (uint8, 0/255)
        expand_ratio: How much to grow the plate OBB before cropping

    Returns:
        masked_plate (BGR) image where only masked food is visible
    """
    plate_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(plate_mask, [plate_obb.astype(np.int32)], 255)

    clipped_mask = cv2.bitwise_and(mask, plate_mask)

    warped_img, w, h = get_rotated_crop(image, plate_obb)
    warped_mask, _, _ = get_rotated_crop(clipped_mask, plate_obb)

    # Apply mask to cropped image
    masked_plate = cv2.bitwise_and(warped_img, warped_img, mask=warped_mask)

    return masked_plate

def detect_food_items(image, plate_obb, full_mask):
    masked_plate = get_masked_plate_image(image, plate_obb, full_mask)
    results = yolo_model(masked_plate)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        name = yolo_model.names[cls_id]
        xyxy = box.xyxy.cpu().numpy().flatten().tolist()

        detections.append({
            "name": name,
            "conf": conf,
            "bbox": xyxy
        })

    return detections

def compute_avg_food_density_and_calories(image, plate_obb, full_mask):
    matched = []
    detections = detect_food_items(image, expand_obb(plate_obb, 0.9), full_mask)

    # print(detections)

    for item in detections:
        name = item['name'].lower()
        if name in food_reference:
            ref = food_reference[name]
            matched.append({
                'density': ref['density'],
                'calories': ref['calories_per_100g']
            })

    if not matched:
        return {
            'avg_density': None,
            'avg_calories_per_100g': None,
            'matched_items': [],
            'unmatched_count': len(detections)
        }

    avg_density = sum(x['density'] for x in matched) / len(matched)
    avg_calories = sum(x['calories'] for x in matched) / len(matched)

    return {
        'avg_density': round(avg_density, 3),
        'avg_calories_per_100g': round(avg_calories, 1),
        'matched_items': [item['name'] for item in detections if item['name'].lower() in food_reference],
        'unmatched_count': len(detections) - len(matched)
    }
#%%
def process_image(image):
    """
    Process an image (as NumPy array):
    - Detect objects
    - Compute plate diameter in cm
    - Estimate plate fill percentage
    """
    detections = detect_objects(image)

    scale = calculate_scale_obb(detections, reference_sizes)
    if not scale:
        raise RuntimeError("No usable reference object found for scaling.")

    plate = next((d for d in detections if d['name'] == 'Plate'), None)
    if not plate:
        raise RuntimeError("No plate detected.")

    diameter_px = rotated_box_to_length(plate['obb'])
    diameter_cm = diameter_px / scale

    plate_obb = expand_obb(plate['obb'])
    result = segment_plate_fill(image, np.array(plate_obb, dtype=np.float32))

    return {
        "diameter_cm": diameter_cm,
        "fill_percent": result['fill_percent'],
        "full_mask": result['full_mask'],
        "plate_obb": plate_obb
    }

def estimate_portion_size(image, average_density=0.85, average_height=3):
    result = process_image(image)
    fill_percent = result['fill_percent']

    portion_size = float(fill_percent) * average_density * average_height
    # print(f"Estimated portion size: {portion_size:.2f} g")

    return portion_size

def estimate_portion_size_and_calories(image, average_height=3, average_density=0.85):
    result = process_image(image)
    fill_percent = result['fill_percent']
    full_mask = result['full_mask']
    plate_obb = result['plate_obb']

    result = compute_avg_food_density_and_calories(image, plate_obb, full_mask)

    avg_density = average_density if result['avg_density'] is None else result['avg_density']
    portion_size = float(fill_percent) * avg_density * average_height

    if result['avg_calories_per_100g'] is None:
        result['avg_calories_per_100g'] = 0

    calories = (portion_size / 100) * result['avg_calories_per_100g']

    # print(f"Average food density: {avg_density} g/cm³")
    # print(f"Average calories per 100g: {result['avg_calories_per_100g']} kcal")
    # print(f"Estimated portion size: {portion_size:.2f} g")
    # print(f"Estimated calories: {calories:.2f} kcal")

    return {
        "portion_size_g": portion_size,
        "calories": calories,
        "avg_density": avg_density,
        "avg_calories_per_100g": result['avg_calories_per_100g'],
    }
#%%
# image = load_image_from_url("https://www.gosupps.com/media/catalog/product/cache/25/image/1500x/040ec09b1e35df139433887a97daa66f/5/1/51slJB3xzdL.jpg")
# result = estimate_portion_size_and_calories(image)
# print(result)
#%%
# image = load_image_from_url("https://t3.ftcdn.net/jpg/02/00/23/02/360_F_200230206_gwEsUBQEgL0lCDrxt3meUBYZkNhGBkPi.jpg")
# detections = detect_objects(image)
# plate = next(d for d in detections if d['name'] == 'Plate')
# plate_obb = np.array(plate['obb'], dtype=np.float32)
#
# result = segment_plate_fill(image, expand_obb(plate_obb))
#
# print(f"Plate area: {result['plate_area_px']} px")
# print(f"Food pixels inside plate: {result['mask_pixels_in_plate']} px")
# print(f"Plate fill: {result['fill_percent']:.2f}%")
#
# debug_segmentation(image, expand_obb(plate_obb, 1.1), result['full_mask'], result['cropped_mask'])
#
# masked_plate = get_masked_plate_image(image, plate_obb, result['full_mask'])
#
# # Show it
# plt.imshow(cv2.cvtColor(masked_plate, cv2.COLOR_BGR2RGB))
# plt.title("Masked Plate Image")
# plt.axis('off')
# plt.show()
#
# foods = detect_food_items(image, expand_obb(plate_obb, 1.1), result['full_mask'])
#
# print(foods)
#
# def draw_detections(image, detections):
#     img = image.copy()
#     for det in detections:
#         x1, y1, x2, y2 = map(int, det['bbox'])
#         label = f"{det['name']} {det['conf']:.2f}"
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, label, (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#     return img
#
# annotated = draw_detections(masked_plate, foods)
# plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
# plt.title("Food Detections on Plate")
# plt.axis('off')
# plt.show()
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a food plate image.")

    # Positional argument (no dashes)
    parser.add_argument("source", help="Image path or URL")

    # Optional arguments
    parser.add_argument("--density", type=float, help="Override average food density (g/cm³)")
    parser.add_argument("--height", type=float, help="Food pile height (in cm)")

    args = parser.parse_args()

    params = {
        "image": load_image_from_file(args.source) if args.source.startswith("http") is False else load_image_from_url(args.source),
    }

    if args.density is not None:
        params["average_density"] = args.density

    if args.height is not None:
        params["average_height"] = args.height

    try:
        result = estimate_portion_size_and_calories(**params)

        print(f"Average food density: {result['avg_density']} g/cm³")
        print(f"Average calories per 100g: {result['avg_calories_per_100g']} kcal")
        print(f"Estimated portion size: {result['portion_size_g']:.2f} g")
        print(f"Estimated calories: {result['calories']:.2f} kcal")
    except Exception as e:
        print(f"Error: {e}")
