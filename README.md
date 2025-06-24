Calorie & Portion Size Detection from Plate Images

This project estimates food portions from images using object detection and segmentation. It detects plates, segments food within them, and classifies food items to estimate average density and calorie content.

>  **Course Project**  
> Developed for **\"Deep Learning With PyTorch\"** at **Sofia University**

---

## Installation

### 1. Clone the repository

\`\`\`bash
git clone https://github.com/didntread/food_calories.git
cd food_calories
\`\`\`

### 2. Install dependencies

Make sure you have **Python 3.8+** installed. Then install all required packages:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## Usage

The main script is `portion-detection.py`. You can run it from the command line:

\`\`\`bash
python portion-detection.py <image_path_or_url> [--height HEIGHT_CM] [--density DENSITY]
\`\`\`

### Arguments

| Argument        | Description                                      |
|-----------------|--------------------------------------------------|
| `source`        | Image file path or URL *(required)*             |
| `--height`      | Approximate food height on the plate (in cm)     |
| `--density`     | Override average food density (g/cm³)            |

### Example

\`\`\`bash
python portion-detection.py ./examples/pasta.jpg
\`\`\`

Or with a URL:

\`\`\`bash
python portion-detection.py https://example.com/image.jpg
\`\`\`

---

## What It Does

- Detects **plate** using YOLOv11 trained on plate & utensil images with OBB (Oriented Bounding Box)
- Segments food within the plate using **SAM (Segment Anything Model)**
- Classifies food types using a second food fine-tuned YOLO model
- Estimates:
  - Plate diameter
  - Fill percentage
  - Food area, volume, and mass
  - Average **density (g/cm³)** and **calories per 100g**
  - Total **calorie estimation**

---

## Models Used

| Task                | Model                         |
|---------------------|-------------------------------|
| Plate detection     | YOLOv11 (with OBB support)    |
| Food segmentation   | Meta's SAM (Segment Anything) |
| Food classification | YOLOv11                       |

---

## Output Example

Average food density: 0.57 g/cm³
Average calories per 100g: 390.0 kcal
Estimated portion size: 168.10 g
Estimated calories: 655.60 kcal=