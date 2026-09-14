import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from PIL import Image

def detect_trees_hybrid(image: Image.Image, min_peak_distance: int = 14, green_thresh: int = 40):
    """
    Optimistic tree crown segmentation & peak detection for dense canopies.
    """
    # Convert PIL to BGR OpenCV format
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # 1. CLAHE enhancement to expose shaded/lower crowns
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

    # 2. Excess Green Index (ExG) for foliage isolation
    b_ch, g_ch, r_ch = cv2.split(enhanced_bgr.astype(np.float32))
    exg = (2 * g_ch - r_ch - b_ch) / (g_ch + r_ch + b_ch + 1e-5)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Gaussian smoothing to suppress noisy leaf texture
    smoothed = cv2.GaussianBlur(exg_norm, (7, 7), 2)

    # 4. Local Maxima Detection (finds high & low crown apexes)
    coords = peak_local_max(
        smoothed, 
        min_distance=min_peak_distance, 
        threshold_abs=green_thresh, 
        exclude_border=False
    )

    # 5. Marker-controlled Watershed
    markers = np.zeros(smoothed.shape, dtype=np.int32)
    for idx, (y, x) in enumerate(coords, 1):
        markers[y, x] = idx

    canopy_mask = (exg_norm > green_thresh).astype(np.uint8)
    labels = watershed(-smoothed, markers, mask=canopy_mask)

    # 6. Generate synthetic bounding boxes & overlay
    boxes = []
    annotated = np.array(image).copy()
    boxes = []
    box_radius = int(min_peak_distance * 0.8)
    
    for y, x in coords:
        xmin = max(0, int(x - box_radius))
        ymin = max(0, int(y - box_radius))
        xmax = min(w, int(x + box_radius))
        ymax = min(h, int(y + box_radius))
        boxes.append((xmin, ymin, xmax, ymax))
        
        # Draw bounding box
        cv2.rectangle(annotated, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 3)
    crown_pixels_count = int(np.count_nonzero(canopy_mask))
    canopy_density = (crown_pixels_count / (h * w)) * 100.0

    return {
        "tree_count": len(boxes),
        "canopy_density": round(canopy_density, 1),
        "crown_pixels": crown_pixels_count,
        "boxes": boxes,
        "annotated_image": annotated
    }