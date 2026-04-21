import numpy as np
import cv2
from PIL import Image, ImageDraw
from pyzbar.pyzbar import decode


# =========================
# COMMON UTILS
# =========================

def rgb_to_grayscale(img):
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return gray.astype(np.float32)


def pad_image(img, pad_h, pad_w, mode='edge'):
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)


def convolve2d(image, kernel):
    image = image.astype(np.float32)
    kernel = np.array(kernel, dtype=np.float32)

    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = pad_image(image, pad_h, pad_w, mode='edge')
    output = np.zeros_like(image, dtype=np.float32)

    kernel = np.flipud(np.fliplr(kernel))

    H, W = image.shape
    for i in range(H):
        for j in range(W):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return output


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_blur(image, size=5, sigma=1.0):
    image = image.astype(np.float32)
    kernel = gaussian_kernel(size=size, sigma=sigma)
    return convolve2d(image, kernel)


def connected_components(binary):
    """
    Find all 8-connected components of foreground pixels (value = 1)
    in a binary image.
    """
    H, W = binary.shape
    visited = np.zeros((H, W), dtype=bool)
    components = []

    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for row in range(H):
        for col in range(W):
            if binary[row, col] != 1 or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            component = []

            while stack:
                x, y = stack.pop()
                component.append((x, y))

                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if binary[nx, ny] == 1 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))

            components.append(component)

    return components


def component_bounding_box(component):
    rows = [p[0] for p in component]
    cols = [p[1] for p in component]
    return min(rows), min(cols), max(rows), max(cols)


def draw_boxes(image, candidates, color=(255, 0, 0), width=3):
    pil_img = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    for cand in candidates:
        r_min, c_min, r_max, c_max = cand["box"]
        draw.rectangle([c_min, r_min, c_max, r_max], outline=color, width=width)

    return np.array(pil_img)


# =========================
# BARCODE PART
# =========================

def sobel_gradients(image):
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    Ky = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    gx = convolve2d(image, Kx)
    gy = convolve2d(image, Ky)

    magnitude = np.sqrt(gx**2 + gy**2)
    return gx, gy, magnitude


def barcode_response(gx, gy):
    response = np.abs(gx) - np.abs(gy)
    response = np.maximum(response, 0)
    return response


def threshold_image(image, thresh):
    return (image >= thresh).astype(np.uint8)


def dilate(binary, kernel_shape=(3, 3), iterations=1):
    out = binary.copy()
    kh, kw = kernel_shape
    pad_h = kh // 2
    pad_w = kw // 2

    for _ in range(iterations):
        padded = np.pad(out, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        new_out = np.zeros_like(out)

        H, W = out.shape
        for i in range(H):
            for j in range(W):
                region = padded[i:i + kh, j:j + kw]
                new_out[i, j] = 1 if np.any(region == 1) else 0

        out = new_out

    return out


def erode(binary, kernel_size=3, iterations=1):
    out = binary.copy()
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = np.pad(out, ((pad, pad), (pad, pad)), mode='constant')
        new_out = np.zeros_like(out)

        H, W = out.shape
        for i in range(H):
            for j in range(W):
                region = padded[i:i + kernel_size, j:j + kernel_size]
                new_out[i, j] = 1 if np.all(region == 1) else 0

        out = new_out

    return out


def closing(binary, kernel_size=5, iterations=1):
    return erode(dilate(binary, kernel_size, iterations), kernel_size, iterations)


def count_transitions(region):
    if region.shape[0] == 0 or region.shape[1] == 0:
        return 0

    transitions = 0
    threshold = np.mean(region)

    for row in region:
        binary_row = row > threshold
        transitions += np.sum(binary_row[:-1] != binary_row[1:])

    return transitions / region.shape[0]


def filter_barcode_candidates_final(components, gray, gx, gy):
    candidates = []

    for comp in components:
        r_min, c_min, r_max, c_max = component_bounding_box(comp)

        h = r_max - r_min + 1
        w = c_max - c_min + 1

        if h <= 0 or w <= 0:
            continue

        area_pixels = len(comp)
        aspect_ratio = w / h

        region = gray[r_min:r_max + 1, c_min:c_max + 1]
        transitions = count_transitions(region)

        gx_region = gx[r_min:r_max + 1, c_min:c_max + 1]
        gy_region = gy[r_min:r_max + 1, c_min:c_max + 1]
        edge_ratio = np.mean(np.abs(gx_region)) / (np.mean(np.abs(gy_region)) + 1e-8)

        fill_ratio = area_pixels / (h * w)

        if aspect_ratio < 1.2:
            continue
        if area_pixels < 300:
            continue
        if transitions < 2:
            continue
        if edge_ratio < 1.2:
            continue
        if fill_ratio < 0.05:
            continue

        candidates.append({
            "box": (r_min, c_min, r_max, c_max),
            "aspect_ratio": aspect_ratio,
            "area_pixels": area_pixels,
            "transitions": transitions,
            "edge_ratio": edge_ratio,
            "fill_ratio": fill_ratio
        })

    return candidates


def select_best_candidate(candidates):
    if len(candidates) == 0:
        return None

    best = max(
        candidates,
        key=lambda c: c["transitions"] * c["edge_ratio"]
    )

    return best


def detect_barcodes(image):
    gray = rgb_to_grayscale(image)
    blurred = gaussian_blur(gray, size=3, sigma=1.2)

    gx, gy, _ = sobel_gradients(blurred)

    resp = barcode_response(gx, gy)
    resp_norm = resp / (resp.max() + 1e-8)

    binary = threshold_image(resp_norm, 0.25)
    dilated = dilate(binary, kernel_shape=(5, 21), iterations=2)

    components = connected_components(dilated)

    candidates = filter_barcode_candidates_final(
        components,
        gray,
        gx,
        gy
    )

    best = select_best_candidate(candidates)

    if best is None:
        result = image.copy()
        final_candidates = []
    else:
        result = draw_boxes(image, [best])
        final_candidates = [best]

    debug = {
        "gray": gray,
        "blurred": blurred,
        "resp": resp,
        "binary": binary,
        "dilated": dilated,
        "components": components,
        "all_candidates": candidates,
        "best_candidate": best
    }

    return result, final_candidates, debug


def generate_barcode_versions(roi):
    versions = []

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    big = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    versions.append(("original", roi))
    versions.append(("gray_big", big))
    versions.append(("gray", gray))

    eq = cv2.equalizeHist(big)
    versions.append(("equalized", eq))

    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    versions.append(("blur", blur))

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions.append(("otsu", otsu))

    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    versions.append(("otsu_inv", otsu_inv))

    adaptive = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 8
    )
    versions.append(("adaptive", adaptive))

    kernel_sharp = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharp = cv2.filter2D(big, -1, kernel_sharp)
    versions.append(("sharp", sharp))

    return versions


def rotate_image_cv(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def try_decode_barcode_roi(roi):
    angles = [0, 90, 180, 270]

    for angle in angles:
        rotated = rotate_image_cv(roi, angle)

        for name, version in generate_barcode_versions(rotated):
            decoded = decode(version)

            if decoded:
                d = decoded[0]
                try:
                    text = d.data.decode("utf-8")
                except Exception:
                    text = str(d.data)

                return {
                    "success": True,
                    "text": text,
                    "type": d.type,
                    "angle": angle,
                    "version": name
                }

    return {
        "success": False,
        "text": None,
        "type": None,
        "angle": None,
        "version": None
    }


def decode_detected_barcodes(image, candidates, padding=12):
    H, W = image.shape[:2]
    decoded_results = []

    for cand in candidates:
        r_min, c_min, r_max, c_max = cand["box"]

        r0 = max(0, r_min - padding)
        c0 = max(0, c_min - padding)
        r1 = min(H - 1, r_max + padding)
        c1 = min(W - 1, c_max + padding)

        roi = image[r0:r1 + 1, c0:c1 + 1]
        result = try_decode_barcode_roi(roi)

        decoded_results.append({
            "box": (r_min, c_min, r_max, c_max),
            "padded_box": (r0, c0, r1, c1),
            "decoded": result["success"],
            "text": result["text"],
            "type": result["type"],
            "angle": result["angle"],
            "version": result["version"]
        })

    return decoded_results


# =========================
# QR PART
# =========================

def local_mean_threshold(image, window_size=15, C=10):
    """
    Apply local mean thresholding.

    Returns:
        binary image (1 = black, 0 = background)
    """
    image = image.astype(np.float32)

    pad = window_size // 2
    padded = np.pad(image, pad_width=pad, mode='reflect')

    H, W = image.shape
    binary = np.zeros((H, W), dtype=np.uint8)

    for row in range(H):
        for col in range(W):
            window = padded[row:row + window_size, col:col + window_size]
            threshold = np.mean(window) - C
            binary[row, col] = 1 if image[row, col] < threshold else 0

    return binary


def filter_square_candidates(components, min_area=80, aspect_tol=0.35, min_fill=0.4):
    """
    Filter connected components that look like squares (QR finder candidates).
    """
    candidates = []

    for comp in components:
        r_min, c_min, r_max, c_max = component_bounding_box(comp)

        h = r_max - r_min + 1
        w = c_max - c_min + 1

        if h == 0 or w == 0:
            continue

        area_pixels = len(comp)
        if area_pixels < min_area:
            continue

        area_box = h * w
        aspect_ratio = w / h
        fill_ratio = area_pixels / area_box

        if abs(aspect_ratio - 1.0) > aspect_tol:
            continue

        if fill_ratio < min_fill:
            continue

        cx = (c_min + c_max) / 2
        cy = (r_min + r_max) / 2
        size = (w + h) / 2

        candidates.append({
            "box": (r_min, c_min, r_max, c_max),
            "center": (cx, cy),
            "w": w,
            "h": h,
            "size": size,
            "fill_ratio": fill_ratio,
            "pixel_area": area_pixels
        })

    return candidates


def euclidean_distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]

    norm1 = np.hypot(v1[0], v1[1])
    norm2 = np.hypot(v2[0], v2[1])

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 180.0

    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.degrees(np.arccos(cos_theta))


def find_qr_finder_triplets(candidates,
                            size_tol=0.5,
                            min_dist_factor=2.0,
                            max_angle_deviation=25):
    triplets = []
    n = len(candidates)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                trio = [candidates[i], candidates[j], candidates[k]]

                sizes = [c["size"] for c in trio]
                s_min, s_max = min(sizes), max(sizes)

                if s_min <= 0:
                    continue
                if (s_max - s_min) / s_min > size_tol:
                    continue

                centers = [c["center"] for c in trio]
                mean_size = np.mean(sizes)

                best_score = None
                best_config = None

                for corner_idx in range(3):
                    other_idx = [idx for idx in range(3) if idx != corner_idx]

                    pc = centers[corner_idx]
                    p1 = centers[other_idx[0]]
                    p2 = centers[other_idx[1]]

                    v1 = (p1[0] - pc[0], p1[1] - pc[1])
                    v2 = (p2[0] - pc[0], p2[1] - pc[1])

                    d1 = euclidean_distance(pc, p1)
                    d2 = euclidean_distance(pc, p2)

                    if d1 < min_dist_factor * mean_size or d2 < min_dist_factor * mean_size:
                        continue

                    angle = angle_between(v1, v2)

                    if abs(angle - 90) > max_angle_deviation:
                        continue

                    dist_balance = abs(d1 - d2) / max(d1, d2)
                    score = abs(angle - 90) + 20 * dist_balance

                    if best_score is None or score < best_score:
                        best_score = score
                        best_config = {
                            "corner_idx": corner_idx,
                            "score": score,
                            "triplet": trio
                        }

                if best_config is not None:
                    triplets.append(best_config)

    return triplets


def triplet_to_qr_box(triplet, padding_factor=0.8):
    boxes = [c["box"] for c in triplet]

    r_min = min(b[0] for b in boxes)
    c_min = min(b[1] for b in boxes)
    r_max = max(b[2] for b in boxes)
    c_max = max(b[3] for b in boxes)

    h = r_max - r_min + 1
    w = c_max - c_min + 1
    size = max(h, w)

    pad = int(size * padding_factor * 0.5)

    return (
        r_min - pad,
        c_min - pad,
        r_max + pad,
        c_max + pad
    )


def clamp_box(box, H, W):
    r_min, c_min, r_max, c_max = box

    return (
        max(0, r_min),
        max(0, c_min),
        min(H - 1, r_max),
        min(W - 1, c_max)
    )


def box_iou(box1, box2):
    r1_min, c1_min, r1_max, c1_max = box1
    r2_min, c2_min, r2_max, c2_max = box2

    inter_r_min = max(r1_min, r2_min)
    inter_c_min = max(c1_min, c2_min)
    inter_r_max = min(r1_max, r2_max)
    inter_c_max = min(c1_max, c2_max)

    if inter_r_max < inter_r_min or inter_c_max < inter_c_min:
        return 0.0

    inter_area = (inter_r_max - inter_r_min + 1) * (inter_c_max - inter_c_min + 1)
    area1 = (r1_max - r1_min + 1) * (c1_max - c1_min + 1)
    area2 = (r2_max - r2_min + 1) * (c2_max - c2_min + 1)

    return inter_area / (area1 + area2 - inter_area + 1e-8)


def box_area(box):
    r_min, c_min, r_max, c_max = box
    return (r_max - r_min + 1) * (c_max - c_min + 1)


def deduplicate_boxes(boxes, iou_thresh=0.5):
    if not boxes:
        return []

    boxes = sorted(boxes, key=box_area)
    kept = []

    for box in boxes:
        duplicate = False
        for k in kept:
            if box_iou(box, k) > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    return kept


def detect_qr_codes(image, return_debug=False):
    H, W = image.shape[:2]

    gray = rgb_to_grayscale(image)
    blurred = gaussian_blur(gray, size=5, sigma=1.0)
    binary = local_mean_threshold(blurred, window_size=15, C=10)

    components = connected_components(binary)

    square_candidates = filter_square_candidates(
        components,
        min_area=80,
        aspect_tol=0.35,
        min_fill=0.4
    )

    triplets_info = find_qr_finder_triplets(
        square_candidates,
        size_tol=0.5,
        min_dist_factor=2.0,
        max_angle_deviation=25
    )

    qr_boxes = []
    for item in triplets_info:
        triplet = item["triplet"]
        box = triplet_to_qr_box(triplet, padding_factor=0.8)
        box = clamp_box(box, H, W)
        qr_boxes.append(box)

    qr_boxes = deduplicate_boxes(qr_boxes, iou_thresh=0.4)

    if not return_debug:
        return qr_boxes

    debug = {
        "gray": gray,
        "blurred": blurred,
        "binary": binary,
        "components": components,
        "square_candidates": square_candidates,
        "triplets_info": triplets_info
    }

    return qr_boxes, debug


detector = cv2.QRCodeDetector()


def decode_qr_from_box(image, box, padding=8):
    r_min, c_min, r_max, c_max = box
    H, W = image.shape[:2]

    r0 = max(0, r_min - padding)
    c0 = max(0, c_min - padding)
    r1 = min(H - 1, r_max + padding)
    c1 = min(W - 1, c_max + padding)

    roi = image[r0:r1 + 1, c0:c1 + 1]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)

    text, points, straight = detector.detectAndDecode(roi_bgr)

    if text:
        return {"decoded": True, "text": text, "box": box}

    roi_big = cv2.resize(roi_bgr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    text, points, straight = detector.detectAndDecode(roi_big)

    if text:
        return {"decoded": True, "text": text, "box": box}

    return {"decoded": False, "text": None, "box": box}


def decode_detected_qrs(image, qr_boxes, padding=8):
    return [
        decode_qr_from_box(image, box, padding=padding)
        for box in qr_boxes
    ]


# =========================
# UNIFIED PART
# =========================

def draw_unified_results(image, decoded_barcodes, decoded_qrs):
    out = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(out)

    for item in decoded_barcodes:
        r_min, c_min, r_max, c_max = item["box"]

        if item["decoded"]:
            color = (0, 255, 0)
            label = f"{item['type']}: {item['text']}"
        else:
            color = (255, 0, 0)
            label = "BARCODE: not decoded"

        draw.rectangle([c_min, r_min, c_max, r_max], outline=color, width=3)
        draw.text((c_min, max(0, r_min - 18)), label, fill=color)

    for item in decoded_qrs:
        r_min, c_min, r_max, c_max = item["box"]

        if item["decoded"]:
            color = (0, 0, 255)
            label = f"QR: {item['text']}"
        else:
            color = (255, 128, 0)
            label = "QR: not decoded"

        draw.rectangle([c_min, r_min, c_max, r_max], outline=color, width=3)
        draw.text((c_min, max(0, r_min - 18)), label, fill=color)

    return np.array(out)


def detect_and_decode_all_codes(image):
    _, barcode_candidates, barcode_debug = detect_barcodes(image)
    decoded_barcodes = decode_detected_barcodes(image, barcode_candidates, padding=12)

    qr_boxes, qr_debug = detect_qr_codes(image, return_debug=True)
    decoded_qrs = decode_detected_qrs(image, qr_boxes, padding=8)

    final_img = draw_unified_results(image, decoded_barcodes, decoded_qrs)

    debug = {
        "barcode_debug": barcode_debug,
        "qr_debug": qr_debug
    }

    return final_img, decoded_barcodes, decoded_qrs, debug
