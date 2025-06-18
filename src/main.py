import csv, os, sys
from PIL import Image
import numpy as np
import cv2
from skimage.morphology import skeletonize
import math
import pandas as pd

base_path = "data"
levels = ["level_1", "level_2", "level_3"]
output_file = "outputs/output_table.csv"

def get_scale(image: np.ndarray) -> float:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds = cv2.bitwise_not(th)
    contours, _ = cv2.findContours(thresholds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w / h > 5:
            return 100 / w

def get_theoretical_length_1(image):
    scale = get_scale(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(th)
    contours, _ = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxc = max(contours, key=cv2.contourArea)
    total = sum(np.linalg.norm(maxc[i] - maxc[i + 1]) for i in range(len(maxc) - 1))
    return total * scale / 2

def get_theoretical_length_2(image: np.ndarray, center=(93, 100),
                           csv_path='data/pixels_par_rayon.csv') -> float:
    scale = get_scale(image)
    
    def recup_frame(img_src):
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 229, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        result = cv2.bitwise_and(img_src, img_src, mask=mask_opened)
        return result

    difference = cv2.subtract(image, recup_frame(image))
    specific_color = np.array([187, 175, 165], dtype=np.uint8)
    black_mask = np.all(difference == [0, 0, 0], axis=-1)
    image[black_mask] = specific_color

    h, w = image.shape[:2]
    maxl = int(math.hypot(w, h))
    unrolled = np.full((360, maxl, 3), fill_value=specific_color, dtype=np.uint8)

    for ang in range(360):
        m = np.zeros((h, w), np.uint8)
        r = math.radians(ang)
        ex = int(center[0] + maxl * math.cos(r))
        ey = int(center[1] - maxl * math.sin(r))
        cv2.line(m, center, (ex, ey), color=255, thickness=1)
        pts = [((x, y), tuple(image[y, x])) for y in range(h) for x in range(w) if m[y, x]]
        pts.sort(key=lambda p: (p[0][0] - center[0])**2 + (p[0][1] - center[1])**2)
        for xp, (_, colv) in enumerate(pts[:maxl]):
            unrolled[ang, xp] = colv

    gi = cv2.cvtColor(unrolled, cv2.COLOR_BGR2GRAY)
    _, bin1 = cv2.threshold(gi, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    clean1 = cv2.morphologyEx(bin1, cv2.MORPH_CLOSE, kernel)
    sk1 = (skeletonize(clean1 // 255).astype(np.uint8) * 255)

    dfpr = pd.read_csv(csv_path)
    def ppr(r): 
        rr = dfpr[dfpr['Rayon'] == r]
        return int(rr['Pixels Utilisés'].values[0]) if not rr.empty else None

    h1, w1 = sk1.shape[:2]
    new_h = ppr(w1)
    if new_h is None:
        return 0.0

    corr = np.full((new_h, w1), fill_value=0, dtype=sk1.dtype)
    for x in range(w1):
        col = sk1[:, x:x+1]
        nh = ppr(x)
        if nh is None: continue
        rs = cv2.resize(col, (1, nh), interpolation=cv2.INTER_NEAREST)
        ys = (new_h - nh) // 2
        corr[ys:ys+nh, x] = rs[:, 0]

    wp_corr = np.sum(corr == 255)
    
    real_corr = wp_corr * scale
    return real_corr


def find_file_level(filename):
    for lvl in levels:
        if os.path.exists(os.path.join(base_path, lvl, filename)):
            return lvl
    return None

with open(os.path.join(base_path, "ground_truth.csv"), newline='', encoding='utf-8') as infile:
    reader = list(csv.DictReader(infile, delimiter=','))  # convert to list for length
    total_rows = len(reader)

    with open(output_file, "w", newline='', encoding='utf-8') as outfile:
        fieldnames = ["filename", "winding_level", "realLength", "theoreticalLength_1", "theoreticalLength_2", "centerX", "centerY", "pctDiffM1", "pctDiffM2"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            f = row["filename"]
            level = find_file_level(f)
            if not level:
                continue
            try:
                path = os.path.join(base_path, level, f)
                image = np.array(Image.open(path).convert("RGB"))
                theoretical_len_1 = get_theoretical_length_1(image)
                theoretical_len_2 = get_theoretical_length_2(image)
                real_len = float(row["realLength"])
                pct_diff_1 = abs(theoretical_len_1 - real_len) / real_len * 100 if real_len != 0 else None
                pct_diff_2 = abs(theoretical_len_2 - real_len) / real_len * 100 if real_len != 0 else None

                writer.writerow({
                    "filename": f,
                    "winding_level": level,
                    "realLength": f"{real_len:.2f}",
                    "theoreticalLength_1": f"{theoretical_len_1:.2f}",
                    "theoreticalLength_2": f"{theoretical_len_2:.2f}",
                    "centerX": row["centerX"],
                    "centerY": row["centerY"],
                    "pctDiffM1": f"{pct_diff_1:.2f}" if pct_diff_1 is not None else "",
                    "pctDiffM2": f"{pct_diff_2:.2f}" if pct_diff_2 is not None else ""
                })
            except:
                continue

            # Progress bar
            bar_len = 50
            filled_len = int(bar_len * idx / total_rows)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            percent = 100.0 * idx / total_rows
            sys.stdout.write(f"\rProgress: |{bar}| {percent:6.2f}% ({idx}/{total_rows})")
            sys.stdout.flush()
