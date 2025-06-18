import cv2
import numpy as np
import pandas as pd
import math
from PIL import Image
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# Liste globale d’images à afficher
image_log = []

def log_image(title, img, cmap=None, xlabel=None, ylabel=None, show_ticks=False):
    image_log.append((title, img, cmap, xlabel, ylabel, show_ticks))

def get_scale(image: np.ndarray) -> float:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds = cv2.bitwise_not(th)
    contours, _ = cv2.findContours(thresholds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w / h > 5:
            return 100 / w

def show_all_images():
    n = len(image_log)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, entry in enumerate(image_log):
        # Compatibilité : supporte tuples à 3, 4, 5 ou 6 éléments
        title = entry[0]
        img = entry[1]
        cmap = entry[2] if len(entry) > 2 else None
        xlabel = entry[3] if len(entry) > 3 else None
        ylabel = entry[4] if len(entry) > 4 else None
        show_ticks = entry[5] if len(entry) > 5 else False

        ax = axes[i]
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or 'gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if show_ticks:
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    # Masquer les sous-graphiques inutilisés
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def get_theoretical_length_1(image):
    #log_image("Original image", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #log_image("Greyscale image", gray)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #log_image("Thresholded image", th)

    inv = cv2.bitwise_not(th)
    #log_image("Reverse thresholded image", inv)

    contours, _ = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxc = max(contours, key=cv2.contourArea)

    contour_img = cv2.cvtColor(inv.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
    #log_image("Inverted threshold with contours", contour_img)

    scale = get_scale(image)

    total = sum(np.linalg.norm(maxc[i] - maxc[i + 1]) for i in range(len(maxc) - 1))
    return total * scale / 2

def get_theoretical_length_2(image: np.ndarray, center=(93, 100), csv_path='data/pixels_par_rayon.csv') -> float:
    def recup_frame(img_src):
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 229, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        result = cv2.bitwise_and(img_src, img_src, mask=mask_opened)
        #log_image("Extraction of the frame", result)
        return result

    image_copy = image.copy()
    cv2.circle(image_copy, (center[0], center[1]), radius=1, color=(0, 0, 255), thickness=-1)  # BGR: rouge
    cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    log_image("Original image with centre in evidence", cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY))

    scale = get_scale(image)
    
    difference = cv2.subtract(image, recup_frame(image))

    specific_color = np.array([187, 175, 165], dtype=np.uint8)
    black_mask = np.all(difference == [0, 0, 0], axis=-1)
    image[black_mask] = specific_color
    #log_image("Original image without frame", image)

   
    def uncoilling(image: np.ndarray, center, specific_color) -> np.ndarray:
        h, w = image.shape[:2]
        maxl = int(math.hypot(w, h))
        unrolled = np.full((360, maxl, 3), fill_value=specific_color, dtype=np.uint8)

        # 2. Masque binaire à 45°
        m_diag = np.zeros((h, w), np.uint8)
        angle_diag = 45
        r_diag = math.radians(angle_diag)
        ex_diag = int(center[0] + maxl * math.cos(r_diag))
        ey_diag = int(center[1] - maxl * math.sin(r_diag))
        cv2.line(m_diag, center, (ex_diag, ey_diag), color=255, thickness=1)
        #log_image("2. Binary mask for angle 45°", m_diag, cmap="gray")

        # 4–7. Extraction à 0°, 90°, 180°, 270° (affichage horizontal)
        selected_angles = [0, 90, 180, 270]
        for ang in range(360):
            m = np.zeros((h, w), np.uint8)
            r = math.radians(ang)
            ex = int(center[0] + maxl * math.cos(r))
            ey = int(center[1] - maxl * math.sin(r))
            cv2.line(m, center, (ex, ey), color=255, thickness=1)

            ys, xs = np.where(m)
            dists = (xs - center[0])**2 + (ys - center[1])**2
            sorted_indices = np.argsort(dists)
            for xp, idx in enumerate(sorted_indices[:maxl]):
                x, y = xs[idx], ys[idx]
                unrolled[ang, xp] = image[y, x]

            if ang in selected_angles:
                line_image = unrolled[ang][:, np.newaxis, :]     # shape (L, 1, 3)
                line_image_h = np.transpose(line_image, (1, 0, 2))  # shape (1, L, 3)
                #log_image(f"{4 + selected_angles.index(ang)}. Unrolled line at {ang}°", line_image_h)

        # 8. Image finale
        #log_image("8. Final unrolled image", unrolled)

        return unrolled


    unrolled = uncoilling(image, center, specific_color)
    log_image(
        "Original image without frame and unrolled",
        unrolled,
        xlabel="Radial distance R (pixels)",
        ylabel="Angular position θ (degrees)",
        show_ticks=True
    )

    gi = cv2.cvtColor(unrolled, cv2.COLOR_BGR2GRAY)
    _, bin1 = cv2.threshold(gi, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    clean1 = cv2.morphologyEx(bin1, cv2.MORPH_CLOSE, kernel)
    sk1 = (skeletonize(clean1 // 255).astype(np.uint8) * 255)
    log_image("Unrolled and skeletonised image", sk1)

    def fixing(sk1):
        dfpr = pd.read_csv(csv_path)
        def ppr(r): 
            rr = dfpr[dfpr['Rayon'] == r]
            return int(rr['Pixels Utilisés'].values[0]) if not rr.empty else None

        h1, w1 = sk1.shape[:2]
        new_h = ppr(w1)
        if new_h is None:
            return 0.0

        corr = np.full((new_h, w1), fill_value=0, dtype=sk1.dtype)

        # 1. Image d'origine (sk1)
        #log_image("1. Input image (sk1)", sk1)

        # Pour visualiser la taille cible ppr(x) pour certaines colonnes sélectionnées
        selected_cols = np.linspace(0, w1 - 1, min(6, w1), dtype=int)  # max 6 cols pour log, pour total max 8 images

        for x in range(w1):
            col = sk1[:, x:x+1]
            
            nh = ppr(x)
            if nh is None: 
                continue
            rs = cv2.resize(col, (1, nh), interpolation=cv2.INTER_NEAREST)
            ys = (new_h - nh) // 2
            corr[ys:ys+nh, x] = rs[:, 0]

            # Log images pour colonnes sélectionnées seulement
            if x in selected_cols:
                # Visualiser la colonne redimensionnée (centrée verticalement dans une image pleine hauteur)
                vis_col = np.zeros((new_h, 1), dtype=sk1.dtype)
                vis_col[ys:ys+nh, 0] = rs[:, 0]
                # Étendre en largeur pour visualisation plus claire (ex: facteur 10)
                vis_col_wide = np.repeat(vis_col, 10, axis=1)
                #log_image(f"Resized column {x}", vis_col_wide, cmap='gray')

        # 2. Image corrigée finale (corr)
        log_image("Corrected image", corr, cmap='gray')

        wp_corr = np.sum(corr == 255)
        return wp_corr

    
    wp_corr = fixing(sk1)
    
    real_corr = wp_corr * scale
    return real_corr

# === Paramètres ===
image_path = "data/example.jpg"
real_length = 0
center_x, center_y = 93, 99

# === Traitement ===
image = np.array(Image.open(image_path).convert("RGB"))
t1 = get_theoretical_length_1(image.copy())
t2 = get_theoretical_length_2(image.copy(), center=(center_x, center_y))

pct_diff_1 = abs(t1 - real_length) / real_length * 100 if real_length != 0 else None
pct_diff_2 = abs(t2 - real_length) / real_length * 100 if real_length != 0 else None

print("Résultats pour l'image :")
print(f"Chemin          : {image_path}")
print(f"Longueur réelle : {real_length:.2f}")
print(f"Théorique M1    : {t1:.2f}")
print(f"Théorique M2    : {t2}")
print(f"% Diff M1       : {pct_diff_1:.2f}" if pct_diff_1 is not None else "Non calculé")
print(f"% Diff M2       : {pct_diff_2:.2f}" if pct_diff_2 is not None else "Non calculé")

# === Affichage final ===
show_all_images()
