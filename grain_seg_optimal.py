import os
import glob
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# ================== CONFIG GLOBALE ==================

# Dossier racine du dataset (avec sous-dossiers "images" et "labels")
DATASET_DIR = "test"   # <-- à adapter si besoin

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")
os.makedirs(LABEL_DIR, exist_ok=True)

# Extensions d'images acceptées
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Liste des images
image_paths = [
    p for p in glob.glob(os.path.join(IMAGE_DIR, "*.*"))
    if os.path.splitext(p)[1].lower() in IMG_EXTS
]
if not image_paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

print(f"Found {len(image_paths)} images in {IMAGE_DIR}")


# ================== FONCTIONS DE BASE ==================

def preprocess_gray(gray, bg_ksize, blur_k, invert):
    """
    Pré-traitement unique :
      - correction de fond (medianBlur)
      - normalisation (soustraction)
      - lissage (GaussianBlur)
      - inversion éventuelle

    Retourne :
      - val : image prétraitée (uint8) utilisée pour le multi-threshold
    """
    if bg_ksize % 2 == 0:
        bg_ksize += 1
    if blur_k % 2 == 0:
        blur_k += 1
    if bg_ksize < 3:
        bg_ksize = 3
    if blur_k < 1:
        blur_k = 1

    bg = cv2.medianBlur(gray, bg_ksize)
    gray_norm = cv2.subtract(gray, bg)
    gray_blur = cv2.GaussianBlur(gray_norm, (blur_k, blur_k), 0)

    if invert:
        val = 255 - gray_blur
    else:
        val = gray_blur

    return val


def segment_multi_threshold_from_val(val,
                                     t_min, t_max, n_steps, min_hits,
                                     min_area, max_area):
    """
    Pipeline multi-threshold à partir d'une image prétraitée 'val'.

    Retourne :
      - mask_clean : masque binaire final (uint8 0/255)
      - stats      : stats de connectedComponentsWithStats
      - centroids  : centroïdes des composantes
      - kept_count : nombre de composantes gardées (après filtrage area)
      - stability  : carte de stabilité (0..n_steps)
    """
    # Sécurité sur les paramètres
    if t_min < 0:
        t_min = 0
    if t_max > 255:
        t_max = 255
    if t_min >= t_max:
        t_max = min(255, t_min + 1)

    if n_steps < 1:
        n_steps = 1
    if min_hits < 1:
        min_hits = 1
    if min_hits > n_steps:
        min_hits = n_steps

    # 1) Multi-threshold
    thresholds = np.linspace(t_min, t_max, n_steps).astype(np.uint8)
    stability = np.zeros_like(val, dtype=np.uint16)

    for t in thresholds:
        # foreground si val > t
        _, mask_t = cv2.threshold(val, int(t), 1, cv2.THRESH_BINARY)
        stability += mask_t.astype(np.uint16)

    # Pixels gardés si "foreground" au moins min_hits fois
    mask_multi = np.zeros_like(val, dtype=np.uint8)
    mask_multi[stability >= min_hits] = 255

    # 2) Morphologie pour nettoyer
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_multi, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3) Composantes connexes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_clean, connectivity=8
    )

    # 4) Filtrage par aire
    kept_count = 0
    for i in range(1, num_labels):  # 0 = background
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        kept_count += 1

    return mask_clean, stats, centroids, kept_count, stability


def optimize_tmin_tmax_for_image_from_val(
    val,
    n_steps,
    min_hits,
    min_area,
    max_area,
    tmin_range=range(0, 40, 4),
    tmax_range=range(40, 120, 4),
):
    """
    Optimise t_min / t_max à partir d'une image prétraitée 'val'
    en maximisant le nombre de bboxes valides (après filtrage par aire).

    min_area / max_area sont des aires en pixels.

    Retourne :
      - best_tmin
      - best_tmax
      - best_count (nombre de composantes gardées)
    """
    best_count = -1
    best_tmin = None
    best_tmax = None

    for t_min in tmin_range:
        for t_max in tmax_range:
            # On impose un minimum d'écart pour éviter les cas absurdes
            if t_max <= t_min + 5:
                continue

            _, _, _, kept_count, _ = segment_multi_threshold_from_val(
                val,
                t_min=t_min, t_max=t_max,
                n_steps=n_steps, min_hits=min_hits,
                min_area=min_area, max_area=max_area
            )

            if kept_count > best_count:
                best_count = kept_count
                best_tmin = t_min
                best_tmax = t_max

    return best_tmin, best_tmax, best_count


def save_yolo_labels_from_stats(stats, image_shape, min_area, max_area, label_path):
    """
    Sauvegarde un fichier YOLO à partir des stats de connected components.

    Format YOLO :
      class x_center_norm y_center_norm w_norm h_norm
    avec class = 0 (single class).
    """
    H, W = image_shape

    with open(label_path, "w") as f:
        for i in range(1, stats.shape[0]):  # 0 = background
            x, y, w, h, area = stats[i]
            if area < min_area or area > max_area:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0

            x_norm = cx / W
            y_norm = cy / H
            w_norm = w / W
            h_norm = h / H

            f.write(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")


# ================== FONCTION POUR UNE IMAGE (PARALLÈLE) ==================

def process_one_image(
    img_path,
    n_steps,
    min_hits,
    bg_ksize,
    blur_k,
    invert,
    min_area_ratio,
    max_area_ratio,
    tmin_range,
    tmax_range,
):
    """
    Traite une image :
      - pré-traitement (une seule fois)
      - optimisation t_min / t_max sur cette image
      - segmentation finale
      - export YOLO
    Retourne une chaîne de log.
    """
    img_name = os.path.basename(img_path)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return f"⚠️ Skipping {img_name}: cannot read."

    H, W = gray.shape[:2]
    img_area = H * W

    # min/max area en pixels pour cette image
    min_area = int(img_area * min_area_ratio)
    max_area = int(img_area * max_area_ratio)
    if min_area < 1:
        min_area = 1
    if max_area <= min_area:
        max_area = min_area + 1

    # Pré-traitement unique
    val = preprocess_gray(gray, bg_ksize, blur_k, invert)

    # Optimisation t_min / t_max pour cette image
    best_tmin, best_tmax, best_count = optimize_tmin_tmax_for_image_from_val(
        val,
        n_steps=n_steps,
        min_hits=min_hits,
        min_area=min_area,
        max_area=max_area,
        tmin_range=tmin_range,
        tmax_range=tmax_range,
    )

    if best_tmin is None:
        return f"⚠️ {img_name}: no valid (t_min, t_max) found."

    # Segmentation finale avec ces paramètres (sur la même 'val')
    mask_clean, stats, centroids, kept_count, stability = segment_multi_threshold_from_val(
        val,
        t_min=best_tmin, t_max=best_tmax,
        n_steps=n_steps, min_hits=min_hits,
        min_area=min_area, max_area=max_area
    )

    # Enregistrement YOLO
    stem, _ = os.path.splitext(img_name)
    label_path = os.path.join(LABEL_DIR, stem + ".txt")
    save_yolo_labels_from_stats(stats, (H, W), min_area, max_area, label_path)

    return (
        f"✅ {img_name}: best_tmin={best_tmin}, best_tmax={best_tmax}, "
        f"kept_count={kept_count}, labels -> {label_path}"
    )


def process_one_image_wrapper(args):
    """Wrapper pour pouvoir utiliser executor.map sans lambda (picklable)."""
    return process_one_image(*args)


# ================== PIPELINE BATCH COMPLET (PARALLÈLE) ==================

def optimize_and_label_all_images(
    n_steps=20,
    min_hits=4,
    bg_ksize=31,
    blur_k=3,
    invert=False,
    # proportions de la surface de l'image (0..1)
    min_area_ratio=0.0005,
    max_area_ratio=0.001,
    # plages plus raisonnables par défaut (à ajuster selon ton dataset)
    tmin_range=range(0, 40, 4),
    tmax_range=range(40, 120, 4),
    num_workers=None,  # None => nb de coeurs CPU
):
    """
    Pour chaque image dans IMAGE_DIR :
      1) calcule min_area / max_area en fonction de la taille de l'image,
      2) optimise t_min / t_max (en maximisant le nb de bboxes valides),
      3) recalcule la segmentation avec ces valeurs,
      4) sauvegarde les labels YOLO dans LABEL_DIR.

    Ce traitement est réalisé en parallèle sur plusieurs processus.
    """

    print("🚀 Batch optimization + labeling started (parallel)...")
    print(f"  n_steps={n_steps}, min_hits={min_hits}, bg_ksize={bg_ksize}, "
          f"blur_k={blur_k}, invert={invert}")
    print(f"  min_area_ratio={min_area_ratio}, max_area_ratio={max_area_ratio}")
    print(f"  tmin_range={tmin_range.start}-{tmin_range.stop} step={tmin_range.step}")
    print(f"  tmax_range={tmax_range.start}-{tmax_range.stop} step={tmax_range.step}")
    print(f"  Images to process: {len(image_paths)}")
    print(f"  num_workers={num_workers or 'auto'}\n")

    # Préparation des arguments communs
    worker_args = []
    for img_path in image_paths:
        worker_args.append(
            (
                img_path,
                n_steps,
                min_hits,
                bg_ksize,
                blur_k,
                invert,
                min_area_ratio,
                max_area_ratio,
                tmin_range,
                tmax_range,
            )
        )

    # Exécution parallèle (sans lambda)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for msg in executor.map(process_one_image_wrapper, worker_args):
            print(msg)

    print("\n🏁 Batch done.")


# ================== MAIN (si exécuté comme script) ==================

if __name__ == "__main__":
    # Tu peux adapter les paramètres ici si besoin :
    optimize_and_label_all_images(
        n_steps=20,          
        min_hits=4,
        bg_ksize=31,
        blur_k=3,
        invert=False,       # True si les grains sont plus sombres que le fond
        min_area_ratio=0.00009,
        max_area_ratio=0.0008,
        tmin_range=range(10, 40, 1),
        tmax_range=range(50, 120, 1),
        num_workers=None,   # None => utilise tous les coeurs dispo
    )
