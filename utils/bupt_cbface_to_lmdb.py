"""
bupt_cbface_to_lmdb.py
======================
Converts BUPT-CBFace-12 dataset to the LMDB format used by this training
framework, applying ArcFace-style 5-point facial alignment to every image.

Dataset structure expected:
  <dataset_dir>/
      images/
          m.XXXXXXX/        <- identity folder
              0.jpg
              1.jpg
              ...
      landmark.tsv          <- tab-separated, one row per image
          columns: NAME X1 Y1 X2 Y2 PTX1 PTY1 PTX2 PTY2 PTX3 PTY3 PTX4 PTY4 PTX5 PTY5

Landmark columns meaning (standard 5-point ArcFace order):
    PT1 = left eye
    PT2 = right eye
    PT3 = nose tip
    PT4 = left mouth corner
    PT5 = right mouth corner

Output LMDB schema (identical to existing training LMDB):
    b"__len__"      -> msgpack int   (total images stored)
    b"__keys__"     -> msgpack list  (list of byte keys)
    b"__classnum__" -> msgpack int   (number of unique classes)
    <key>           -> msgpack [jpg_bytes, int_label]

Usage:
    python utils/bupt_cbface_to_lmdb.py \\
        --dataset_dir "/path/to/BUPT-CBFace-12" \\
        --destination ./dataset \\
        --file_name bupt_cbface \\
        --min_images 2 \\
        --workers 8
"""

import argparse
import csv
import io
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import makedirs, path

import cv2
import lmdb
import msgpack
import numpy as np
from PIL import Image
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ArcFace canonical 5-point landmark positions for 112x112 output
# ---------------------------------------------------------------------------
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def get_similarity_transform(src_pts, dst_pts):
    """
    Compute a 2x3 similarity transform matrix mapping src_pts -> dst_pts.
    Uses least-squares solution for scale/rotation/translation.
    """
    num = src_pts.shape[0]
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)

    src_c = src_pts - src_mean
    dst_c = dst_pts - dst_mean

    src_var = (src_c ** 2).sum() / num

    W = (dst_c.T @ src_c) / num

    U, S, Vt = np.linalg.svd(W)
    d = np.linalg.det(U) * np.linalg.det(Vt)
    diag = np.diag([1.0, d])

    T = (U @ diag @ Vt)
    scale = S.dot([1.0, d]) / src_var

    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = scale * T
    M[:, 2] = dst_mean - scale * T @ src_mean
    return M


def align_face(img_bgr, landmarks_5pt, output_size=112):
    """
    Warp `img_bgr` so that the 5 facial landmarks snap to ArcFace canonical
    positions. Returns a (output_size x output_size) RGB uint8 ndarray.
    """
    src = landmarks_5pt.astype(np.float32)
    dst = ARCFACE_DST.copy()

    M = get_similarity_transform(src, dst)
    aligned = cv2.warpAffine(
        img_bgr, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Worker: load one image, align, encode to JPEG bytes
# ---------------------------------------------------------------------------
def _process_one(args):
    """
    args = (img_path, pts5, output_size)
    Returns (jpg_bytes, success_bool, error_msg)
    """
    img_path, pts5, output_size = args
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, False, f"Cannot read: {img_path}"
        rgb = align_face(img, pts5, output_size)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), True, ""
    except Exception as e:
        return None, False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Convert BUPT-CBFace-12 to the training LMDB format."
    )
    p.add_argument(
        "--dataset_dir", required=True,
        help="Root directory of BUPT-CBFace-12 (contains images/ and landmark.tsv)"
    )
    p.add_argument(
        "--destination", default="./dataset",
        help="Output directory for the LMDB file (default: ./dataset)"
    )
    p.add_argument(
        "--file_name", default="bupt_cbface",
        help="LMDB file name without extension (default: bupt_cbface)"
    )
    p.add_argument(
        "--min_images", type=int, default=2,
        help="Minimum images per identity to include (default: 2)"
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel alignment workers (default: 4)"
    )
    p.add_argument(
        "--output_size", type=int, default=112,
        help="Output aligned face image size in pixels (default: 112)"
    )
    p.add_argument(
        "--lmdb_map_size_gb", type=int, default=50,
        help="LMDB map size in GB (default: 50)"
    )
    return p.parse_args()


def load_landmarks(landmark_tsv):
    """
    Parse landmark.tsv into a dict:
        "m.XXXXXXX/0"  -> np.array shape (5, 2)  [PTX1/PTY1 ... PTX5/PTY5]

    The NAME column uses forward slash as a separator between identity and index.
    """
    logger.info(f"Loading landmarks from: {landmark_tsv}")
    landmarks = {}
    with open(landmark_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row["NAME"]  # e.g. "m.051vglx/0"
            pts = np.array([
                [float(row["PTX1"]), float(row["PTY1"])],
                [float(row["PTX2"]), float(row["PTY2"])],
                [float(row["PTX3"]), float(row["PTY3"])],
                [float(row["PTX4"]), float(row["PTY4"])],
                [float(row["PTX5"]), float(row["PTY5"])],
            ], dtype=np.float32)
            landmarks[name] = pts
    logger.info(f"Loaded {len(landmarks):,} landmark entries")
    return landmarks


def collect_samples(images_dir, landmarks, min_images):
    """
    Walk images/ directory, filter identities by min_images, and build a flat
    list of (img_path, pts5, int_label).
    """
    logger.info("Scanning identity folders ...")
    identity_map = {}  # identity_str -> [img_path, ...]

    for identity in sorted(os.listdir(images_dir)):
        identity_path = os.path.join(images_dir, identity)
        if not os.path.isdir(identity_path):
            continue
        imgs_with_lm = []
        for fname in sorted(os.listdir(identity_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]          # "0", "1", ...
            lm_key = f"{identity}/{stem}"              # "m.XXXXXXX/0"
            if lm_key not in landmarks:
                continue                               # skip if no landmark
            imgs_with_lm.append((
                os.path.join(identity_path, fname),
                landmarks[lm_key]
            ))
        if len(imgs_with_lm) >= min_images:
            identity_map[identity] = imgs_with_lm

    logger.info(f"Identities after min_images={min_images} filter: {len(identity_map):,}")

    samples = []  # [(img_path, pts5, label_int), ...]
    for label_int, (identity, img_pts_list) in enumerate(sorted(identity_map.items())):
        for img_path, pts5 in img_pts_list:
            samples.append((img_path, pts5, label_int))

    logger.info(f"Total samples to process: {len(samples):,}")
    return samples, len(identity_map)


def write_lmdb(samples, n_classes, lmdb_path, workers, output_size, map_size_gb):
    """
    Align all images in parallel, write them to LMDB in batches.
    """
    makedirs(path.dirname(lmdb_path), exist_ok=True)

    map_size = map_size_gb * 1024 ** 3
    env = lmdb.open(lmdb_path, map_size=map_size)

    keys = []
    n_success = 0
    n_fail = 0

    # Build args list for pool
    pool_args = [(img_path, pts5, output_size) for img_path, pts5, _ in samples]
    labels = [label for _, _, label in samples]

    BATCH = 1000  # write to LMDB in batches to avoid large transactions

    logger.info(f"Starting alignment with {workers} workers ...")
    start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_one, arg): idx
            for idx, arg in enumerate(pool_args)
        }
        buffer = {}  # idx -> (jpg_bytes, label)

        with tqdm(total=len(samples), desc="Aligning & encoding") as pbar:
            for fut in as_completed(futures):
                idx = futures[fut]
                jpg_bytes, ok, err = fut.result()
                if ok:
                    buffer[idx] = (jpg_bytes, labels[idx])
                    n_success += 1
                else:
                    n_fail += 1
                    if n_fail <= 20:
                        logger.warning(f"Skipped [{idx}]: {err}")
                pbar.update(1)

                # Flush buffer periodically
                if len(buffer) >= BATCH:
                    _flush_buffer(env, buffer, keys)
                    buffer.clear()

        # Flush remaining
        if buffer:
            _flush_buffer(env, buffer, keys)
            buffer.clear()

    # Write metadata
    with env.begin(write=True) as txn:
        txn.put(b"__len__", msgpack.dumps(n_success))
        txn.put(b"__keys__", msgpack.dumps(keys))
        txn.put(b"__classnum__", msgpack.dumps(n_classes))

    elapsed = time.time() - start
    logger.info(
        f"Done in {elapsed/60:.1f} min — "
        f"Written: {n_success:,}  Skipped: {n_fail:,}  Classes: {n_classes:,}"
    )
    logger.info(f"LMDB saved to: {lmdb_path}")


def _flush_buffer(env, buffer, keys_list):
    """Write a batch of (jpg_bytes, label) to LMDB and extend keys_list."""
    with env.begin(write=True) as txn:
        for idx, (jpg_bytes, label) in buffer.items():
            key = f"{idx:08d}".encode("ascii")
            value = msgpack.dumps([jpg_bytes, label])
            txn.put(key, value)
            keys_list.append(key)


def main():
    args = parse_args()

    images_dir = os.path.join(args.dataset_dir, "images")
    landmark_tsv = os.path.join(args.dataset_dir, "landmark.tsv")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images/ folder not found at: {images_dir}")
    if not os.path.isfile(landmark_tsv):
        raise FileNotFoundError(f"landmark.tsv not found at: {landmark_tsv}")

    lmdb_path = os.path.join(args.destination, f"{args.file_name}.lmdb")

    # Step 1: load landmarks
    landmarks = load_landmarks(landmark_tsv)

    # Step 2: collect samples
    samples, n_classes = collect_samples(images_dir, landmarks, args.min_images)

    if len(samples) == 0:
        logger.error("No samples found! Check --dataset_dir path and landmark.tsv.")
        return

    # Step 3: write LMDB
    write_lmdb(
        samples, n_classes, lmdb_path,
        workers=args.workers,
        output_size=args.output_size,
        map_size_gb=args.lmdb_map_size_gb,
    )


if __name__ == "__main__":
    main()
