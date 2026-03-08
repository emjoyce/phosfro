from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf


def load_warm_fn_bases(pickle_path = "/Users/emilyjoyce/repos/phosfro/phosfro/warm_start/tiled_warm_start_base_fns.pkl"):
    with open(pickle_path, "rb") as file:
        fn_bases = pickle.load(file)
    return fn_bases

# load all the mosaic responses, img responses
def load_warm_mosaic_and_imgs(pickle_path = "/Users/emilyjoyce/repos/phosfro/phosfro/warm_start/tiled_warm_start_base_fns.pkl",
                            img_dir = '/Users/emilyjoyce/repos/chromopho/chromopho/imgs/ece_done',
                            mosaic_dir = '/Users/emilyjoyce/repos/chromopho/chromopho/imgs/ece_feats'):
    fn_bases = load_warm_fn_bases(pickle_path)
    mosaic_responses = []
    imgs = []
    for base in fn_bases:
        # print(base)
        mosaic_pth = os.path.join(mosaic_dir, f"{base}_mosaic_output.npy")
        img_pth = os.path.join(img_dir, f"{base}.png")
        # print(f"Loading mosaic from {mosaic_pth} and image from {img_pth}")
        mosaic_response = np.load(mosaic_pth)
        # change to flat tf with just valid coords 
        valid_mask = (mosaic_response != -1)
        valid_coords = np.argwhere(valid_mask).astype(np.int32)
        valid_coords_tf = tf.convert_to_tensor(valid_coords, dtype=tf.int32)
        mosaic_response = tf.gather_nd(mosaic_response, valid_coords_tf)  
        img = plt.imread(img_pth)
        mosaic_responses.append(mosaic_response)
        imgs.append(img)
    return mosaic_responses, imgs

def find_warm_start(target_mosaic_response, warm_mosaic = None, warm_imgs = None,
                        return_idx = False):
    target_mosaic_response = tf.cast(target_mosaic_response, dtype=tf.float32)
    if warm_mosaic is None or warm_imgs is None:
        warm_mosaic, warm_imgs = load_warm_mosaic_and_imgs()
    best_idx = -1
    best_dist = float('inf')
    for idx, mosaic_response in enumerate(warm_mosaic):
        dist = tf.reduce_sum(tf.square(mosaic_response - target_mosaic_response))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
            # print(best_dist)
    # now pull that img
    img = warm_imgs[best_idx]
    print(f"best distance: {best_dist.numpy():.6f}")
    if return_idx:
        return img, best_idx
    return img 


def find_warm_stimuli_at_distance(
    target_mosaic_response,
    distance: float,
    n: int,
    warm_mosaic = None,
    warm_imgs = None,
    exclude_image = None,
    return_idx = False
):
    """Return up to ``n`` warm-start stimuli whose mosaic responses are
    approximately a given Euclidean distance away from ``target_mosaic_response``.

    Distances are computed in mosaic-response space (L2 norm between the
    flattened valid-coordinate responses). The ``n`` entries whose distance is
    closest to ``distance`` are returned.

    Parameters
    ----------
    target_mosaic_response:
        Mosaic response for the target (tf or numpy). Will be flattened.
    distance : float
        Desired L2 distance in mosaic space.
    n : int
        Number of warm-start stimuli to return (maximum).
    warm_mosaic, warm_imgs:
        Optional pre-loaded outputs from ``load_warm_mosaic_and_imgs``.
        If either is None, they are loaded inside this function.
    exclude_image:
        Optional starting image to exclude from the returned set if it is
        present in ``warm_imgs``.

    Returns
    -------
    selected_mosaics : List[tf.Tensor]
        The selected mosaic responses.
    selected_images : List[np.ndarray]
        The corresponding images.
    selected_distances : List[float]
        The actual L2 distances to ``target_mosaic_response``.
    selected_indices : List[int]
        Optionally returned based on return_idx. Indices into the warm-start dataset (same order as
        load_warm_mosaic_and_imgs / fn_bases).
    """
    target = tf.cast(tf.reshape(target_mosaic_response, [-1]), tf.float32)

    if warm_mosaic is None or warm_imgs is None:
        warm_mosaic, warm_imgs = load_warm_mosaic_and_imgs()

    # Optionally find index of image to exclude (if it exists in warm_imgs).
    exclude_idx = None
    if exclude_image is not None:
        exclude_np = np.asarray(exclude_image)
        for i, img in enumerate(warm_imgs):
            if np.array_equal(np.asarray(img), exclude_np):
                exclude_idx = i
                break

    candidates = []  # (idx, dist, |dist - target_distance|)
    for idx, mosaic in enumerate(warm_mosaic):
        if exclude_idx is not None and idx == exclude_idx:
            continue

        m = tf.cast(tf.reshape(mosaic, [-1]), tf.float32)
        d = float(tf.linalg.norm(m - target).numpy())
        diff = abs(d - float(distance))
        candidates.append((idx, d, diff))

    # Sort by closeness of distance to the requested value and take up to n.
    candidates.sort(key=lambda x: x[2])
    n = int(n)
    if n < 1:
        return [], [], []

    selected = candidates[: min(n, len(candidates))]
    selected_indices = [c[0] for c in selected]
    selected_distances = [c[1] for c in selected]

    selected_mosaics = [warm_mosaic[i] for i in selected_indices]
    selected_images = [warm_imgs[i] for i in selected_indices]

    if return_idx:
        return selected_mosaics, selected_images, selected_distances, selected_indices
    return selected_mosaics, selected_images, selected_distances


def find_warm_stimuli_from_start_image(
    start_image,
    target_mosaic_response,
    distance: float,
    n: int,
    warm_mosaic = None,
    warm_imgs = None,
):
    """Convenience wrapper that takes a starting image and returns
    ``n`` warm-start stimuli whose mosaic responses are approximately
    ``distance`` away from the target.

    The starting image is excluded from the candidate set if it appears in
    the warm-start dataset.
    """
    return find_warm_stimuli_at_distance(
        target_mosaic_response=target_mosaic_response,
        distance=distance,
        n=n,
        warm_mosaic=warm_mosaic,
        warm_imgs=warm_imgs,
        exclude_image=start_image,
    )


