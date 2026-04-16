import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
import lpips

# lpips_mode = "spatial"

# -------------------------------
# User config
# -------------------------------
# hand_name = "aged-hand_choose"
# hand_name = "aged-hand_choose_rotate"
# hand_name = "young-hand"
# hand_name = "young-hand_rotate"
# hand_name = "light-skin-hand"
# hand_name = "light-skin-hand_rotate"
hand_name = "dark-skin-hand"
# hand_name = "dark-skin-hand_rotate"


# hand_name = "blackgirl_newnecklaces"
# hand_name = "whitegirl_newnecklaces"

his_dir = "./his2"          # histogram adjustment dir, ring
# his_dir = "./his3"          # histogram adjustment dir, necklace
restore_dir = "./restore2"  # diffusion result dir
folder = "./hdrs_chosen"    # folder containing .exr files

# Output CSV. One row per sample_id. Re-running the script updates existing rows.
out_csv = f"./metric_results/metric_results_{hand_name}_test.csv"

# Jewelry whitelist
# jeweleries = [
#     "circletriangle_gold",
#     "circletriangle_silver",
#     "medallion_gold",
#     "medallion_silver",
#     "circletriangle_pendulum1_gold",
#     "circletriangle_pendulum1_silver",
#     "circletriangle_pendulum2_gold",
#     "circletriangle_pendulum2_silver",
#     "circletriangle_pendulum3_gold",
#     "circletriangle_pendulum3_silver",
#     "circletriangle_pendulum4_gold",
#     "circletriangle_pendulum4_silver",
#     "circletriangle_pendulum5_gold",
#     "circletriangle_pendulum5_silver",
#     "medallion_pendulum1_gold",
#     "medallion_pendulum1_silver",
#     "medallion_pendulum2_gold",
#     "medallion_pendulum2_silver",
#     "medallion_pendulum3_gold",
#     "medallion_pendulum3_silver",
#     "medallion_pendulum4_gold",
#     "medallion_pendulum4_silver",
#     "medallion_pendulum5_gold",
#     "medallion_pendulum5_silver",
# ]

#black girl
# jeweleries = [
# "circletriangle_gold",
# "circletriangle_silver",
# "medallion_gold",
# "medallion_silver",
# "curve_pendulum1_gold",
# "curve_pendulum1_silver",
# "curve_pendulum2_gold",
# "curve_pendulum2_silver",
# "curve_pendulum3_gold",
# "curve_pendulum3_silver",
# "curve_pendulum4_gold",
# "curve_pendulum4_silver",
# "curve_pendulum5_gold",
# "curve_pendulum5_silver",
# "pendulum1_gold",
# "pendulum1_silver",
# "pendulum2_gold",
# "pendulum2_silver",
# "pendulum3_gold",
# "pendulum3_silver",
# "pendulum4_gold",
# "pendulum4_silver",
# "pendulum5_gold",
# "pendulum5_silver"
# ]

jeweleries = [
"ch-gladysiewski_gold", 
"ch-gladysiewski_silver", 
"silver-moon-ring_gold",
"silver-moon-ring_silver",
"riccardoferrari-ring_gold",
"riccardoferrari-ring_silver",
"golden-ring-test_gold",
"golden-ring-test_silver"
]

# -------------------------------
# Metric helpers
# -------------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# LPIPS_MODEL = lpips.LPIPS(net="alex").to(DEVICE).eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    LPIPS_MODEL = lpips.LPIPS(net="alex", spatial=True).to(DEVICE).eval()
    LPIPS_SPATIAL = True
except TypeError:

    LPIPS_MODEL = lpips.LPIPS(net="alex").to(DEVICE).eval()
    LPIPS_SPATIAL = False


def load_rgb(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load image as float32 RGB in [0, 1]."""
    img = Image.open(path).convert("RGB")
    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr



def load_mask(path: str, target_size: Optional[Tuple[int, int]] = None, threshold: int = 127) -> np.ndarray:
    """Load binary mask as bool array."""
    m = Image.open(path).convert("L")
    if target_size is not None and m.size != target_size:
        m = m.resize(target_size, Image.NEAREST)
    arr = np.asarray(m)
    return arr > threshold



def masked_bbox(mask: np.ndarray, pad: int = 8) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask is empty.")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = mask.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return x0, y0, x1 + 1, y1 + 1  # python slice end-exclusive



def crop_with_mask(img: np.ndarray, mask: np.ndarray, pad: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = masked_bbox(mask, pad=pad)
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]



def masked_psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    mask3 = mask[..., None].astype(np.float32)
    diff2 = ((img1 - img2) ** 2) * mask3
    n_pixels = mask3.sum()
    if n_pixels == 0:
        return float("nan")
    # Average over masked pixels and 3 channels
    mse = diff2.sum() / (n_pixels * 3)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)



def masked_ssim(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    # Compute SSIM map on crop and average only masked pixels.
    h, w = mask.shape
    min_side = min(h, w)
    if min_side < 3:
        return float("nan")

    # skimage requires an odd win_size <= min_side.
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3
    if win_size > min_side:
        win_size = min_side if min_side % 2 == 1 else min_side - 1
    if win_size < 3:
        return float("nan")

    _, ssim_map = ssim(
        img1,
        img2,
        channel_axis=2,
        data_range=1.0,
        full=True,
        win_size=win_size,
        gaussian_weights=True,
    )
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return float("nan")
    return float(ssim_map[valid].mean())



# def masked_lpips(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
#     # LPIPS is not natively masked. We use a common proxy:
#     # crop to mask bbox and zero out background outside the mask.
#     mask3 = mask[..., None].astype(np.float32)
#     img1m = img1 * mask3
#     img2m = img2 * mask3

#     t1 = torch.from_numpy(img1m).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
#     t2 = torch.from_numpy(img2m).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

#     # [0,1] -> [-1,1]
#     t1 = t1 * 2.0 - 1.0
#     t2 = t2 * 2.0 - 1.0

#     with torch.no_grad():
#         score = LPIPS_MODEL(t1, t2)
#     return float(score.item())

# def masked_lpips(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
#     """
#     Stricter masked LPIPS:
#     - if LPIPS spatial map is available, average only inside the resized mask;
#     - otherwise fall back to the zero-background approximation.
#     """
#     t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
#     t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

#     # [0,1] -> [-1,1]
#     t1 = t1 * 2.0 - 1.0
#     t2 = t2 * 2.0 - 1.0

#     with torch.no_grad():
#         if LPIPS_SPATIAL:
#             score_map = LPIPS_MODEL(t1, t2)

#             # Usually shape [1,1,H,W]
#             if score_map.ndim == 4:
#                 score_map_np = score_map[0, 0].detach().cpu().numpy().astype(np.float32)
#             elif score_map.ndim == 3:
#                 score_map_np = score_map[0].detach().cpu().numpy().astype(np.float32)
#             else:
#                 return float(score_map.item())

#             # Resize mask to score-map resolution
#             mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
#             mask_resized = mask_img.resize(
#                 (score_map_np.shape[1], score_map_np.shape[0]),
#                 Image.NEAREST
#             )
#             mask_resized = np.asarray(mask_resized) > 127

#             if mask_resized.sum() == 0:
#                 return float("nan")
#             return float(score_map_np[mask_resized].mean())

#         else:
#             # fallback: zero background outside mask
#             mask3 = mask[..., None].astype(np.float32)
#             img1m = img1 * mask3
#             img2m = img2 * mask3

#             tt1 = torch.from_numpy(img1m).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
#             tt2 = torch.from_numpy(img2m).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

#             tt1 = tt1 * 2.0 - 1.0
#             tt2 = tt2 * 2.0 - 1.0

#             score = LPIPS_MODEL(tt1, tt2)
#             return float(score.item())

def masked_lpips(pred_crop, gt_crop, mask_crop, model):
    pred_t = torch.from_numpy(pred_crop).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) * 2 - 1
    gt_t   = torch.from_numpy(gt_crop).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) * 2 - 1

    try:
        # spatial LPIPS path
        score_map = model(pred_t, gt_t)   # [1,1,h,w] if spatial=True works

        if score_map.ndim == 0:
            return float(score_map.item()), "scalar_direct"

        score_map = score_map.squeeze().detach().cpu().numpy()

        if score_map.ndim == 0:
            return float(score_map.item()), "scalar_direct"

        h_s, w_s = score_map.shape[-2], score_map.shape[-1]
        if mask_crop.shape != (h_s, w_s):
            mask_crop_rs = cv2.resize(
                mask_crop.astype(np.uint8),
                (w_s, h_s),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_crop_rs = mask_crop.astype(bool)

        if mask_crop_rs.sum() == 0:
            return float(score_map.mean()), "spatial_empty_mask"

        return float(score_map[mask_crop_rs].mean()), "spatial"

    except Exception:
        # fallback scalar LPIPS on zero-background masked crop
        masked_pred = pred_crop.copy()
        masked_gt = gt_crop.copy()
        masked_pred[~mask_crop] = 0.0
        masked_gt[~mask_crop] = 0.0

        masked_pred_t = torch.from_numpy(masked_pred).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) * 2 - 1
        masked_gt_t   = torch.from_numpy(masked_gt).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) * 2 - 1

        score = model(masked_pred_t, masked_gt_t).item()
        return float(score), "fallback"
# --------------------------------------------------------------------------
# Histogram chi-squared distance (Tan et al., TAP 2015, Sections 3.5 & 5.3)
# --------------------------------------------------------------------------

def _intensity_histogram(pixels: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Compute a normalized intensity histogram from a 1-D array of [0,255] values."""
    hist, _ = np.histogram(pixels, bins=n_bins, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return hist.astype(np.float64)
    return hist.astype(np.float64) / total


def _chi2_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Symmetric chi-squared distance between two histograms (Eq. 2, Tan et al.)."""
    denom = h1 + h2
    valid = denom > 0
    if valid.sum() == 0:
        return 0.0
    # return float(np.sum((h1[valid] - h2[valid]) ** 2 / denom[valid]))
    return float(0.5 * np.sum((h1[valid] - h2[valid]) ** 2 / denom[valid]))


def _make_boundary_mask(fg_mask: np.ndarray, border_frac: float = 0.05) -> np.ndarray:
    """
    Create a boundary mask: a ring of pixels surrounding the foreground.
    border_frac: border width as a fraction of image width (5% = Tan et al.).
    """
    import cv2
    h, w = fg_mask.shape
    border_px = max(1, int(border_frac * w))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border_px + 1, 2 * border_px + 1))
    dilated = cv2.dilate(fg_mask.astype(np.uint8), kernel)
    boundary = (dilated > 0) & (~fg_mask.astype(bool))
    return boundary


def hist_chi2_fg_boundary(composite_path: str, mask_path: str,
                          target_size: Optional[Tuple[int, int]] = None,
                          border_frac: float = 0.05) -> float:
    """
    D_fg-boundary: chi2 distance between the intensity histogram of the
    foreground (jewelry) region and a surrounding boundary ring in the
    *composite* image.  Measures lighting inconsistency between the
    jewelry and its immediate surroundings (Tan et al. 2015, Table I).
    """
    comp_pil = Image.open(composite_path).convert("RGB")
    if target_size is None:
        target_size = comp_pil.size
    elif comp_pil.size != target_size:
        comp_pil = comp_pil.resize(target_size, Image.BILINEAR)

    gray = np.asarray(comp_pil.convert("L"))                # uint8, shape (H, W)
    mask = load_mask(mask_path, target_size=target_size)     # bool, shape (H, W)

    boundary = _make_boundary_mask(mask, border_frac=border_frac)

    fg_pixels = gray[mask]
    bd_pixels = gray[boundary]

    if len(fg_pixels) == 0 or len(bd_pixels) == 0:
        return float("nan")

    return _chi2_distance(
        _intensity_histogram(fg_pixels),
        _intensity_histogram(bd_pixels),
    )


def hist_chi2_fg_fg(composite_path: str, gt_path: str, mask_path: str,
                    target_size: Optional[Tuple[int, int]] = None) -> float:
    """
    D_fg-fg: chi2 distance between the intensity histogram of the
    foreground region in the composite and the same region in the
    ground-truth rendering.  Measures how much the jewelry appearance
    has changed from the reference (Tan et al. 2015, Table I).
    """
    comp_pil = Image.open(composite_path).convert("RGB")
    if target_size is None:
        target_size = comp_pil.size
    elif comp_pil.size != target_size:
        comp_pil = comp_pil.resize(target_size, Image.BILINEAR)

    gt_pil = Image.open(gt_path).convert("RGB")
    if gt_pil.size != target_size:
        gt_pil = gt_pil.resize(target_size, Image.BILINEAR)

    gray_comp = np.asarray(comp_pil.convert("L"))
    gray_gt   = np.asarray(gt_pil.convert("L"))
    mask = load_mask(mask_path, target_size=target_size)

    fg_comp = gray_comp[mask]
    fg_gt   = gray_gt[mask]

    if len(fg_comp) == 0 or len(fg_gt) == 0:
        return float("nan")

    return _chi2_distance(
        _intensity_histogram(fg_comp),
        _intensity_histogram(fg_gt),
    )


def compute_metrics(pred_path: str, gt_path: str, mask_path: str) -> Dict[str, float]:
    # Use prediction size as reference.
    pred_img_pil = Image.open(pred_path).convert("RGB")
    target_size = pred_img_pil.size

    pred = np.asarray(pred_img_pil).astype(np.float32) / 255.0
    gt = load_rgb(gt_path, target_size=target_size)
    mask = load_mask(mask_path, target_size=target_size)

    pred_crop, mask_crop = crop_with_mask(pred, mask, pad=8)
    gt_crop, _ = crop_with_mask(gt, mask, pad=8)

    # --- existing metrics (pred vs gt, foreground only) ---
    # m_lpips = masked_lpips(pred_crop, gt_crop, mask_crop)
    m_lpips, lpips_mode = masked_lpips(pred_crop, gt_crop, mask_crop, LPIPS_MODEL)
    m_ssim  = masked_ssim(pred_crop, gt_crop, mask_crop)
    m_psnr  = masked_psnr(pred_crop, gt_crop, mask_crop)

    # --- histogram chi2 distances (Tan et al. 2015) ---
    d_fg_boundary = hist_chi2_fg_boundary(pred_path, mask_path,
                                          target_size=target_size,
                                          border_frac=0.05)
    d_fg_fg       = hist_chi2_fg_fg(pred_path, gt_path, mask_path,
                                    target_size=target_size)

    return {
        "masked_lpips":    m_lpips,
        "masked_ssim":     m_ssim,
        "masked_psnr":     m_psnr,
        "hist_chi2_fg_boundary": d_fg_boundary,
        "hist_chi2_fg_fg":       d_fg_fg,
        "mask_pixels": int(mask.sum()),
        "crop_h": int(mask_crop.shape[0]),
        "crop_w": int(mask_crop.shape[1]),
        "lpips_mode": lpips_mode,
    }


# -------------------------------
# File helpers
# -------------------------------

def find_diffusion_and_mask(diffusion_img_path: str) -> Tuple[Optional[str], Optional[str]]:
    pngs = [f for f in os.listdir(diffusion_img_path) if f.lower().endswith(".png")]
    mask_candidates = [f for f in pngs if "mask" in f.lower()]
    chosen_candidates = [f for f in pngs if "mask" not in f.lower()]

    # Also allow non-png masks if needed.
    if not mask_candidates:
        others = [f for f in os.listdir(diffusion_img_path) if "mask" in f.lower()]
        mask_candidates.extend(others)

    chosen_path = os.path.join(diffusion_img_path, chosen_candidates[0]) if chosen_candidates else None
    mask_path = os.path.join(diffusion_img_path, mask_candidates[0]) if mask_candidates else None
    return chosen_path, mask_path



def upsert_rows(csv_path: str, rows: List[Dict]) -> None:
    new_df = pd.DataFrame(rows)
    if new_df.empty:
        print("No rows to write.")
        return

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        if "sample_id" not in old_df.columns:
            raise ValueError(f"Existing CSV {csv_path} does not contain a 'sample_id' column.")
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["sample_id"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(by=["hand_name", "hdr_name", "jewelry_name", "method"])
    combined.to_csv(csv_path, index=False)
    print(f"Saved {len(new_df)} rows (upsert) to {csv_path}. Total rows now: {len(combined)}")


# -------------------------------
# Main loop
# -------------------------------

def main() -> None:
    rows: List[Dict] = []

    for file in os.listdir(folder):
        if not file.lower().endswith(".exr"):
            continue

        hdr_name = os.path.splitext(file)[0]
        diffusion_img_folder = os.path.join(restore_dir, hand_name, hdr_name)
        if not os.path.isdir(diffusion_img_folder):
            print(f"[Skip] Missing diffusion folder: {diffusion_img_folder}")
            continue

        print(f"hdr_name = {hdr_name}\n==================")

        for jewelry_name in os.listdir(diffusion_img_folder):
            if not any(sub in jewelry_name for sub in jeweleries):
                continue

            diffusion_img_path = os.path.join(diffusion_img_folder, jewelry_name)
            if not os.path.isdir(diffusion_img_path):
                continue

            chosen_png_path, mask_path = find_diffusion_and_mask(diffusion_img_path)
            hist_png_dir = os.path.join(his_dir, hand_name, hdr_name)
            hist_png_path = os.path.join(hist_png_dir, jewelry_name + "_hist.png")
            composite_png_path = os.path.join(hist_png_dir, jewelry_name + "_composite.png")
            gt_png_path = os.path.join(hist_png_dir, jewelry_name + "_gt.png")

            print("Chosen diffusion PNG:", chosen_png_path)
            print("hist path:", hist_png_path)
            print("composite path:", composite_png_path)
            print("gt path:", gt_png_path)
            print("mask path:", mask_path)
            print("~~~~~~~~~~~~~~~")

            required = {
                "diffusion": chosen_png_path,
                "hist": hist_png_path,
                "composite": composite_png_path,
                "gt": gt_png_path,
                "mask": mask_path,
            }
            missing = [k for k, v in required.items() if not v or not os.path.exists(v)]
            if missing:
                print(f"[Skip] Missing files for {jewelry_name} under {hdr_name}: {missing}")
                continue

            for method_name, pred_path in [
                ("diffusion", chosen_png_path),
                ("hist", hist_png_path),
                ("composite", composite_png_path),
            ]:
                try:
                    metrics = compute_metrics(pred_path, gt_png_path, mask_path)
                    sample_id = f"{hand_name}|{hdr_name}|{jewelry_name}|{method_name}"
                    row = {
                        "sample_id": sample_id,
                        "hand_name": hand_name,
                        "hdr_name": hdr_name,
                        "jewelry_name": jewelry_name,
                        "method": method_name,
                        "pred_path": pred_path,
                        "gt_path": gt_png_path,
                        "mask_path": mask_path,
                        **metrics,
                    }
                    rows.append(row)
                    print(
                        f"[{method_name}] LPIPS={metrics['masked_lpips']:.4f}, "
                        f"SSIM={metrics['masked_ssim']:.4f}, "
                        f"PSNR={metrics['masked_psnr']:.2f}, "
                        f"D_fg_bd={metrics['hist_chi2_fg_boundary']:.4f}, "
                        f"D_fg_fg={metrics['hist_chi2_fg_fg']:.4f}"
                    )
                except Exception as e:
                    print(f"[Error] {hand_name} / {hdr_name} / {jewelry_name} / {method_name}: {e}")

    upsert_rows(out_csv, rows)


if __name__ == "__main__":
    main()
