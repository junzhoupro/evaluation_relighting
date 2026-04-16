# -------------------------------
# Metric helpers (replace / add only these parts)
# -------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Try spatial LPIPS first. If unsupported by your lpips version, fall back automatically.
try:
    LPIPS_MODEL = lpips.LPIPS(net="alex", spatial=True).to(DEVICE).eval()
    LPIPS_SPATIAL = True
except TypeError:
    LPIPS_MODEL = lpips.LPIPS(net="alex").to(DEVICE).eval()
    LPIPS_SPATIAL = False


def masked_psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """
    Strict masked RGB PSNR:
    only masked RGB values are used, and MSE is averaged over valid channels.
    """
    valid = mask[..., None].astype(bool)
    vals1 = img1[valid]
    vals2 = img2[valid]
    if vals1.size == 0:
        return float("nan")
    mse = np.mean((vals1 - vals2) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def masked_ssim(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute SSIM map on crop, then average only masked pixels.
    This is the standard practical masked-SSIM approach.
    """
    h, w = mask.shape
    min_side = min(h, w)
    if min_side < 3:
        return float("nan")

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


def masked_lpips(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """
    Prefer spatial LPIPS map + masked averaging when available.
    Otherwise fall back to your original bbox-crop + zero-background approximation.
    """
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # [0,1] -> [-1,1]
    t1 = t1 * 2.0 - 1.0
    t2 = t2 * 2.0 - 1.0

    with torch.no_grad():
        if LPIPS_SPATIAL:
            score_map = LPIPS_MODEL(t1, t2)
            # Usually [1,1,H,W]
            if score_map.ndim == 4:
                score_map = score_map[0, 0]
            elif score_map.ndim == 3:
                score_map = score_map[0]
            score_map = score_map.detach().cpu().numpy().astype(np.float32)

            valid = mask.astype(bool)
            if valid.sum() == 0:
                return float("nan")
            return float(score_map[valid].mean())
        else:
            # fallback: zero out outside mask
            mask3 = mask[..., None].astype(np.float32)
            img1m = img1 * mask3
            img2m = img2 * mask3

            tt1 = torch.from_numpy(img1m).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            tt2 = torch.from_numpy(img2m).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            tt1 = tt1 * 2.0 - 1.0
            tt2 = tt2 * 2.0 - 1.0

            score = LPIPS_MODEL(tt1, tt2)
            return float(score.item())


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] image to grayscale [0,1]."""
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def hist_chi2_distance(vals1: np.ndarray, vals2: np.ndarray, bins: int = 64) -> float:
    """
    Chi-square distance between two normalized 1D histograms.
    """
    if vals1.size == 0 or vals2.size == 0:
        return float("nan")

    h1, _ = np.histogram(vals1, bins=bins, range=(0.0, 1.0), density=False)
    h2, _ = np.histogram(vals2, bins=bins, range=(0.0, 1.0), density=False)

    h1 = h1.astype(np.float64)
    h2 = h2.astype(np.float64)

    h1 /= max(h1.sum(), 1e-12)
    h2 /= max(h2.sum(), 1e-12)

    eps = 1e-12
    return float(0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)))


def d_fg_fg(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, bins: int = 64) -> float:
    """
    Foreground-foreground histogram chi-square distance:
    grayscale histogram distance between pred foreground and GT foreground
    over the same mask region.
    """
    pred_gray = rgb_to_gray(pred)
    gt_gray = rgb_to_gray(gt)

    pred_vals = pred_gray[mask]
    gt_vals = gt_gray[mask]

    return hist_chi2_distance(pred_vals, gt_vals, bins=bins)


def compute_metrics(pred_path: str, gt_path: str, mask_path: str) -> Dict[str, float]:
    # Keep your original file reading logic.
    pred_img_pil = Image.open(pred_path).convert("RGB")
    target_size = pred_img_pil.size

    pred = np.asarray(pred_img_pil).astype(np.float32) / 255.0
    gt = load_rgb(gt_path, target_size=target_size)
    mask = load_mask(mask_path, target_size=target_size)

    # Keep your original bbox crop logic.
    pred_crop, mask_crop = crop_with_mask(pred, mask, pad=8)
    gt_crop, _ = crop_with_mask(gt, mask, pad=8)

    return {
        "masked_lpips": masked_lpips(pred_crop, gt_crop, mask_crop),
        "masked_ssim": masked_ssim(pred_crop, gt_crop, mask_crop),
        "masked_psnr": masked_psnr(pred_crop, gt_crop, mask_crop),
        "d_fg_fg": d_fg_fg(pred_crop, gt_crop, mask_crop, bins=64),
        "mask_pixels": int(mask.sum()),
        "crop_h": int(mask_crop.shape[0]),
        "crop_w": int(mask_crop.shape[1]),
    }