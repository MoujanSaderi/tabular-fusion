import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate
from scipy.ndimage import shift
from scipy.ndimage import zoom
from scipy.signal.windows import tukey
from src.utils.data_enums import SeriesType
import random

from numpy.fft import fft2, ifft2, fftshift, ifftshift


def reshape_slice(slice_data: np.ndarray, series_type, zero_pad=True):
    """Crops slice to the specified size.

    args:
        slice_data: numpy array containing the series volume, of shape
            (length, width)
        series_type: src.utils.data_enums.SeriesType enum
        zero_pad (bool): whether to zero pad the slice

    returns:
        numpy array containing the reshaped volume, of shape (length, width)
    """
    # center crop the slice
    slice_shape = (slice_data.shape[0], slice_data.shape[1])
    crop_size = series_type.value["crop_size"]

    # crop or pad height
    if slice_shape[0] > crop_size[0]:
        start = (slice_shape[0] - crop_size[0]) // 2
        slice_data = slice_data[start : start + crop_size[0], :]
    elif zero_pad and slice_shape[0] < crop_size[0]:
        diff_h = crop_size[0] - slice_shape[0]
        pad_top = diff_h // 2
        pad_bottom = diff_h - pad_top
        slice_data = np.pad(slice_data, ((pad_top, pad_bottom), (0, 0)))

    # crop or pad width
    if slice_data.shape[1] > crop_size[1]:
        start = (slice_data.shape[1] - crop_size[1]) // 2
        slice_data = slice_data[:, start : start + crop_size[1]]
    elif zero_pad and slice_data.shape[1] < crop_size[1]:
        diff_w = crop_size[1] - slice_data.shape[1]
        pad_left = diff_w // 2
        pad_right = diff_w - pad_left
        slice_data = np.pad(slice_data, ((0, 0), (pad_left, pad_right)))

    return slice_data


def upsample_and_crop(vol: np.ndarray, crop_size: tuple) -> np.ndarray:
    """
    Upsamples the volume by a factor of 2 and crops to the specified field of view (FOV).

    args:
        vol: numpy array containing the volume, of shape (num_slices, length, width)
        crop_size: tuple indicating the (height, width) of the desired FOV

    returns:
        numpy array containing the upsampled and cropped volume
    """
    # Upsample by a factor of 2
    vol_upsampled = zoom(vol, (1.0, 2.0, 2.0), order=1)

    # Crop to the specified FOV
    center_h, center_w = vol_upsampled.shape[1] // 2, vol_upsampled.shape[2] // 2
    crop_h, crop_w = crop_size
    start_h = max(center_h - crop_h // 2, 0)
    start_w = max(center_w - crop_w // 2, 0)

    cropped_vol = vol_upsampled[
        :, start_h : start_h + crop_h, start_w : start_w + crop_w
    ]

    return cropped_vol


def reshape_volume(vol, series_type, zero_pad=True):
    """Crops and drops slices from the volume, and rotates it to the shape expected
    by the model input.

    args:
        vol: numpy array containing the series volume, of shape
            (length, width, num_slices)
        series_type: src.utils.data_enums.SeriesType enum
        zero_pad (bool): whether to zero pad the volume

    returns:
        numpy array containing the reshaped volume, of shape (num_slices, length, width)
    """
    crop_size = series_type.value["crop_size"]
    num_slices = series_type.value["num_slices"]

    if zero_pad:  # zero pad the volume if it is smaller than the desired size
        # center crop (or pad) each slice
        center_cropped_vol = np.zeros((crop_size[0], crop_size[1], vol.shape[-1]))
        for slice_idx in range(vol.shape[-1]):
            center_cropped_vol[:, :, slice_idx] = reshape_slice(
                vol[:, :, slice_idx], series_type, zero_pad=True
            )

        # drop slices (or pad) to the desired number of slices
        if center_cropped_vol.shape[-1] < num_slices:
            slices_to_add = [
                (num_slices - center_cropped_vol.shape[-1]) // 2,
                num_slices
                - (
                    center_cropped_vol.shape[-1]
                    + ((num_slices - center_cropped_vol.shape[-1]) // 2)
                ),
            ]
            depth_adjusted_vol = np.pad(
                center_cropped_vol,
                (
                    (0, 0),
                    (0, 0),
                    (slices_to_add[0], slices_to_add[1]),
                ),
            )
        elif center_cropped_vol.shape[-1] > num_slices:
            slices_to_keep = (
                (center_cropped_vol.shape[-1] - num_slices) // 2,
                ((center_cropped_vol.shape[-1] - num_slices) // 2) + num_slices,
            )
            depth_adjusted_vol = center_cropped_vol[
                :,
                :,
                slices_to_keep[0] : slices_to_keep[1],
            ]
        else:
            depth_adjusted_vol = center_cropped_vol

    else:  # zoom to fit the volume if it is smaller than the desired size
        # center crop each slice
        center_cropped_vol = np.zeros(
            (
                np.min([vol.shape[0], crop_size[0]]),
                np.min([vol.shape[1], crop_size[1]]),
                vol.shape[-1],
            )
        )
        for slice_idx in range(vol.shape[-1]):
            center_cropped_vol[:, :, slice_idx] = reshape_slice(
                vol[:, :, slice_idx], series_type, zero_pad=False
            )

        # drop slices to the desired number of slices
        if center_cropped_vol.shape[-1] > num_slices:
            slices_to_keep = (
                (center_cropped_vol.shape[-1] - num_slices) // 2,
                ((center_cropped_vol.shape[-1] - num_slices) // 2) + num_slices,
            )
            depth_adjusted_vol = center_cropped_vol[
                :,
                :,
                slices_to_keep[0] : slices_to_keep[1],
            ]
        else:
            depth_adjusted_vol = center_cropped_vol

        # zoom to fit the desired size
        target_shape = (crop_size[0], crop_size[1], num_slices)
        scale_factors = np.array(target_shape) / np.array(depth_adjusted_vol.shape)
        depth_adjusted_vol = zoom(depth_adjusted_vol, scale_factors)

    # rotate volume to the shape expected by model input (num_slices, length, width)
    rotated_vol = np.moveaxis(depth_adjusted_vol, -1, 0)

    return rotated_vol


def scale_volume(vol, series_type):
    """Naive scale of the volume to the specified pixel range

    args:
        vol: numpy array containing the series volume, of shape
            (num_slices, length, width)
        series_type: src.utils.data_enums.SeriesType enum

    returns:
        numpy array containing the scaled volume, of shape (num_slices, length, width)
    """
    # scale pixel values to specified range
    pixel_range = series_type.value["pixel_range"]
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    vol = vol * (pixel_range[1] - pixel_range[0]) + pixel_range[0]

    return vol


def normalize_slice(exam_slice: np.ndarray, eps=1e-11) -> np.ndarray:
    """Normalize a single slice

    args:
        exam_slice: numpy array containing a slice, of shape (length, width)
        eps: small value to avoid division by zero

    returns:
        numpy array containing a normalized slice, of shape (length, width)
    """
    mean = exam_slice.mean().item()
    std = exam_slice.std().item()

    return (exam_slice - mean) / (std + eps)



def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Normalize a volume, just a wrapper for normalize_slice.

    args:
        vol: numpy array containing a volume, of shape (num_slices, length, width)

    returns:
        numpy array containing a normalized volume, of shape (num_slices, length, width)
    """
    return normalize_slice(vol)


def augment_slice(
    exam_slice: np.ndarray,
    flip_prob=0.5,
    rotate_prob=0.5,
    shear_prob=0.5,
    filter_prob=0.5,
) -> np.ndarray:
    """Augment a single slice

    args:
        exam_slice: numpy array containing a slice, of shape (length, width)
        flip_prob: probability of flipping the slice
        rotate_prob: probability of rotating the slice
        shear_prob: probability of shearing the slice
        filter_prob: probability of applying a gaussian filter to the slice

    returns:
        numpy array containing an augmented slice, of shape (length, width)
    """
    augmented_slice = exam_slice.copy()

    # flip
    if np.random.random() < flip_prob:
        augmented_slice = np.fliplr(augmented_slice)

    # rotate
    if np.random.random() < rotate_prob:
        angle = np.random.uniform(-180, 180)
        augmented_slice = rotate(augmented_slice, angle, reshape=False)

    # shear
    if np.random.random() < shear_prob:
        shear_range = np.random.uniform(-0.2, 0.2, size=2)
        augmented_slice = shift(augmented_slice, shear_range)

    # gaussian filter
    if np.random.random() < filter_prob:
        sigma = np.random.uniform(0.5, 2.0)
        augmented_slice = gaussian_filter(augmented_slice, sigma)

    return augmented_slice


def augment_slice_by_slice(vol: np.ndarray) -> np.ndarray:
    """Augment a volume slice by slice

    args:
        vol: numpy array containing a volume, of shape (num_slices, length, width)

    returns:
        numpy array containing an augmented volume, of shape (num_slices, length, width)
    """
    for slice_idx in range(vol.shape[0]):
        vol[slice_idx, :, :] = augment_slice(vol[slice_idx, :, :])
    return vol


def augment_volume(vol: np.ndarray, flip_prob=0.5, shear_prob=0.5) -> np.ndarray:
    """Augment a volume

    args:
        vol: numpy array containing a volume, of shape (num_slices, length, width)

    returns:
        numpy array containing an augmented volume, of shape (num_slices, length, width)
    """
    augmented_vol = vol.copy()
    # flip the volume
    if np.random.random() < flip_prob:
        for i in range(vol.shape[0]):
            augmented_vol[i, :, :] = np.fliplr(vol[i, :, :])

    # shear the volume
    vol = augmented_vol.copy()
    if np.random.random() < shear_prob:
        shear_range = np.random.uniform(-0.1, 0.1, size=2)
        for i in range(vol.shape[0]):
            augmented_vol[i, :, :] = shift(vol[i, :, :], shear_range)

    return augmented_vol


def add_adaptive_rician_noise(volume: np.ndarray, sigma=0.0) -> np.ndarray:
    """
    Add adaptive Rician noise to a normalized 3D volume.

    Args:
        volume (np.ndarray): Input 3D volume normalized to [0, 1] range.
        sigma (float): Optional stddev of noise. If None, randomly chosen from [0, 0.2].
        noise1, noise2 (np.ndarray): Optional precomputed noise fields.

    Returns:
        np.ndarray: Noisy 3D volume.
    """

    noise1 = np.random.normal(0, sigma, volume.shape)
    noise2 = np.random.normal(0, sigma, volume.shape)

    return np.sqrt((volume + noise1) ** 2 + noise2**2)


def random_downsample(vol: np.ndarray, factors: list[int]) -> np.ndarray:
    """
    Randomly downsample a 3D volume by a factor selected from the given list.

    Args:
        vol (np.ndarray): 3D input volume (num_slices, height, width)
        factors (list[int]): List of downsampling factors to choose from

    Returns:
        np.ndarray: Downsampled and upsampled volume of same shape as input
    """
    factor = random.choice(factors)

    if factor == 1:
        return vol  # no downsampling

    # Downsample the spatial dimensions only (keep slices axis unchanged)
    zoom_factors = (1.0, 1.0 / factor, 1.0 / factor)
    vol_down = zoom(vol, zoom_factors, order=1)

    # Upsample back to original shape
    zoom_back_factors = (
        1.0,
        vol.shape[1] / vol_down.shape[1],
        vol.shape[2] / vol_down.shape[2],
    )
    vol_restored = zoom(vol_down, zoom_back_factors, order=1)

    return vol_restored


def tukey_window_2d(h, w, alpha=0.3):
    """Generate a 2D Tukey window with tunable tapering."""
    window_h = tukey(h, alpha=alpha)
    window_w = tukey(w, alpha=alpha)
    return np.outer(window_h, window_w)


def pad_to_shape(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad a 2D array asymmetrically to the desired shape."""
    h, w = array.shape
    target_h, target_w = target_shape

    pad_h_total = target_h - h
    pad_w_total = target_w - w

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top

    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    return np.pad(
        array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant"
    )


def downsample_mri_kspace(
    img: np.ndarray, factor: int, tukey_alpha: float = 0.3
) -> np.ndarray:
    """
    Downsample an MRI image by cropping in k-space, applying a Tukey window
    to reduce Gibbs ringing, and zero-padding back to original size.

    Args:
        img (np.ndarray): 2D or 3D image (single slice or full volume).
        factor (int): Downsampling factor (e.g., 2 means keeping 1/2 resolution).
        tukey_alpha (float): Tapering parameter (0 = rectangular, 1 = Hann).

    Returns:
        np.ndarray: Image downsampled via k-space cropping with Tukey window.
    """
    if factor <= 1:
        return img  # No downsampling

    if img.ndim == 3:
        return np.stack(
            [downsample_mri_kspace(slice_, factor, tukey_alpha) for slice_ in img],
            axis=0,
        )

    # FFT to k-space
    kspace = fftshift(fft2(img))
    h, w = kspace.shape
    new_h, new_w = h // factor, w // factor

    center_h, center_w = h // 2, w // 2
    cropped = kspace[
        center_h - new_h // 2 : center_h + (new_h + 1) // 2,
        center_w - new_w // 2 : center_w + (new_w + 1) // 2,
    ]

    # Apply Tukey window
    window = tukey_window_2d(cropped.shape[0], cropped.shape[1], alpha=tukey_alpha)
    cropped *= window

    # Pad to original shape
    padded = pad_to_shape(cropped, (h, w))

    # Inverse FFT to image space
    img_down = np.abs(ifft2(ifftshift(padded)))

    return img_down




def preprocess_volume(
    vol: np.ndarray,
    series_type,
    normalize: bool = True,
    augment: str = "none",
    zero_pad: bool = True,
    sigma=None,
    noise_sigma_range=(0.0, 0.15),
    downsample_factors=None,
) -> np.ndarray:
    """Process a 3D volume for model input.

    The volume is reshaped and optionally augmented.  Augmentation can
    include Rician noise, k-space downsampling, or simple flipping/shearing.

    Args:
        vol: Input volume of shape ``(H, W, S)``.
        series_type: :class:`~src.utils.data_enums.SeriesType` describing the
            sequence.
        normalize: Whether to apply per-volume z-score normalization.
        augment: ``"none"``, ``"standard"``, ``"noise```, or ``"downsample"``.
        zero_pad: If ``True``, pad/crop to match ``series_type``.
        sigma: Optional fixed noise level for ``augment="noise"``.
        noise_sigma_range: Range of noise stddev when ``sigma`` is ``None``.
        downsample_factors: Possible factors for ``augment="downsample"``.

    Returns:
        np.ndarray: The processed volume.
    """
    vol = reshape_volume(vol, series_type, zero_pad=zero_pad)

    if augment != "none":
        if augment == "noise":
            vol_min = vol.min()
            vol_max = vol.max()
            vol = (vol - vol_min) / (vol_max - vol_min + 1e-6)
            if sigma is None:
                sigma = random.uniform(noise_sigma_range[0], noise_sigma_range[1])
            vol = add_adaptive_rician_noise(vol, sigma=sigma)
        elif augment == "downsample":
            if downsample_factors is None:
                downsample_factors = [1]
            factor = random.choice(downsample_factors)
            vol = downsample_mri_kspace(vol, factor)
        elif augment == "standard":
            vol = augment_volume(vol)
        else:
            raise ValueError(f"Unknown augmentation mode: {augment}")

    if normalize:
        vol = normalize_volume(vol)

    return vol


def preprocess_slice(
    slice_data: np.ndarray, series_type=SeriesType.AXT2, normalize=True, augment=False
) -> np.ndarray:
    """Process a single slice.

    args:
        slice_data: numpy array containing a slice, of shape (length, width)
        series_type: src.utils.data_enums.SeriesType enum
        normalize (bool): Indicates whether or not to normalize data
        augment (bool): Indicates whether or not to augment data

    returns:
        numpy array containing a processed slice, of shape (length, width)
    """
    # reshape slice
    slice_data = reshape_slice(slice_data, series_type)

    # normalize slice
    if normalize:
        slice_data = normalize_slice(slice_data)
    if augment:
        slice_data = augment_slice(
            slice_data, flip_prob=0.5, rotate_prob=0.5, shear_prob=0.0, filter_prob=0.0
        )

    return slice_data
