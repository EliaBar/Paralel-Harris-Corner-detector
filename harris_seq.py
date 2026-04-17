import time
import numpy as np
from scipy.ndimage import maximum_filter

def grayscale_weighted(img_bgr):
    img_float = img_bgr.astype(np.float32)
    gray = (0.114 * img_float[:,:,0]) + (0.587 * img_float[:,:,1]) + (0.299 * img_float[:,:,2])
    return gray.astype(np.uint8)

def get_gaussian_kernel_1d(sigma):
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1
    x = np.linspace(-size//2, size//2, size)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_filter(image, sigma):
    kernel = get_gaussian_kernel_1d(sigma)
    pad = len(kernel)//2
    img = image.astype(np.float32)

    padded_h = np.pad(img, ((0,0),(pad,pad)), mode='edge')
    h_res = np.zeros_like(img)
    for i, w in enumerate(kernel):
        h_res += w * padded_h[:, i:i+img.shape[1]]

    padded_v = np.pad(h_res, ((pad,pad),(0,0)), mode='edge')
    v_res = np.zeros_like(img)
    for i, w in enumerate(kernel):
        v_res += w * padded_v[i:i+img.shape[0], :]
    return v_res

def sobel(img):
    dx = np.zeros_like(img)
    dx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    Ix = np.zeros_like(img)
    Ix[1:-1, :] = dx[:-2, :] + 2*dx[1:-1, :] + dx[2:, :]
    dy = np.zeros_like(img)
    dy[1:-1, :] = img[:-2, :] - img[2:, :]
    Iy = np.zeros_like(img)
    Iy[:, 1:-1] = dy[:, :-2] + 2*dy[:, 1:-1] + dy[:, 2:]
    return Ix, Iy

def compute_harris_response(Ix, Iy, params):
    Ixx = Ix*Ix
    Ixy = Ix*Iy
    Iyy = Iy*Iy
    A = gaussian_filter(Ixx, params.sigma_i)
    B = gaussian_filter(Ixy, params.sigma_i)
    C = gaussian_filter(Iyy, params.sigma_i)
    det = A*C - B**2
    trace = A + C
    return det - params.k * (trace**2)

def refine_subpixel_accuracy(R, corners):
    refined = []
    h, w = R.shape
    for y, x in corners:
        if 0 < y < h-1 and 0 < x < w-1:
            dx = (R[y, x+1] - R[y, x-1]) / 2
            dy = (R[y+1, x] - R[y-1, x]) / 2
            dxx = R[y, x+1] - 2*R[y, x] + R[y, x-1]
            dyy = R[y+1, x] - 2*R[y, x] + R[y-1, x]
            dxy = (R[y+1, x+1] + R[y-1, x-1] - R[y+1, x-1] - R[y-1, x+1]) / 4
            det = dxx*dyy - dxy**2
            if abs(det) > 1e-6:
                off_x = (dxy*dy - dyy*dx) / det
                off_y = (dxy*dx - dxx*dy) / det
                if abs(off_x) <= 1 and abs(off_y) <= 1:
                    refined.append((y+off_y, x+off_x))
                    continue
        refined.append((float(y), float(x)))
    return refined

def run_sequential(img_data, params):
    start = time.perf_counter()
    gray = grayscale_weighted(img_data)
    gray_smoothed = gaussian_filter(gray, params.sigma_d)
    Ix, Iy = sobel(gray_smoothed)
    R = compute_harris_response(Ix, Iy, params)
    R_max = np.max(R)
    threshold = R_max * params.threshold_ratio
    window_size = 2 * params.nms_radius + 1
    local_max = maximum_filter(R, size=window_size)
    mask = (R == local_max) & (R > threshold)
    raw_corners = np.argwhere(mask)
    corners = refine_subpixel_accuracy(R, raw_corners)
    end = time.perf_counter()
    return (end - start) * 1000, corners