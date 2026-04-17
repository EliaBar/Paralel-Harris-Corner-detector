import numpy as np
import time
from scipy.ndimage import maximum_filter
from harris_seq import (
    grayscale_weighted, gaussian_filter, sobel, 
    compute_harris_response, refine_subpixel_accuracy
)

def split_image_blocks(img, block_size, overlap):
    h, w = img.shape[:2]
    blocks = []
    for y in range(0, h, block_size - overlap):
        for x in range(0, w, block_size - overlap):
            y_end, x_end = min(y + block_size, h), min(x + block_size, w)
            blocks.append((img[y:y_end, x:x_end], y, x))
    return blocks

def compute_R_and_local_max(img_block, params):
    gray = grayscale_weighted(img_block)
    gray_smoothed = gaussian_filter(gray, params.sigma_d)
    Ix, Iy = sobel(gray_smoothed)
    R_block = compute_harris_response(Ix, Iy, params)
    return R_block, np.max(R_block)

def process_nms_block(R_block, offset_y, offset_x, global_max_R, params):
    threshold = global_max_R * params.threshold_ratio
    local_max = maximum_filter(R_block, size=2 * params.nms_radius + 1)
    mask = (R_block == local_max) & (R_block > threshold)
    corners = refine_subpixel_accuracy(R_block, np.argwhere(mask))
    return [(y + offset_y, x + offset_x) for y, x in corners]

def get_max_from_chunk(chunk):
    return max(chunk) if chunk else 0

def find_max_chunked_parallel(local_maxes, pool, num_processes):
    n = len(local_maxes)
    if n <= num_processes:
        return max(local_maxes) if local_maxes else 0
    chunk_size = (n + num_processes - 1) // num_processes
    chunks = [local_maxes[i:i + chunk_size] for i in range(0, n, chunk_size)]
    tasks = [pool.apply_async(get_max_from_chunk, (chunk,)) for chunk in chunks]
    sub_maxes = [t.get() for t in tasks]
    return max(sub_maxes)

def run_parallel_mp(img_data, params, pool):    
    start = time.perf_counter()
    blocks = split_image_blocks(img_data, params.block_size_mp, params.overlap)
    results_R = pool.starmap(compute_R_and_local_max, [(block, params) for block, _, _ in blocks])
    local_maxes = [res[1] for res in results_R]
    global_max_R = find_max_chunked_parallel(local_maxes, pool, params.processes_mp)

    args_nms = []
    for i in range(len(blocks)):
        R_block = results_R[i][0]
        offset_y = blocks[i][1]
        offset_x = blocks[i][2]
        args_nms.append((R_block, offset_y, offset_x, global_max_R, params))
    
    results_nms = pool.starmap(process_nms_block, args_nms)
    all_corners = []
    for block_corners_list in results_nms:
        for corner in block_corners_list:
            all_corners.append(corner)
    
    end = time.perf_counter()
    return (end - start) * 1000, all_corners