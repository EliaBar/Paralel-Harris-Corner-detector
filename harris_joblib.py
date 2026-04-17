import time
from joblib import Parallel, delayed
from harris_mp import compute_R_and_local_max, split_image_blocks, process_nms_block

def get_max_from_chunk(chunk):
    return max(chunk) if chunk else 0

def find_max_chunked_joblib(local_maxes, parallel, num_processes):
    n = len(local_maxes)
    if n <= num_processes:
        return max(local_maxes) if local_maxes else 0
    
    chunk_size = (n + num_processes - 1) // num_processes
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append(local_maxes[i:i + chunk_size])
    sub_maxes = parallel(delayed(get_max_from_chunk)(chunk) for chunk in chunks)
    return max(sub_maxes)

def run_parallel_joblib(img_data, params, parallel):
    start = time.perf_counter()
    img_blocks = split_image_blocks(img_data, params.block_size_joblib, params.overlap)
    results_R = parallel(
        delayed(compute_R_and_local_max)(block, params) 
        for block, _, _ in img_blocks
    )
    local_maxes = [res[1] for res in results_R]
    global_max_R = find_max_chunked_joblib(local_maxes, parallel, params.processes_jl)
    results_nms = parallel(
        delayed(process_nms_block)(results_R[i][0], y, x, global_max_R, params) 
        for i, (_, y, x) in enumerate(img_blocks)
    )

    all_corners = []
    for block_corners_list in results_nms:
        for corner in block_corners_list:
            all_corners.append(corner)
    
    end = time.perf_counter()
    return (end - start) * 1000, all_corners