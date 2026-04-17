import os
import cv2
import time
import shutil
import tempfile
import numpy as np
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from PIL import Image, ImageSequence
from joblib import Parallel
from joblib.externals.loky import get_reusable_executor

from harris_seq import run_sequential
from harris_mp import run_parallel_mp
from harris_joblib import run_parallel_joblib

HarrisConfig = namedtuple('HarrisConfig', [
    'k', 
    'sigma_d', 
    'sigma_i', 
    'threshold_ratio', 
    'nms_radius',
    'IN_DIR',
    'OUT_DIR', 
    'iterations', 
    'processes_mp',
    'processes_jl',
    'block_size_mp',
    'block_size_joblib',
    'overlap'
])

CONFIG = HarrisConfig(
    k=0.04,
    sigma_d=1.5,
    sigma_i=1.5,
    threshold_ratio=0.01,
    nms_radius=int(2 * 1.5),
    IN_DIR="Harris_algorithm/test",
    OUT_DIR="Harris_algorithm/results",
    iterations=20,
    processes_mp=6,
    processes_jl=8,
    block_size_mp=256,
    block_size_joblib=768,
    overlap=15
    )

RUN_TEST_1_COMPARISON = False
RUN_TEST_2_SEQ_GRID = False
RUN_TEST_3_VALIDATION = False
RUN_TEST_4_MP_OPTIMIZATION = False
RUN_TEST_5_JOBLIB_OPTIMIZATION = True

def extract_gif_frames(file_path):
    frames = []
    gif = Image.open(file_path)
    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert('RGB')
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)
    return frames

def prepare_processing_environment(config):
    if not os.path.exists(config.OUT_DIR): os.makedirs(config.OUT_DIR)
    valid_ext = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.gif')
    return [f for f in os.listdir(config.IN_DIR) if f.lower().endswith(valid_ext)]

def load_all_data(config):
    if not os.path.exists(config.OUT_DIR):
        os.makedirs(config.OUT_DIR)
    
    valid_ext = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.gif')
    filenames = [f for f in os.listdir(config.IN_DIR) if f.lower().endswith(valid_ext)]
    
    all_units = []
    for name in filenames:
        path = os.path.join(config.IN_DIR, name)
        if name.lower().endswith('.gif'):
            frames = extract_gif_frames(path)
            for i, frame in enumerate(frames):
                all_units.append({'name': name, 'unit': f"frame_{i}", 'data': frame})
        else:
            img = cv2.imread(path)
            if img is not None:
                all_units.append({'name': name, 'unit': "standard", 'data': img})
    return all_units

def test_2_parameter_grid(target_unit):
    print("\n=== TEST 2: Parameter Impact Analysis ===")
    
    img = target_unit['data']
    unit_name = target_unit['unit']
    file_name = target_unit['name']
    
    print(f"Analyzing unit: {unit_name} from {file_name}")

    thresholds = np.arange(0.002, 0.022, 0.002)
    t_results = []
    
    for t in thresholds:
        temp_config = CONFIG._replace(threshold_ratio=t)
        _, corners = run_sequential(img, temp_config)
        count = len(corners)
        t_results.append(count)

    sigmas = np.arange(1.0, 2.6, 0.1)
    s_results = []
    
    for s in sigmas:
        new_radius = int(s * 2)
        temp_config = CONFIG._replace(sigma_d=s, sigma_i=s, nms_radius=new_radius)
        _, corners = run_sequential(img, temp_config)
        count = len(corners)
        s_results.append(count)

    table_data = []
    for i in range(len(thresholds)):
        table_data.append({"Parameter": "Threshold", "Value": round(thresholds[i], 4), "Corners": t_results[i]})
    for i in range(len(sigmas)):
        table_data.append({"Parameter": "Sigma", "Value": round(sigmas[i], 1), "Corners": s_results[i]})

    df_results = pd.DataFrame(table_data)
    print(df_results.to_string(index=False))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(thresholds, t_results, color='blue', linestyle='-', marker='o')
    plt.title("Impact of Threshold on Corner Count")
    plt.xlabel("Threshold Ratio")
    plt.ylabel("Number of Corners")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(sigmas, s_results, color='green', linestyle='-', marker='s')
    plt.title("Impact of Sigma on Corner Count")
    plt.xlabel("Sigma Value")
    plt.ylabel("Number of Corners")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_3_validation_matrix(target_unit, pool, parallel_obj):
    print("\n=== TEST 3: Parallel Validation Matrix ===")
    
    img = target_unit['data']
    
    _, corners_seq = run_sequential(img, CONFIG)
    
    seq_set = set()
    for y, x in corners_seq:
        seq_set.add((round(y, 1), round(x, 1)))

    test_methods = [
        ("Multiprocessing", run_parallel_mp, pool),
        ("Joblib", run_parallel_joblib, parallel_obj)
    ]

    final_stats = []

    for name, func, executor in test_methods:
        _, corners_par = func(img, CONFIG, executor)
        
        par_set = set()
        for y, x in corners_par:
            par_set.add((round(y, 1), round(x, 1)))

        tp = len(seq_set.intersection(par_set))
        fp = len(par_set.difference(seq_set))
        fn = len(seq_set.difference(par_set))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        final_stats.append({
            "Method": name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)
        })

    df_metrics = pd.DataFrame(final_stats)
    print(df_metrics.to_string(index=False))

    plot_df = df_metrics.set_index("Method")[["TP", "FP", "FN"]]
    plt.figure(figsize=(8, 4))
    sns.heatmap(plot_df, annot=True, cmap="YlGnBu", fmt="d")
    plt.title("Error Matrix: TP (Success), FP (Extra), FN (Missed)")
    plt.show()

def remove_outliers_iqr(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return filtered_df    

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    all_work_units = load_all_data(CONFIG)
    
    if not all_work_units:
        print("Зображень не знайдено!")
        exit()
    
    temp_dir = tempfile.mkdtemp()

    if RUN_TEST_1_COMPARISON:
        print(f"\n=== Test 1: Comparison of algorithms ===")
        raw_data = []
        for _ in range(len(all_work_units)):
            raw_data.append([0, 0, 0, None, None, None])

        for idx, unit in enumerate(all_work_units):
            img_data = unit['data']
            run_sequential(img_data, CONFIG)
            seq_times = []
            last_corners = None
            for _ in range(CONFIG.iterations):
                t_seq, c_seq = run_sequential(img_data, CONFIG)
                seq_times.append(t_seq)
                last_corners = c_seq
            raw_data[idx][0] = np.mean(seq_times)
            raw_data[idx][3] = last_corners

        with mp.Pool(processes=CONFIG.processes_mp) as pool:
            for idx, unit in enumerate(all_work_units):
                img_data = unit['data']
                run_parallel_mp(img_data, CONFIG, pool)
                mp_times = []
                last_corners = None
                for _ in range(CONFIG.iterations):
                    t_mp, c_mp = run_parallel_mp(img_data, CONFIG, pool)
                    mp_times.append(t_mp)
                    last_corners = c_mp
                raw_data[idx][1] = np.mean(mp_times)
                raw_data[idx][4] = last_corners

        temp_dir = tempfile.mkdtemp()
        try:
            with Parallel(n_jobs=CONFIG.processes_jl,
                          backend='loky',
                          max_nbytes=0,
                          temp_folder=temp_dir,
                          verbose=0,
                          batch_size=8,         
                          pre_dispatch='2*n_jobs'
                          ) as parallel:
                for idx, unit in enumerate(all_work_units):
                    img_data = unit['data']
                    run_parallel_joblib(img_data, CONFIG, parallel)
                    joblib_times = []
                    last_corners = None
                    for _ in range(CONFIG.iterations):
                        t_jl, c_jl = run_parallel_joblib(img_data, CONFIG, parallel)
                        joblib_times.append(t_jl)
                        last_corners = c_jl
                    raw_data[idx][2] = np.mean(joblib_times)
                    raw_data[idx][5] = last_corners
                get_reusable_executor().shutdown(wait=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        results = []
        for idx, unit in enumerate(all_work_units):
            name, unit_name, img_data = unit['name'], unit['unit'], unit['data']
            avg_seq, avg_mp, avg_jl, last_seq, last_mp, last_jl = raw_data[idx]
            acc_mp = avg_seq / avg_mp
            eff_mp = acc_mp / CONFIG.processes_mp
            acc_jl = avg_seq / avg_jl
            eff_jl = acc_jl / CONFIG.processes_jl
            results.append((f"{name}_{unit_name}", avg_seq, avg_mp, avg_jl, acc_mp, eff_mp, acc_jl, eff_jl, len(last_seq)))

        header = f"{'Source Unit':<45} | {'T_seq':>8} | {'T_mp':>8} | {'T_jl':>8} | {'S_mp':>5} | {'E_mp':>5} | {'S_jl':>5} | {'E_jl':>5}| {'Corners':>7}"
        sep = "-" * len(header)
        print("\n" + header + "\n" + sep)
        for r in results:
            print(f"{r[0]:<45} | {r[1]:8.2f} | {r[2]:8.2f} | {r[3]:8.2f} | {r[4]:5.2f} | {r[5]:5.2f} | {r[6]:5.2f} | {r[7]:5.2f}| {r[8]:7d}")
    
    if RUN_TEST_2_SEQ_GRID:
        test_2_parameter_grid(all_work_units[5])

    if RUN_TEST_3_VALIDATION:
        with mp.Pool(processes=CONFIG.processes_mp) as pool:
            parallel_joblib = Parallel(
                n_jobs=CONFIG.processes_jl,
                backend='loky',
                temp_folder=temp_dir,
                verbose=0,
                batch_size=8,
                pre_dispatch='2*n_jobs'
            )
            for i in range(0,8):
                test_3_validation_matrix(all_work_units[i], pool, parallel_joblib)

    if RUN_TEST_4_MP_OPTIMIZATION:
        print("\n=== TEST 4: Multiprocessing Optimization  ===")
        target_img = all_work_units[min(5, len(all_work_units)-1)]['data']
        mp_bench_data = []

        block_sizes = [128, 256, 384, 512, 640, 768, 896, 1024]
        with mp.Pool(processes=CONFIG.processes_mp) as pool:
            run_parallel_mp(target_img, CONFIG, pool) 
            for size in block_sizes:
                t_cfg = CONFIG._replace(block_size_mp=size)
                for _ in range(CONFIG.iterations):
                    dur, _ = run_parallel_mp(target_img, t_cfg, pool)
                    mp_bench_data.append({"Type": "Block Size", "Parameter": size, "Time": dur})

        process_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        for p in process_counts:
            t_cfg = CONFIG._replace(processes_mp=p)
            with mp.Pool(processes=p) as pool:
                run_parallel_mp(target_img, t_cfg, pool) 
                for _ in range(CONFIG.iterations):
                    dur, _ = run_parallel_mp(target_img, t_cfg, pool)
                    mp_bench_data.append({"Type": "Processes", "Parameter": p, "Time": dur})


        df_mp = pd.DataFrame(mp_bench_data)
        cleaned_mp_list = []
        for (t_type, param), group in df_mp.groupby(['Type', 'Parameter']):
            cleaned_mp_list.append(remove_outliers_iqr(group, "Time"))
        df_mp_clean = pd.concat(cleaned_mp_list)

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df_mp_clean[df_mp_clean["Type"] == "Block Size"], x="Parameter", y="Time", palette="pastel")
        plt.title("Multiprocessing: Block Size Impact")
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df_mp_clean[df_mp_clean["Type"] == "Processes"], x="Parameter", y="Time", palette="flare")
        plt.title("Multiprocessing: Process Count Impact")
        plt.tight_layout()
        plt.show()

        print("\nMultiprocessing Stats:")
        print(df_mp_clean.groupby(['Type', 'Parameter'])['Time'].agg(['mean', 'std', 'median']).reset_index())

    if RUN_TEST_5_JOBLIB_OPTIMIZATION:
        print("\n=== TEST 5: Joblib Architecture Optimization ===")
        target_img = all_work_units[min(5, len(all_work_units)-1)]['data']
        jl_bench_data = []

        for size in [128, 256, 384, 512, 640, 768, 896, 1024]:
            t_cfg = CONFIG._replace(block_size_joblib=size)
            with Parallel(
                n_jobs=CONFIG.processes_jl,
                backend='loky',
                temp_folder=temp_dir,
                verbose=0,
                batch_size=8,
                pre_dispatch='2*n_jobs'
            ) as parallel:
                for _ in range(CONFIG.iterations):
                    s_time = time.perf_counter()
                    run_parallel_joblib(target_img, t_cfg, parallel)  
                    jl_bench_data.append({
                        "Category": "Block Size", 
                        "Value": size, 
                        "Time": (time.perf_counter() - s_time) * 1000
                    })

        for p in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            p_cfg = CONFIG._replace(processes_jl=p)  
            with Parallel(
                n_jobs=p, 
                backend='loky',
                temp_folder=temp_dir,
                verbose=0,
                batch_size=8,
                pre_dispatch='2*n_jobs'
            ) as parallel:
                for _ in range(CONFIG.iterations):
                    s_time = time.perf_counter()
                    run_parallel_joblib(target_img, p_cfg, parallel)  
                    jl_bench_data.append({
                        "Category": "Processes", 
                        "Value": p, 
                        "Time": (time.perf_counter() - s_time) * 1000
                    })

        for be in ['loky', 'multiprocessing', 'threading']:
            with Parallel(n_jobs=CONFIG.processes_jl, 
                        backend=be,
                        temp_folder=temp_dir,
                        verbose=0,
                        batch_size=8,         
                        pre_dispatch='2*n_jobs') as parallel:
                for _ in range(CONFIG.iterations):
                    s_time = time.perf_counter()
                    run_parallel_joblib(target_img, CONFIG, parallel)
                    jl_bench_data.append({
                        "Category": "Backend", 
                        "Value": be, 
                        "Time": (time.perf_counter()-s_time)*1000
                    })

        for b in [1, 2, 4, 8, 16, 32]:
            with Parallel(n_jobs=CONFIG.processes_jl, 
                        backend='loky', 
                        temp_folder=temp_dir,
                        verbose=0,
                        batch_size=b,
                        pre_dispatch='2*n_jobs') as parallel:
                for _ in range(CONFIG.iterations):
                    s_time = time.perf_counter()
                    run_parallel_joblib(target_img, CONFIG, parallel)
                    jl_bench_data.append({
                        "Category": "Batch Size", 
                        "Value": b, 
                        "Time": (time.perf_counter()-s_time)*1000
                    })
        df_jl = pd.DataFrame(jl_bench_data)
        cleaned_jl_list = []
        for (cat, val), group in df_jl.groupby(['Category', 'Value']):
            cleaned_jl_list.append(remove_outliers_iqr(group, "Time"))
        df_jl_clean = pd.concat(cleaned_jl_list)

        plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 2, 1)
        sns.boxplot(data=df_jl_clean[df_jl_clean["Category"] == "Block Size"], x="Value", y="Time", palette="pastel")
        plt.title("Joblib: Block Size Impact")
        
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df_jl_clean[df_jl_clean["Category"] == "Processes"], x="Value", y="Time", palette="flare")
        plt.title("Joblib: Process Count Impact (n_jobs)")
        
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df_jl_clean[df_jl_clean["Category"] == "Backend"], x="Value", y="Time", palette="Set3")
        plt.title("Joblib: Backend Engine Comparison")
        
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df_jl_clean[df_jl_clean["Category"] == "Batch Size"], x="Value", y="Time", palette="muted")
        plt.title("Joblib: Batch Size Task Distribution")
        
        plt.tight_layout()
        plt.show()

        print("\nJoblib Stats:")
        print(df_jl_clean.groupby(['Category', 'Value'])['Time'].agg(['mean', 'std', 'median']).reset_index())

    get_reusable_executor().shutdown(wait=True)
    shutil.rmtree(temp_dir, ignore_errors=True)