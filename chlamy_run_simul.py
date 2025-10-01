import pickle
import numpy as np
import scipy
import scipy.signal as sp
from chlamy_packages import compute_full_trajectory
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import sys


"""Note for later : RN the picles are writtin in a subdir, which is cool, but the treatment code used to work for pickles directly written 
in the main dir. It'd be cool to change that but it'd take me more time than to manually move pkls from subdir to dir. Thus,
I dont feel like doing it."""

def main_task(args):
    (n, meta, N, tau, R, H, eta_0, kbT, D_r, v, T, Delta_P, L, W, oscillations, run_dir) = args

    out = {
        "N_trajectories": meta["N_trajectories"],
        "tau": tau,
        "t": meta["t"],  
        "params": dict(N=N, tau=tau, R=R, H=H, eta_0=eta_0, kbT=kbT, D_r=D_r,
                       v=v, T=T, Delta_P=Delta_P, L=L, W=W, oscillations=oscillations)
    }

    for i in range(meta["N_trajectories"]):
        x, y, phi, *_ = compute_full_trajectory(
            R, H, eta_0, kbT, D_r, v, N, T, tau, Delta_P, L, W, oscillations
        )
        out[f'x_{i}'] = x
        out[f'y_{i}'] = y
        out[f'phi_{i}'] = phi

    filename = os.path.join(run_dir, f"N_{N}-tau_{tau}-R_{R}-H_{H}-kbT_{kbT}-D_r_{D_r}-v_{v}-T_{T}-tau_{tau}-delta_P_{Delta_P}-L_{L}-W_{W}-oscillations_{oscillations}-task_{n}.pickle")

    tmpfile = filename + ".tmp"
    with open(tmpfile, 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmpfile, filename)

    return f"Task {n} completed"

def main_execution():
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("-N", "--N", type=int, default=1_000_000, help="Number of time steps. (default: 1e6)")
    parser.add_argument("-tau", "--tau", type=float, default=1e-3, help="Time step duration in seconds. (default: 1e-3)")
    parser.add_argument("-R", "--R", type=float, default=5e-6, help="Particle radius in m. (default: 5e-6)")
    parser.add_argument("-H", "--H", type=float, default=100e-6, help="Cell height in m. (default: 100e-6)")
    parser.add_argument("-eta_0", "--eta_0", type=float, default=1e-3, help="Fluid bulk viscosity. (default: 1e-3)")
    parser.add_argument("-kbT", "--kbT", type=float, default=300 * 1.380649e-23, help="Thermal energy. (default: 300*1.380649e-23)")
    parser.add_argument("-D_r", "--D_r", type=float, default=3.35, help="Rotational diffusivity. (default: 3.35)")
    parser.add_argument("-v", "--v", type=float, default=100e-6, help="Particle self-propelling speed. (default: 100e-6)")
    parser.add_argument("-T", "--T", type=float, default=1., help="Sinusoidal oscillation temporal period. (default: 1.)")
    parser.add_argument("-Delta_P", "--Delta_P", type=float, default=1e5, help="Pressure gradient. (default: 1e5)")
    parser.add_argument("-L", "--L", type=float, default=100e-3, help="Canal length. (default: 100e-3)")
    parser.add_argument("-W", "--W", type=float, default=180e-6, help="Canal Width. (default: 180e-6)")
    parser.add_argument("-oscillations", "--oscillations", type=int, default=1, help="Whether flow oscillates or not. 1=yes, 0=no")
    parser.add_argument("-o", "--output", type=str, default="save", help="Top-level output directory")
    parser.add_argument("-Nt", "--N_trajectories", type=int, default=10, help="Number of trajectories per task")
    parser.add_argument("-nt", "--n_tasks", type=int, default=10, help="Number of tasks to run")
    parser.add_argument("-np", "--n_workers", type=int, default=10, help="Number of workers")
    args = parser.parse_args()

    if args.oscillations not in (0, 1):
        sys.exit("oscillations must be 0 or 1")
    oscillations = bool(args.oscillations) # i think this is still finnicky...

    N = args.N
    tau = args.tau
    R = args.R
    H = args.H
    eta_0 = args.eta_0
    kbT = args.kbT
    D_r = args.D_r
    v = args.v
    T = args.T
    Delta_P = args.Delta_P
    L = args.L
    W = args.W
    output_dir = args.output
    N_trajectories = args.N_trajectories
    n_tasks = args.n_tasks
    n_workers = args.n_workers

    # Create the top-level output directory early
    os.makedirs(output_dir, exist_ok=True)

    params_slug = '-'.join(
        f'{key}_{value}' for key, value in {
            "N": N, "tau": tau, "R": R, "H": H, "eta_0": eta_0,
            "kbT": kbT, "D_r": D_r, "v": v, "T": T, "delta_P": Delta_P,
            "L": L, "W": W, "oscillations": oscillations, "N_trajectories": N_trajectories,
        }.items()
    )
    run_dir = os.path.join(output_dir, params_slug)
    os.makedirs(run_dir, exist_ok=True)
    print(f"writing pickles under: {os.path.abspath(run_dir)}")

    meta = {
        "N_trajectories": N_trajectories,
        "t": np.arange(N) * tau,
    }

    args_list = [
        (n, meta, N, tau, R, H, eta_0, kbT, D_r, v, T, Delta_P, L, W, oscillations, run_dir)
        for n in range(n_tasks)
    ]

    t1 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(main_task, args_list))
    t2 = time.time()

    for result in results:
        print(result)
    print(f"elapsed: {t2 - t1:.2f} s")

if __name__ == "__main__":
    main_execution()
