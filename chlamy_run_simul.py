import pickle
import copy
import numpy as np
import scipy
import scipy.signal as sp
from chlamy_packages import compute_full_trajectory
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import os


def main_task(args):
    """Main execution. Function computes trajectories, then the displacements for 
    each selected time step, and fills the corresponding histograms which are returned."""
    n, data_dict, N, tau, R, H, eta_0, kbT, D_r, v, T, Delta_P, L, W, oscillations, output_file = args
    data = copy.deepcopy(data_dict)

    for i in range(data_dict["N_trajectories"]):
        x, y, phi, N, tau, R, H, eta_0, kbT, D_r, v, T, Delta_P, L, W, oscillations = compute_full_trajectory(
                R, H, eta_0, kbT, 
                D_r, v, N,
                T, tau, Delta_P, L, W, oscillations)


        data[f'x_{i}'] = x
        data[f'y_{i}'] = y
        data[f'phi_{i}'] = phi


    filename = output_file + "_" + str(n) + ".pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return f"Task {n} completed"

def main_execution():
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("-N", "--N", type=int, default=1e6, help="Number of time steps. (default: 1e6)")
    parser.add_argument("-tau", "--tau", type=float, default=1e-3, help="Time step duration in seconds. (default: 1e-3)")
    parser.add_argument("-R", "--R", type=float, default=5e-6, help="Particle radius in m. (default: 5e-6)")
    parser.add_argument("-H", "--H", type=float, default=100e-6, help="Cell height in m. (default: 100e-6)")
    parser.add_argument("-eta_0", "--eta_0", type=float, default=1e-3, help="Fluid bulk viscosity. (default: 1e-3)")
    parser.add_argument("-kbT", "--kbT", type=float, default=300. * 1.380649e-23, help="Thermal energy. (default: 300. * 1.380649e-23)")
    parser.add_argument("-D_r", "--D_r", type=float, default=3.35, help="Rotational diffusivity. (default: 0.08.)")
    parser.add_argument("-v", "--v", type=float, default=100e-6, help="Particle self-propelling speed. (default: 100e-6)")
    parser.add_argument("-T", "--T", type=float, default=1., help="Sinusoidal oscillation temporal period. (default: 1.)")
    parser.add_argument("-Delta_P", "--Delta_P", type=float, default=1e5, help="Pressure gradient. (default: 1e5)")
    parser.add_argument("-L", "--L", type=float, default=100e-3, help="Canal length. (default: 100e-3)")
    parser.add_argument("-W", "--W", type=float, default=180e-6, help="Canal Width. (default: 180e-6)")
    parser.add_argument("-oscillations", "--oscillations", type=bool, default=False, help="Whether flow oscillates or not. (default: True for yes)")
    parser.add_argument("-o", "--output", type=str, default="save", help="Output directory")
    parser.add_argument("-Nt", "--N_trajectories", type=int, default=10, help="Number of trajectories per task")
    parser.add_argument("-nt", "--n_tasks", type=int, default=10, help="Number of tasks to run")
    parser.add_argument("-np", "--n_workers", type=int, default=10, help="Number of workers")
    parser.add_argument("-j", "--job_ID", type=int, default=10, help="JOB ID number")
    args = parser.parse_args()

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
    oscillations = args.oscillations
    output_dir = args.output
    N_trajectories = args.N_trajectories
    n_tasks = args.n_tasks
    job_ID = args.job_ID


    data_dict = {}

    # Add the number of trajectories
    data_dict["N_trajectories"] = N_trajectories
    data_dict["tau"] = tau
    data_dict["t"] = np.arange(N) * tau

    for i in range(data_dict["N_trajectories"]) :
        data_dict[f"x_{i}"] = np.zeros(N)
        data_dict[f"y_{i}"] = np.zeros(N)
        data_dict[f"phi_{i}"] = np.zeros(N)



    # data_dict["time_lags"] = time_lags

    # Construct the output file path
    output_file = os.path.join(
        output_dir,
        '-'.join(
            f'{key}_{value}' for key, value in {
                "N": N, "tau": tau, "R": R, "H": H, "eta_0": eta_0,
                "kbT": kbT, "D_r": D_r, "v": v, "T": T, "delta_P": Delta_P, 
                "L": L, "W": W, "oscillations": oscillations, "N_trajectories": N_trajectories, "Job_ID": job_ID,
            }.items()
        )
    )


    os.makedirs(output_dir, exist_ok=True)

    args_list = [
        (n, data_dict, N, tau, R, H, eta_0, kbT, D_r, v, T, Delta_P, L, W, oscillations, output_file)
        for n in range(n_tasks)
    ]

    t1 = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(main_task, args_list))
    t2 = time.time()

    for result in results:
        print(result)

    print(t2 - t1)

if __name__ == "__main__":
    main_execution()
