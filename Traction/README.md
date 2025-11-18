# Drucker–Prager Deep Energy Method (Traction)

This directory contains the code used in the paper to solve a plate-with-hole traction test with a Drucker–Prager material. The main driver is `Plasticity_DEM_GPU.py`, which uses the core library in `DEM_Lib.py` and the solver/training logic in `DeepMixedMethod.py`.

## Prerequisites
- Python 3.9+.
- PyTorch (GPU build recommended for speed), NumPy, Matplotlib, PyVista (for VTK export; optional but recommended for visualization).
- CUDA-capable GPU is used automatically if available; otherwise the scripts fall back to CPU.
- Mesh input: an Abaqus-style mesh file named `Hole.inp` (or `Hole` with `.inp` extension) in this directory, matching the plate-with-hole geometry (BoundingBox `[4.0, 4.0, 1.0]`). If you already have the Abaqus deck from the verification run, place/copy `Hole.inp` here.

## How to Run
1) Install dependencies (example using pip):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy matplotlib pyvista
   # Install a CUDA-enabled torch build appropriate for your system, e.g.:
   # pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2) Ensure `Hole.inp` is present in this folder (the code will also try `Hole` and `Hole.inp` automatically).
3) Run the driver:
   ```bash
   python Plasticity_DEM_GPU.py
   ```
   The script prints progress to the console and also writes a timestamped log (e.g., `DEM_simulation_YYYYMMDD_HHMMSS.txt`).

### Optional configuration
- `DP_EXTRA_DISP`: comma-separated extra displacement checkpoints (e.g., `DP_EXTRA_DISP="0.0005,0.00075"`).
- `DP_K_RATIO`: Abaqus K parameter (σ_t/σ_c ratio), default `1.0`.
- `DP_COHESION_MODE` / `DP_SIGMA_TENSION_TARGET`: cohesion calibration for non-symmetric yield.
- `DEM_NR_DEBUG`, `DEM_BACKWARD_DEBUG`: enable/disable Newton/backward debug prints.
- `DEM_ANOMALY=1`: enable PyTorch anomaly detection (slower).

## Outputs
All outputs are written under `Example5/`:
- `Step<N>Results.vtk`: per-step VTK files with displacement, strain, stress, von Mises, and PEEQ (for visualization in ParaView/PyVista).
- `Step<N>Training_loss.npy`: loss history for each load step.
- `AllDiff.npy`: aggregated per-step loss/diagnostic array.
- `DiffLog`: placeholder log file.
- `PlateWithHole_Y_Tension_SS.txt`: running mean stress/strain summary (appended each step).

## Expected Behavior (plate-with-hole tension)
- Load is displacement-controlled in +Y from 0 to 0.004 over 20 steps (extra checkpoints if configured).
- Elastic response up to yield strain ε_y ≈ σ_c/E ≈ 8.1e-4 (σ_c = 24.35 MPa, E = 30 GPa).
- Reaction force at yield ≈ 19 N (for ~0.785 mm² nominal area), then a near-plateau as perfect plasticity engages.
- Yield surface checks target |f| ≤ 1e-6 at plastic points; surface/volume/energy forces typically agree within a few percent.
- Successful runs end with `Simulation sequence completed successfully.` in the log and produce VTK files for each step.
