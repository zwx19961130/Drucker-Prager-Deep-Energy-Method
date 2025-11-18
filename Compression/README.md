# Drucker–Prager Deep Energy Method (Compression)

Python driver for the uniaxial compression study in the SCI submission. The workflow is self contained in three files in this folder:
- `Plasticity_DEM_GPU.py`: entry point that builds the domain, schedule, and runs the simulation (`EXAMPLE=5`).
- `DeepMixedMethod.py`: training/solver loop (L-BFGS + guardrails) and post-processing/outputs.
- `DEM_Lib.py`: utilities for mesh import, material routines, constitutive updates, and logging.

The `AbaqusVerify/` folder is not required for running the Python scripts.

## Requirements
- Python 3.9+.
- PyTorch (GPU build preferred; falls back to CPU if CUDA is unavailable).
- NumPy, Matplotlib, PyVista (for VTK output; optional but recommended), CSV is from the stdlib.
- Optional: `rff` for positional encoding (falls back to plain MLP if missing).

## Input data
- Provide an Abaqus mesh file named `cube.inp` in this directory. It should contain the cylinder mesh used for uniaxial compression with a bounding box of `[1.0, 1.0, 1.0]`. Hex (8-node) or tet (4-node) elements are detected automatically.

## How to run
```bash
python Plasticity_DEM_GPU.py
```

Environment toggles (optional):
- `DEM_ANOMALY=1` enables PyTorch anomaly detection.
- `DEM_NR_DEBUG=1` prints Newton-loop diagnostics; `DEM_BACKWARD_DEBUG=0` silences backward logs.

## What the script does
1. Reads `cube.inp`, builds the domain, and sets material parameters for Drucker–Prager plasticity (φ = 43.3°, ψ = 0°, σc = 24.35 in compression-positive sign convention).
2. Generates a displacement schedule from 0 to −0.002 with finer resolution around yield.
3. Trains the mixed neural solver with L-BFGS (up to 10k iterations per step) and Newton corrector (100 iters) using double precision physics.
4. Saves checkpoints and diagnostics every load step.

## Outputs
All results are written to `./Example5/` plus a global log:
- `DEM_simulation_<timestamp>.txt`: full log with material parameters, convergence info, and consistency checks.
- `AllDiff.npy`: per-step loss/convergence history.
- `top_surface_reaction_displacement.csv`: applied displacement vs. reaction force and S33 at the loaded face.
- `Uniaxial_Compression_CylinderResults.vtk`: field data (displacement, strain, stress, PEEQ, backstress) for visualization in ParaView.
- `Uniaxial_Compression_Cylinder_SS.txt`: volume-averaged stress/strain per step.
- `TrainedModel_Step*` and `model_step_*.pth`: model checkpoints after each step.

## Expected results (for reviewers)
- Linear elastic response up to yield strain εy ≈ σc / E ≈ 8.1e‑4.
- Yield force on the top face ≈ σc · A ≈ 19 N (cylinder area ~0.785); reaction and assembled forces agree within ~1–2% in elastic steps.
- Post-yield the reaction flattens (perfect plasticity, H = 0), with PEEQ increasing smoothly.
- Yield surface consistency: |q − p·tanβ − d| ≤ 1e-6 at plastic points; no overshoot beyond σc.
- Compression-positive convention: compressive stress/strain and reaction forces are reported as positive magnitudes.

## Tips
- GPU memory: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the driver; reduce batch sizes only if your GPU is very small.
- Visualization: open the VTK file in ParaView; S33 and PEEQ fields are stored per element, displacements per node.
- Reproducibility: seeds are fixed (`torch.manual_seed(2022)`), and all physics runs in float64 for stability.
