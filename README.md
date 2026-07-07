# Drucker–Prager Deep Energy Method

The repository hosts the Python drivers, helper routines, and Abaqus mesh inputs for the two benchmark problems discussed in the paper:

- **Uniaxial compression of a Drucker–Prager cylinder** (`Compression/`)
- **Plate with a central hole in traction** (`Traction/`)

Each example contains the driver, solver, and helper routines required to reproduce the figures in the manuscript, together with diagnostic outputs for independent verification.

## Repository layout

| Path | Description |
| ---- | ----------- |
| `Compression/Plasticity_DEM_GPU.py` | Entry point for the compression benchmark. |
| `Compression/DeepMixedMethod.py` | Mixed energy minimization + Newton corrector used by both studies. |
| `Compression/DEM_Lib.py` | Shared utilities (mesh import, constitutive updates, logging, plotting). |
| `Traction/…` | Same structure as `Compression/`, specialized to the plate-with-hole traction test. |
| `*/AbaqusVerify/` | Abaqus verification decks and mesh files. Copy the benchmark mesh one level up before running the Python driver. |

Each subdirectory provides a concise README with problem-specific notes; start here for a high-level overview.

## Method at a glance

1. **Physics-informed energy minimization:** The neural mixed method minimizes an incremental energy-like computational objective coupled with a Drucker-Prager return-mapping update for the internal variables.
2. **Double precision & guardrails:** All tensors are evaluated in float64 with L-BFGS optimization followed by a Newton corrector to improve equilibrium and yield-surface consistency, with typical yield-function residual targets of `|f| <= 1e-6`.
3. **Verification hooks:** The scripts log displacement–reaction curves, energy terms, and VTK field data to facilitate comparison with the reported figures or independent FEM solutions (e.g., Abaqus).

## Requirements

- Python 3.9 or newer.
- PyTorch (GPU build recommended but CPU is automatically supported).
- NumPy, Matplotlib, PyVista (for VTK export), and the Python standard library.
- Optional: `rff` for positional encoding (falls back to a plain MLP when missing).
- Mesh data in Abaqus `.inp` format are included with the verification files:
  - `Compression/AbaqusVerify/cube.inp`: cylinder mesh with bounding box `[1.0, 1.0, 1.0]`.
  - `Traction/AbaqusVerify/Hole.inp`: plate-with-hole mesh with bounding box `[4.0, 4.0, 1.0]`.
  - Before running a case, copy the required mesh into the corresponding case directory as `cube.inp` or `Hole.inp`, because the Python drivers read the mesh from the working directory.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install numpy matplotlib pyvista
pip install torch --index-url https://download.pytorch.org/whl/cu121  # choose the wheel for your CUDA setup
```

After installing the dependencies, copy the appropriate Abaqus mesh into the matching directory and run the corresponding driver:

```bash
cd Compression
cp AbaqusVerify/cube.inp cube.inp      # Windows PowerShell: Copy-Item AbaqusVerify\cube.inp cube.inp
python Plasticity_DEM_GPU.py

# or, for the plate-with-hole benchmark:
cd ../Traction
cp AbaqusVerify/Hole.inp Hole.inp      # Windows PowerShell: Copy-Item AbaqusVerify\Hole.inp Hole.inp
python Plasticity_DEM_GPU.py
```

Environment flags such as `DEM_ANOMALY=1`, `DEM_NR_DEBUG=1`, or the traction-specific `DP_EXTRA_DISP` can be set before running to toggle debugging, anomaly detection, or extra displacement checkpoints.

## Example workflows

### 1. Uniaxial compression (`Compression/`)

- Loads the Drucker–Prager cylinder example used to produce the compression plots in the paper.
- Applies a displacement-controlled schedule from 0 to −0.002 with refined increments near yield (σc = 24.35 MPa, φ = 43.3°, ψ = 0°).
- Saves logs, per-step training histories, reaction/displacement CSVs, and a ParaView-ready `Uniaxial_Compression_CylinderResults.vtk` file.
- Expected response: elastic up to ε_y ≈ 8.1×10⁻⁴ followed by a plastic plateau (perfect plasticity, H = 0); reaction and assembled forces agree within ~1–2%.

### 2. Plate with hole in traction (`Traction/`)

- Reproduces the traction benchmark in the paper with the same material parameters and boundary conditions as the Abaqus verification study.
- Displacement-controlled loading from 0 to 0.004 (default 20 steps, optional extra checkpoints via `DP_EXTRA_DISP`).
- Produces VTK snapshots (`Step<N>Results.vtk`), `PlateWithHole_Y_Tension_SS.txt`, and loss histories for each step.
- Expected response: elastic up to ε_y ≈ 8.1×10⁻⁴ followed by the reaction trend reported in the manuscript. Logs report yield-function residuals and force/energy balance checks for validation.

## Outputs reviewers should expect

- Timestamped `DEM_simulation_*.txt` logs summarizing material parameters, solver tolerances, convergence history, and success/failure codes.
- `AllDiff.npy` plus per-step loss `.npy` files to inspect training stability.
- Stress/strain summaries (`*_SS.txt`) matching the curves shown in the manuscript within a few percent.
- VTK files containing displacement, strain, stress, equivalent plastic strain, and yield-function-related fields.

## Tips for reproduction

- **Determinism:** Random seeds are fixed (`torch.manual_seed(2022)`) and double precision is enforced throughout.
- **Hardware:** GPU execution is automatic when CUDA is available; otherwise the scripts run on CPU with the same results (slower).
- **Debugging:** `DEM_BACKWARD_DEBUG=0` silences overly verbose backward logs; enabling `DEM_NR_DEBUG=1` prints Newton loop diagnostics useful for reviewers.
- **Verification:** Abaqus reference decks are provided under each `AbaqusVerify/` folder; they are not required but help double-check meshes or constitutive settings.

Questions about the code or reproduction steps are welcome—feel free to open an issue in the corresponding repository or contact the authors through the journal’s communication channel.
