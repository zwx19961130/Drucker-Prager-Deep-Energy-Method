# Modified Plasticity_DEM_GPU.py
import os
# DEPTHCORE-Ξ-PATCH-11: Prevent allocator degradation from transient spike
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16'
from DEM_Lib import *
from DeepMixedMethod import DeepMixedMethod
import logging
import numpy as np

# Get the logger from DEM_Lib
logger = logging.getLogger('DEM_simulation')

# Import global epsilon for numerical stability
from DEM_Lib import eps_global

EXAMPLE = 5

# --- ACTION 1: ENFORCE TENSOR TYPES AT THE SOURCE ---
# All physical constants MUST be defined as torch.tensors to prevent type errors.

# Material Properties
YM = torch.as_tensor(30000.0, dtype=torch.float64, device=dev)
PR = torch.as_tensor(0.25, dtype=torch.float64, device=dev)

# For other examples that might use sig_y0
sig_y0 = torch.as_tensor(35.5, dtype=torch.float64, device=dev)  # Default yield stress for other examples

# ==================== CRITICAL PARAMETER DEFINITION START ======================
# Drucker-Prager Parameters as defined in Abaqus
FRICTION_ANGLE = 43.3

# --- FLOW RULE SELECTION ---
# Use NON-ASSOCIATED flow by default to match Abaqus plate-with-hole reference (ψ = 0°).
# To switch to associated, set DILATION_ANGLE = FRICTION_ANGLE.
# Match Abaqus deck: non-associated flow (ψ = 0°)
DILATION_ANGLE = 0.0
logger.info(f"*** FLOW RULE: Non-Associated (ψ = {DILATION_ANGLE}°) unless changed ***")

# The hardening input, which is the UNIAXIAL compressive yield strength, NOT the final stress.
# PATCH-50-CRITICAL: Add Abaqus K parameter (flow stress ratio σ_t/σ_c) support
# Abaqus material card shows K=1.0 → SYMMETRIC YIELD (σ_t = σ_c)
# NOT the mathematical DP asymmetry (σ_t = 0.522×σ_c)!
DP_K_RATIO = float(os.environ.get('DP_K_RATIO', '1.0'))  # CONFIG: Abaqus K parameter
sigma_c_yield = torch.as_tensor(24.35, dtype=torch.float64, device=dev)

# No more smooth_alpha parameter - the custom autograd function handles everything

# Convert friction/dilation angles (degrees) to p–q parameters
phi_rad = FRICTION_ANGLE * np.pi / 180.0
psi_rad = DILATION_ANGLE * np.pi / 180.0
sin_phi = np.sin(phi_rad)
cos_phi = np.cos(phi_rad)
sin_psi = np.sin(psi_rad)

# p–q form parameters for DP (Abaqus): use friction angles directly
# q = sqrt(3 J2), p = −I1/3, yield f = q − p*tan(β) − d
tan_beta_pq = np.tan(phi_rad)
tan_psi_pq = np.tan(psi_rad)
TAN_BETA = torch.as_tensor(tan_beta_pq, dtype=torch.float64, device=dev)
SIN_BETA = torch.as_tensor(sin_phi, dtype=torch.float64, device=dev)
COS_BETA = torch.as_tensor(cos_phi, dtype=torch.float64, device=dev)
TAN_PSI = torch.as_tensor(tan_psi_pq, dtype=torch.float64, device=dev)

# ============================================================================
# FIX F: Correct Drucker-Prager parameter conversion per RevisionIdea.md Section F
# ============================================================================
# Standard DP form: f = sqrt(J2) + α*I1 - k = 0
# Where: α = (2*sin(φ))/(sqrt(3)*(3-sin(φ)))
#        k = (6*d*cos(φ))/(sqrt(3)*(3-sin(φ)))
#        d = σ_c * (1-sin(φ))/(2*cos(φ))
# 
# Alternative p-q form: f = q - p*tan(β) - d_pq = 0
# Where: q = sqrt(3*J2), p = -I1/3 (compression positive)
#        tan(β) = sqrt(3)*α
#        d_pq = k/sqrt(3)
# ============================================================================
from math import sqrt

# Calculate α and k using standard DP formulation
alpha_dp = (2.0 * sin_phi) / (sqrt(3.0) * (3.0 - sin_phi))
d_material = sigma_c_yield.item() * (1.0 - sin_phi) / (2.0 * cos_phi)
k_dp = (6.0 * d_material * cos_phi) / (sqrt(3.0) * (3.0 - sin_phi))

# Reference cohesion (p–q form) from MC mapping (not used by solver)
cohesion_d_mc = torch.as_tensor(k_dp / sqrt(3.0), dtype=torch.float64, device=dev)

# Cohesion calibration mode for DP (Abaqus parity for tension tests)
#   'compression': d = (1 - tanβ/3)·σ_c  (matches uniaxial compression yield)
#   'tension'    : d = (1 + tanβ/3)·σ_t  (matches uniaxial tension yield)
#   'symmetric'  : d = (1 - tanβ/3)·σ_0  (K=1.0, σ_t=σ_c=σ_0)
#   'manual'     : set via env var DP_COHESION_MANUAL (absolute value)
# PATCH-50: Auto-select based on K parameter
if abs(DP_K_RATIO - 1.0) < 0.01:
    # K ≈ 1.0 → Symmetric yield (Abaqus default)
    COHESION_CALIBRATION = 'symmetric'
else:
    # K ≠ 1.0 → Use mathematical DP asymmetry
    COHESION_CALIBRATION = os.environ.get('DP_COHESION_MODE', 'tension').lower()

SIGMA_TENSION_TARGET = os.environ.get('DP_SIGMA_TENSION_TARGET', None)

if COHESION_CALIBRATION == 'symmetric':
    # PATCH-50-CRITICAL: K=1.0 means σ_t = σ_c (symmetric yield)
    # Use compression formula with σ_0 = σ_c
    sigma_t_effective = sigma_c_yield  # Same yield in tension and compression
    cohesion_d = (1.0 - TAN_BETA / 3.0) * sigma_c_yield
elif COHESION_CALIBRATION == 'compression':
    cohesion_d = (1.0 - TAN_BETA / 3.0) * sigma_c_yield
elif COHESION_CALIBRATION == 'tension':
    if SIGMA_TENSION_TARGET is not None:
        try:
            sigma_t_ref = torch.as_tensor(float(SIGMA_TENSION_TARGET), dtype=torch.float64, device=dev)
        except Exception:
            # PATCH-49-CRITICAL: Convert compression yield to tension yield for DP asymmetry
            # Abaqus provides TYPE=COMPRESSION (σ_c), but we need σ_t for tension test
            # Formula: σ_t = σ_c × (1 - tan(β)/3) / (1 + tan(β)/3)
            sigma_t_ref = sigma_c_yield * (1.0 - TAN_BETA / 3.0) / (1.0 + TAN_BETA / 3.0)
    else:
        # PATCH-49-CRITICAL: Convert compression yield to tension yield for DP asymmetry
        # Abaqus TYPE=COMPRESSION gives σ_c=24.35 MPa (compression yield)
        # For tension test, convert: σ_t = σ_c × (1-tanβ/3)/(1+tanβ/3) = 24.35×0.522 = 12.71 MPa
        sigma_t_ref = sigma_c_yield * (1.0 - TAN_BETA / 3.0) / (1.0 + TAN_BETA / 3.0)
    cohesion_d = (1.0 + TAN_BETA / 3.0) * sigma_t_ref
elif COHESION_CALIBRATION == 'manual':
    cohesion_manual = float(os.environ.get('DP_COHESION_MANUAL', '0.0'))
    cohesion_d = torch.as_tensor(cohesion_manual, dtype=torch.float64, device=dev)
else:
    cohesion_d = (1.0 - TAN_BETA / 3.0) * sigma_c_yield

logger.info(f"*** DP Parameters (Aligned to Uniaxial Yield) ***")
logger.info(f"  Input σ_c (uniaxial compressive) = {sigma_c_yield.item():.4f}")
logger.info(f"  Abaqus K parameter (σ_t/σ_c ratio) = {DP_K_RATIO:.2f}")
logger.info(f"  Friction angle φ = {FRICTION_ANGLE}°")
logger.info(f"  α (MC ref) = {alpha_dp:.6f}")
logger.info(f"  d_material (MC ref) = {d_material:.4f}")
logger.info(f"  k (MC ref) = {k_dp:.4f}")
logger.info(f"  d_pq (MC ref) = {cohesion_d_mc.item():.4f}")
logger.info(f"  Cohesion calibration mode = {COHESION_CALIBRATION}")
if COHESION_CALIBRATION == 'symmetric':
    logger.info(f"  Using cohesion d = (1 - tanβ/3)·σ_0 = {cohesion_d.item():.4f}")
    logger.info(f"  K=1.0 → Symmetric yield: σ_t = σ_c = {sigma_c_yield.item():.4f} MPa")
elif COHESION_CALIBRATION == 'compression':
    logger.info(f"  Using cohesion d = (1 - tanβ/3)·σ_c = {cohesion_d.item():.4f}")
elif COHESION_CALIBRATION == 'tension':
    sigma_t_implied = cohesion_d.item() / (1.0 + float(TAN_BETA.item()) / 3.0)
    logger.info(f"  Using cohesion d = (1 + tanβ/3)·σ_t = {cohesion_d.item():.4f}")
    logger.info(f"  Implied tension yield: σ_t = {sigma_t_implied:.4f} MPa")
elif COHESION_CALIBRATION == 'manual':
    logger.info(f"  Using cohesion d (manual) = {cohesion_d.item():.4f}")
logger.info(f"  Verification: tan(β) (p–q) = {TAN_BETA.item():.6f}")

# Define the FlowStress function for Drucker-Prager (为兼容性保留，但不再使用)
def FlowStressDP(PEEQ):
    # 线性硬化：屈服应力随 PEEQ 增加
    H = HardeningModulusDP(PEEQ)
    return sigma_c_yield + H * PEEQ

# Define the HardeningModulus function (为兼容性保留，但不再使用)
def HardeningModulusDP(PEEQ):
    # Perfect plasticity for plateau: H = 0
    if isinstance(PEEQ, torch.Tensor):
        return torch.zeros_like(PEEQ)
    return 0.0
# ===================== CRITICAL PARAMETER DEFINITION END =====================

def FlowStressLinear( eps_p_eff ):
    return sig_y0 +  ( YM / 2. ) * eps_p_eff
def FlowStressKinematic( eps_p_eff ):
    return sig_y0 + 0 * eps_p_eff
def HardeningModulusLinear( eps_p_eff ):
    return YM / 2.
def ZeroFunc(eps):
    # Helper for EXAMPLE 4: ideal plastic DP (no hardening)
    if isinstance(eps, torch.Tensor):
        return torch.zeros_like(eps)
    return 0.0


# Setup examples
UNIFORM = True

# --- START OF MODIFICATIONS ---

# Define the geometry file and its bounding box for the plate-with-hole problem
# Note: You mentioned geometry/mesh is handled; here we point to the provided input.
GeometryFile = 'Hole'
# BoundingBox: [Lx, Ly, Lz] in your model units (Y-direction length should be 4.0 per requirement)
BoundingBox = [4.0, 4.0, 1.0]

def run_simulation():
    """
    Uniaxial compression test with Drucker-Prager plasticity.
    
    EXPECTED BEHAVIOR per RevisionIdea.md:
    1. Linear elastic response up to yield strain ε_y ≈ σ_c/E ≈ 8.12e-4
    2. At yield: reaction force F ≈ σ_c * A ≈ 19 N (for A ≈ 0.785 mm²)
    3. Post-yield: force plateaus (perfect plasticity) with PEEQ increasing slowly
    4. Stress remains on yield surface: |f| = |q - p*tan(β) - d| ≤ 1e-6
    5. No stress overshoot beyond σ_c (return mapping enforced)
    """
    if EXAMPLE != 5:
        logger.error(f"Example {EXAMPLE} not implemented in this driver.")
        return

    logger.info('Uniaxial Tension Test: Plate with central hole, loading in +Y')
    logger.info("--- Flow Rule ---")

    logger.info(f"Uniaxial Compressive Yield Input (sigma_c): {sigma_c_yield.item():.4f}")
    logger.info(f"Material Friction Angle (β): {FRICTION_ANGLE}°")
    logger.info(f"Material Dilation Angle (ψ): {DILATION_ANGLE}° ({'ASSOCIATED' if DILATION_ANGLE==FRICTION_ANGLE else 'NON-ASSOCIATED'} FLOW)")

    logger.info(f"Calculated True Cohesion (d): {cohesion_d.item():.4f}")

    # Expected uniaxial yield stress is exactly the input σc (by construction)
    expected_yield_stress = sigma_c_yield
    logger.info(f"Theoretical uniaxial yield stress (S22): {expected_yield_stress.item():.4f}")

    domain = setup_domain(GeometryFile, BoundingBox)
    if domain is None:
        logger.error('Failed to load domain; aborting simulation.')
        return

    ref_file = 'PlateWithHole_Y_Tension'
    KINEMATIC = False
    FlowStress = FlowStressDP
    HardeningModulus = HardeningModulusDP

    # Displacement-controlled tension along +Y up to 0.01 at Y=Ly
    start_disp = 0.0
    end_disp = 0.004
    n_steps = 20
    disp_schedule = np.linspace(start_disp, end_disp, n_steps + 1).tolist()
    # Optional: inject extra displacement checkpoints via env var (comma-separated), e.g. DP_EXTRA_DISP="0.0005,0.00075"
    extra_disp_env = os.environ.get('DP_EXTRA_DISP', '').strip()
    if extra_disp_env:
        try:
            extras = [float(s) for s in extra_disp_env.split(',') if s.strip()]
            disp_schedule = sorted(set(disp_schedule + extras))
        except Exception as e:
            logger.warning(f"Failed to parse DP_EXTRA_DISP ('{extra_disp_env}'): {e}")

    # Estimate number of steps until yield and plastic region
    # Yield displacement ε_y ≈ σ_c/E
    yield_disp = sigma_c_yield.item() / YM.item()
    step_size = (end_disp - start_disp) / n_steps
    num_elastic_steps = int(np.ceil(yield_disp / step_size))
    # Ensure at least one yield step
    num_yield_steps = max((len(disp_schedule) - 1) - num_elastic_steps, 1)
    n = len(disp_schedule) - 1
    rel_tol = np.ones(n) * 1e-9
    critical_start = max(num_elastic_steps - 2, 0)
    critical_end = min(num_elastic_steps + num_yield_steps + 2, n)
    rel_tol[critical_start:critical_end] = 1e-12

    base = f'./Example{EXAMPLE}/'
    os.makedirs(base, exist_ok=True)

    logger.info(f'Number of nodes is {domain["nN"]}')
    logger.info(f'Number of elements is {domain["nE"]}')

    step_max = len(disp_schedule) - 1
    # DEPTHCORE-Ξ-PATCH-8: Disable L-BFGS to test if it improves convergence speed
    LBFGS_Iteration = 0  # Disabled for performance testing (was: 10000)
    Num_Newton_itr = 100
    Settings = [
        KINEMATIC, FlowStress, HardeningModulus, disp_schedule, rel_tol,
        step_max, LBFGS_Iteration, Num_Newton_itr, EXAMPLE, YM, PR,
        cohesion_d, FRICTION_ANGLE, DILATION_ANGLE, TAN_BETA, SIN_BETA,
        COS_BETA, TAN_PSI, base, UNIFORM
    ]

    # >>>>> CRITICAL FIX D: Reduce Learning Rate <<<<<
    # Reduced LR from 0.01 to 1e-5 to stabilize Adam optimization for plastic steps.
    # Further shrink NN width so corrections stay closer to FE baseline.
    x_var = {'x_lr': 1e-5, 'neuron': 384, 'act_func': 'silu'}
    # Original: x_var = {'x_lr': 0.01, ...}
    # >>>>> END OF FIX D <<<<<
    
    # DEPTHCORE-Ξ-PATCH-10: Disable refinement phases to eliminate 15GB leak + util drop
    refine_lbfgs_iters = 0  # Disable refinement L-BFGS (was causing 915 evals)
    refine_adam_epochs = 0  # Disable refinement Adam
    
    lr = x_var['x_lr']
    H = int(x_var['neuron'])
    act_fn = x_var['act_func']
    logger.info(f'LR: {lr}, H: {H}, act fn: {act_fn}')

    diff_log_path = os.path.join(base, 'DiffLog')
    with open(diff_log_path, 'w', encoding='utf-8'):
        pass

    snet = S_Net(3, H, 3, act_fn)
    # PENALTY CONFIGURATION per RevisionIdea.md Section D:
    # - Sobolev and lateral traction: disabled (implementation issues)
    # - Equilibrium penalty: False here means "not forced on", but will be DYNAMICALLY ACTIVATED
    #   once plasticity is detected (max(PEEQ) > 1e-5) via the _equilibrium_penalty_on flag
    #   in DeepMixedMethod.train_model() around line 975.
    # This implements RevisionIdea.md recommendation: "Put back a minimal equilibrium penalty 
    # only in plastic steps" to prevent optimizer drift to low-σ, non-equilibrated states.
    # 
    # DEPTHCORE-FIX: CRITICAL STIFFNESS DEGRADATION BUG (Step 4)
    # ROOT CAUSE: loss_function minimizes internal_energy = W_elastic + W_hardening.
    #   Physics: W ∝ ε² MUST increase with loading (disp: 0.0002 → 0.0008)
    #   Optimizer: Minimizes W → reduces stress (σ = E·ε) to reduce W → stiffness degrades
    #   Evidence: Loss 5.14e-4 → 7.50e-3 (15×), S22 error 12%, F_top error 30%, f=24.91>0
    # FIX: Increase assembled_equilibrium_weight from 1e-4 to 1.0 to make residual dominant.
    #   This forces equilibrium satisfaction instead of energy minimization.
    #   Weight 1.0 makes ||R_equilibrium||² term comparable to W (~1e-3), ensuring balance.
    penalty_cfg = {
        'enable_sobolev': False,
        'enable_equilibrium': False,  # Dynamic activation in plastic regime (see above)
        'enable_lateral_traction': False,
        # Keep elastic-phase residual supportive, then tighten equilibrium once plasticity is detected
        'equilibrium_weight': 5e-2,
        'equilibrium_weight_plastic': 1e-1,
        'assembled_equilibrium_weight': 1e-2,
        'assembled_equilibrium_weight_plastic': 8e-2,
    }
    DEM = DeepMixedMethod([snet, lr, domain, Settings, penalty_cfg])
    # DEPTHCORE-Ξ-PATCH-10: Apply refinement phase settings
    DEM.refine_lbfgs_iters = refine_lbfgs_iters
    DEM.refine_adam_epochs = refine_adam_epochs
    # Remove manual traction weight settings (redundant when penalties are disabled)
    all_diff = DEM.train_model(disp_schedule, ref_file)
    np.save(os.path.join(base, 'AllDiff.npy'), all_diff)

    logger.info('Simulation sequence completed successfully.')
    return all_diff


if __name__ == '__main__':
    run_simulation()
