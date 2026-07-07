# Modified Plasticity_DEM_GPU.py
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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

# Non-associated flow for Drucker-Prager geomaterial response.
DILATION_ANGLE = 0.0
logger.info(f"*** FLOW RULE: Non-Associated (ψ = {DILATION_ANGLE}°) ***")

# The hardening input, which is the UNIAXIAL compressive yield strength, NOT the final stress.
sigma_c_yield = torch.as_tensor(24.35, dtype=torch.float64, device=dev)

# No more smooth_alpha parameter - the custom autograd function handles everything

# Convert Abaqus Drucker-Prager friction/dilation angles (degrees) to p-q parameters.
# For the linear circular DP form used here (K=1), Abaqus gives the DP angle directly:
# q = sqrt(3 J2), p = -I1/3, yield f = q - p*tan(beta) - d.
phi_rad = FRICTION_ANGLE * np.pi / 180.0
sin_phi = np.sin(phi_rad)
cos_phi = np.cos(phi_rad)

tan_beta_pq = np.tan(phi_rad)
TAN_BETA = torch.as_tensor(tan_beta_pq, dtype=torch.float64, device=dev)
SIN_BETA = torch.as_tensor(sin_phi, dtype=torch.float64, device=dev)
COS_BETA = torch.as_tensor(cos_phi, dtype=torch.float64, device=dev)
psi_rad = DILATION_ANGLE * np.pi / 180.0
TAN_PSI = torch.as_tensor(np.tan(psi_rad), dtype=torch.float64, device=dev)

# ============================================================================
# Reference Mohr-Coulomb-to-Drucker-Prager conversion values for logging only.
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

# Reference cohesion (p-q form) from MC mapping (not used by solver)
cohesion_d_mc = torch.as_tensor(k_dp / sqrt(3.0), dtype=torch.float64, device=dev)

# Use uniaxial-consistent cohesion so yield occurs at σ_c: d = (1 - tanβ/3)·σ_c
cohesion_d = (1.0 - TAN_BETA / 3.0) * sigma_c_yield

logger.info(f"*** DP Parameters (Aligned to Uniaxial Yield) ***")
logger.info(f"  Input σ_c (uniaxial compressive) = {sigma_c_yield.item():.4f}")
logger.info(f"  Friction angle φ = {FRICTION_ANGLE}°")
logger.info(f"  α (MC reference, not used) = {alpha_dp:.6f}")
logger.info(f"  d_material (MC reference, not used) = {d_material:.4f}")
logger.info(f"  k (MC reference, not used) = {k_dp:.4f}")
logger.info(f"  d_pq (MC reference, not used) = {cohesion_d_mc.item():.4f}")
logger.info(f"  Using cohesion d = (1 - tanβ/3)·σ_c = {cohesion_d.item():.4f}")
logger.info(f"  Verification: tan(β) (Abaqus DP p-q) = {TAN_BETA.item():.6f}")

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

# Define the geometry file and its bounding box for uniaxial compression
GeometryFile = 'cube' 
# BoundingBox for the cylinder geometry used in uniaxial compression test
# Note: Update the cube.inp file to contain cylinder mesh in Abaqus
# Compression is now applied in Z-direction with centers at (0.5, 0.5, 0) and (0.5, 0.5, 1)
BoundingBox = [1.0, 1.0, 1.0]

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

    logger.info('Uniaxial Compression Test with Cylinder Geometry and Z-axis Compression')
    logger.info(f"--- Flow Rule: Non-Associated (ψ = {DILATION_ANGLE}°) ---")
    logger.info(f"Uniaxial Compressive Yield Input (sigma_c): {sigma_c_yield.item():.4f}")
    logger.info(f"Material Friction Angle (β): {FRICTION_ANGLE}°")
    logger.info(f"Material Dilation Angle (ψ): {DILATION_ANGLE}° (NON-ASSOCIATED FLOW)")
    logger.info(f"Calculated True Cohesion (d): {cohesion_d.item():.4f}")

    # Expected uniaxial yield stress is exactly the input σc (by construction)
    expected_yield_stress = sigma_c_yield
    logger.info(f"Theoretical actual yield stress (S33): {expected_yield_stress.item():.4f}")

    domain = setup_domain(GeometryFile, BoundingBox)
    if domain is None:
        logger.error('Failed to load domain; aborting simulation.')
        return

    ref_file = 'Uniaxial_Compression_Cylinder'
    KINEMATIC = False
    FlowStress = FlowStressDP
    HardeningModulus = HardeningModulusDP

    num_elastic_steps = 10
    num_yield_steps = 25
    num_plastic_steps = 15

    start_disp = 0.0
    yield_point_approx = -(expected_yield_stress / YM).item()
    logger.info(f"Theoretical yield stress: {expected_yield_stress.item():.4f}")
    logger.info(f"Theoretical yield strain: {yield_point_approx:.6f}")
    end_disp = -0.002

    disp_schedule = np.r_[
        np.linspace(start_disp, yield_point_approx, num_elastic_steps, endpoint=False),
        np.linspace(yield_point_approx, yield_point_approx * 1.2, num_yield_steps, endpoint=False),
        np.linspace(yield_point_approx * 1.2, end_disp, num_plastic_steps, endpoint=True)
    ].tolist()

    n = len(disp_schedule) - 1
    rel_tol = np.ones(n) * 1e-9
    critical_start = max(num_elastic_steps - 2, 0)
    critical_end = min(num_elastic_steps + num_yield_steps + 2, n)
    rel_tol[critical_start:critical_end] = 1e-12

    run_tag = os.environ.get('DP_RUN_TAG', '').strip()
    base = f'./Example{EXAMPLE}_{run_tag}/' if run_tag else f'./Example{EXAMPLE}/'
    os.makedirs(base, exist_ok=True)

    logger.info(f'Number of nodes is {domain["nN"]}')
    logger.info(f'Number of elements is {domain["nE"]}')

    step_max = len(disp_schedule) - 1
    # CRITICAL FIX 5: Increase L-BFGS iterations for better precision
    LBFGS_Iteration = 10000
    Num_Newton_itr = 100
    Settings = [
        KINEMATIC, FlowStress, HardeningModulus, disp_schedule, rel_tol,
        step_max, LBFGS_Iteration, Num_Newton_itr, EXAMPLE, YM, PR,
        cohesion_d, FRICTION_ANGLE, DILATION_ANGLE, TAN_BETA, SIN_BETA,
        COS_BETA, TAN_PSI, base, UNIFORM
    ]

    # Runtime controls for ablation/sensitivity studies.
    x_var = {
        'x_lr': float(os.environ.get('DP_LR', '1e-5')),
        'neuron': int(os.environ.get('DP_NEURON', '512')),
        'act_func': os.environ.get('DP_ACT', 'silu').strip().lower(),
    }
    
    lr = x_var['x_lr']
    H = int(x_var['neuron'])
    act_fn = x_var['act_func']
    logger.info(f'LR: {lr}, H: {H}, act fn: {act_fn}')
    logger.info(f"DP_TANGENT_MODE: {os.environ.get('DP_TANGENT_MODE', 'consistent').strip().lower()}")

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
    penalty_cfg = {
        'enable_sobolev': False,
        'enable_equilibrium': False,  # Dynamic activation in plastic regime (see above)
        'enable_lateral_traction': False,
        'equilibrium_weight': 1e-4,
    }
    DEM = DeepMixedMethod([snet, lr, domain, Settings, penalty_cfg])
    # Remove manual traction weight settings (redundant when penalties are disabled)
    all_diff = DEM.train_model(disp_schedule, ref_file)
    np.save(os.path.join(base, 'AllDiff.npy'), all_diff)

    logger.info('Simulation sequence completed successfully.')
    return all_diff


if __name__ == '__main__':
    run_simulation()
