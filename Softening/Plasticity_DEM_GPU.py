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

# The hardening input is the uniaxial compressive yield strength from
# *Drucker Prager Hardening, TYPE=COMPRESSION.
# K=1.0 makes the Abaqus linear Drucker-Prager deviatoric section circular;
# it does not make the uniaxial tensile and compressive yield stresses equal
# for f = q - p*tan(beta) - d.
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

cohesion_d = dp_softening_cohesion(torch.tensor(0.0, dtype=torch.float64, device=dev))

logger.info(f"*** DP Parameters (Softening Table) ***")
logger.info(f"  Input σ_c (uniaxial compressive) = {sigma_c_yield.item():.4f}")
logger.info(f"  Abaqus K parameter = {DP_K_RATIO:.2f} (K=1 circular deviatoric section)")
logger.info(f"  Friction angle φ = {FRICTION_ANGLE}°")
logger.info(f"  α (MC ref) = {alpha_dp:.6f}")
logger.info(f"  d_material (MC ref) = {d_material:.4f}")
logger.info(f"  k (MC ref) = {k_dp:.4f}")
logger.info(f"  d_pq (MC ref) = {cohesion_d_mc.item():.4f}")
logger.info(f"  Softening table anchor d(0) = {cohesion_d.item():.4f}")
logger.info(f"  Verification: tan(β) (p–q) = {TAN_BETA.item():.6f}")

# Shared softening helpers from DEM_Lib.py
def FlowStressDP(PEEQ):
    return dp_softening_cohesion(PEEQ)

# Local slope dd/dPEEQ from the same softening table
def HardeningModulusDP(PEEQ):
    return dp_softening_hardening_modulus(PEEQ)
# ===================== CRITICAL PARAMETER DEFINITION END =====================

def FlowStressLinear( eps_p_eff ):
    return sig_y0 +  ( YM / 2. ) * eps_p_eff
def FlowStressKinematic( eps_p_eff ):
    return sig_y0 + 0 * eps_p_eff
def HardeningModulusLinear( eps_p_eff ):
    return YM / 2.
def ZeroFunc(eps):
    # Helper for EXAMPLE 4: zero-slope plateau fallback
    if isinstance(eps, torch.Tensor):
        return torch.zeros_like(eps)
    return 0.0


# Setup examples
UNIFORM = True

# --- START OF MODIFICATIONS ---

# Define the geometry file and its bounding box for the reviewer softening benchmark.
# AbaqusVerify/GEOMODEL2.inp is a unit cylinder loaded in Z by the companion Abaqus deck.
GeometryFile = os.environ.get('DP_GEOMETRY_FILE', os.path.join('AbaqusVerify', 'GEOMODEL2.inp'))
BoundingBox = [1.0, 1.0, 1.0]
LOAD_AXIS = os.environ.get('DP_LOAD_AXIS', 'Z').strip().upper()
STRESS_AXIS_LABEL = {'X': 'S11', 'Y': 'S22', 'Z': 'S33'}.get(LOAD_AXIS, 'S??')

def run_simulation():
    """
    Uniaxial compression test with Drucker-Prager plasticity.
    
    EXPECTED BEHAVIOR per RevisionIdea.md:
    1. Linear elastic response up to yield strain ε_y ≈ σ_c/E ≈ 8.12e-4
    2. At yield: reaction force F ≈ σ_c * A ≈ 19 N (for A ≈ 0.785 mm²)
    3. Post-yield: force follows the shared softening table with PEEQ increasing
    4. Stress remains on yield surface: |f| = |q - p*tan(β) - d| ≤ 1e-6
    5. No stress overshoot beyond σ_c (return mapping enforced)
    """
    if EXAMPLE != 5:
        logger.error(f"Example {EXAMPLE} not implemented in this driver.")
        return

    logger.info(f'Uniaxial Compression Test: AbaqusVerify cylinder, loading in {LOAD_AXIS}')
    logger.info("--- Flow Rule ---")

    logger.info(f"Uniaxial Compressive Yield Input (sigma_c): {sigma_c_yield.item():.4f}")
    logger.info(f"Material Friction Angle (β): {FRICTION_ANGLE}°")
    logger.info(f"Material Dilation Angle (ψ): {DILATION_ANGLE}° ({'ASSOCIATED' if DILATION_ANGLE==FRICTION_ANGLE else 'NON-ASSOCIATED'} FLOW)")

    logger.info(f"Calculated True Cohesion (d): {cohesion_d.item():.4f}")

    # Expected uniaxial yield stress is exactly the input σc (by construction)
    expected_yield_stress = sigma_c_yield
    logger.info(f"Theoretical uniaxial yield stress ({STRESS_AXIS_LABEL}): {expected_yield_stress.item():.4f}")

    domain = setup_domain(GeometryFile, BoundingBox)
    if domain is None:
        logger.error('Failed to load domain; aborting simulation.')
        return

    ref_file = 'Uniaxial_Compression_Cylinder_Softening'
    KINEMATIC = False
    FlowStress = FlowStressDP
    HardeningModulus = HardeningModulusDP

    # Displacement-controlled compression along Z, matching AbaqusVerify/main.inp.
    start_disp = 0.0
    end_disp = float(os.environ.get('DP_END_DISP', '-0.01'))
    n_steps = int(os.environ.get('DP_N_STEPS', '20'))
    max_steps = os.environ.get('DP_MAX_STEPS', '').strip()
    disp_schedule = np.linspace(start_disp, end_disp, n_steps + 1).tolist()
    if max_steps:
        try:
            keep_steps = max(1, min(int(max_steps), n_steps))
            disp_schedule = disp_schedule[:keep_steps + 1]
            logger.info(f"DP_MAX_STEPS={keep_steps}: truncating displacement schedule for diagnostic run")
        except ValueError:
            logger.warning(f"Ignoring invalid DP_MAX_STEPS={max_steps!r}")
    # Optional: inject extra displacement checkpoints via env var (comma-separated), e.g. DP_EXTRA_DISP="0.0005,0.00075"
    extra_disp_env = os.environ.get('DP_EXTRA_DISP', '').strip()
    if extra_disp_env:
        try:
            extras = [float(s) for s in extra_disp_env.split(',') if s.strip()]
            disp_schedule = sorted(set(disp_schedule + extras), reverse=(end_disp < start_disp))
        except Exception as e:
            logger.warning(f"Failed to parse DP_EXTRA_DISP ('{extra_disp_env}'): {e}")
    logger.info(
        f"Displacement schedule: start={start_disp:.6e}, end={end_disp:.6e}, "
        f"steps={len(disp_schedule) - 1}, first_step={disp_schedule[1] - disp_schedule[0]:.6e}"
    )

    # Estimate number of steps until yield and plastic region
    # Yield displacement ε_y ≈ σ_c/E
    yield_disp = sigma_c_yield.item() / YM.item()
    step_size = abs(end_disp - start_disp) / max(n_steps, 1)
    num_elastic_steps = int(np.ceil(yield_disp / max(step_size, 1e-12)))
    # Ensure at least one yield step
    num_yield_steps = max((len(disp_schedule) - 1) - num_elastic_steps, 1)
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
    # Main L-BFGS refinement after Adam. The network remains float32 on CUDA;
    # constitutive/energy calculations remain float64 inside the solver.
    LBFGS_Iteration = int(os.environ.get('DP_LBFGS_ITERS', '500'))
    Num_Newton_itr = 100
    Settings = [
        KINEMATIC, FlowStress, HardeningModulus, disp_schedule, rel_tol,
        step_max, LBFGS_Iteration, Num_Newton_itr, EXAMPLE, YM, PR,
        cohesion_d, FRICTION_ANGLE, DILATION_ANGLE, TAN_BETA, SIN_BETA,
        COS_BETA, TAN_PSI, base, UNIFORM
    ]

    # >>>>> CRITICAL FIX D: Reduce Learning Rate <<<<<
    # Reduced LR from 0.01 to 1e-5 to stabilize Adam optimization for plastic steps.
    # Runtime controls for ablation/sensitivity studies.
    x_var = {
        'x_lr': float(os.environ.get('DP_LR', '1e-5')),
        'neuron': int(os.environ.get('DP_NEURON', '384')),
        'act_func': os.environ.get('DP_ACT', 'silu').strip().lower(),
    }
    # Original: x_var = {'x_lr': 0.01, ...}
    # >>>>> END OF FIX D <<<<<
    
    # Keep the separate post-refinement phases off; the main L-BFGS pass above
    # is the controlled precision step used after Adam.
    refine_lbfgs_iters = 0
    refine_adam_epochs = 0
    
    lr = x_var['x_lr']
    H = int(x_var['neuron'])
    act_fn = x_var['act_func']
    logger.info(f'LR: {lr}, H: {H}, act fn: {act_fn}')
    logger.info(f'L-BFGS iterations: {LBFGS_Iteration}')
    logger.info(f"DP_TANGENT_MODE: {os.environ.get('DP_TANGENT_MODE', 'consistent').strip().lower()}")

    diff_log_path = os.path.join(base, 'DiffLog')
    with open(diff_log_path, 'w', encoding='utf-8'):
        pass

    # The current softening benchmark is the AbaqusVerify cylinder, not the
    # plate-with-hole case, so use plain normalized (x,y,z) features.
    net_input_dim = 3
    logger.info(f"S_Net input_dim={net_input_dim}")
    snet = S_Net(net_input_dim, H, 3, act_fn)
    # PENALTY CONFIGURATION per RevisionIdea.md Section D:
    # - Sobolev and lateral traction: disabled (implementation issues)
    # - Equilibrium penalty: False here means "not forced on", but will be DYNAMICALLY ACTIVATED
    #   once history plasticity is detected (max(PEEQ) > history_plastic_gate_tol)
    #   via the _equilibrium_penalty_on flag in DeepMixedMethod.train_model().
    # This implements RevisionIdea.md recommendation: "Put back a minimal equilibrium penalty 
    # only in plastic steps" to prevent optimizer drift to low-σ, non-equilibrated states.
    # 
    # Keep the assembled virtual-work residual mechanically active in plastic
    # steps.  It is gradient-scaled in DeepMixedMethod so it enforces weak
    # equilibrium without returning to the previous over-stiff branch.
    asm_weight_elastic = float(os.environ.get('DP_ASM_WEIGHT_ELASTIC', '1e-2'))
    asm_weight_plastic = float(os.environ.get('DP_ASM_WEIGHT_PLASTIC', '1.0'))
    asm_weight_first_plastic = float(os.environ.get('DP_ASM_WEIGHT_FIRST_PLASTIC', str(asm_weight_elastic)))
    penalty_cfg = {
        'enable_sobolev': False,
        'enable_equilibrium': False,  # Dynamic activation in plastic regime (see above)
        'enable_lateral_traction': False,
        'load_axis': LOAD_AXIS.lower(),
        # Keep elastic-phase residual supportive, then tighten equilibrium once plasticity is detected
        'equilibrium_weight': 5e-2,
        'equilibrium_weight_plastic': 1e-1,
        'assembled_equilibrium_weight': asm_weight_elastic,
        'assembled_equilibrium_weight_plastic': asm_weight_plastic,
        'assembled_equilibrium_weight_first_plastic': asm_weight_first_plastic,
        'assembled_equilibrium_target_grad_ratio': float(os.environ.get('DP_ASM_TARGET_GRAD_RATIO', '2.0')),
        'assembled_equilibrium_weight_min': float(os.environ.get('DP_ASM_WEIGHT_MIN', '1e-2')),
        'assembled_equilibrium_weight_max': float(os.environ.get('DP_ASM_WEIGHT_MAX', '5.0')),
        # Non-associated DP softening is not governed by a scalar energy
        # minimum.  Plastic dissipation can be enabled as a diagnostic
        # regularizer, but the default must not penalize PEEQ growth and delay
        # softening relative to the Abaqus virtual-work solution.
        'softening_dissipation_weight': float(os.environ.get('DP_SOFTENING_DISS_WEIGHT', '0.0')),
        'softening_dissipation_weight_first_plastic': float(os.environ.get('DP_SOFTENING_DISS_WEIGHT_FIRST_PLASTIC', '0.0')),
        # Small branch-continuation regularizer: keeps the neural solution close
        # to the previous converged field extrapolated to the current displacement.
        # This is only a path-selection/trust-region term for strain-softening,
        # not a replacement for virtual-work equilibrium.
        'continuation_weight_first_plastic': float(os.environ.get('DP_CONTINUATION_WEIGHT_FIRST_PLASTIC', '5e-3')),
        'continuation_weight_plastic': float(os.environ.get('DP_CONTINUATION_WEIGHT_PLASTIC', '2e-3')),
        # First plastic loading must remain incrementally local.  The predictor
        # trust band prevents both frozen PEEQ (over-stiff branch) and O(1e-3)
        # overshoots (over-softened branch) without fitting Abaqus forces.
        'peeq_increment_trust_weight_first_plastic': float(os.environ.get('DP_PEEQ_TRUST_WEIGHT_FIRST_PLASTIC', '1.0')),
        'peeq_increment_trust_weight_plastic': float(os.environ.get('DP_PEEQ_TRUST_WEIGHT_PLASTIC', '1.0')),
        'peeq_increment_trust_cap_factor': float(os.environ.get('DP_PEEQ_TRUST_CAP_FACTOR', '2.0')),
        'peeq_increment_trust_min_cap': float(os.environ.get('DP_PEEQ_TRUST_MIN_CAP', '5e-5')),
        'peeq_increment_trust_lower_factor': float(os.environ.get('DP_PEEQ_TRUST_LOWER_FACTOR', '0.25')),
        # In non-associated softening there is no global scalar potential for
        # fully developed plastic flow.  The first plastic transition still
        # needs the elastic energy path to select the correct Dirichlet branch.
        'internal_energy_weight_elastic': float(os.environ.get('DP_INTERNAL_WEIGHT_ELASTIC', '1.0')),
        'internal_energy_weight_first_plastic': float(os.environ.get('DP_INTERNAL_WEIGHT_FIRST_PLASTIC', '1.0')),
        'internal_energy_weight_plastic': float(os.environ.get('DP_INTERNAL_WEIGHT_PLASTIC', '0.0')),
        # Plate-with-hole plasticity is localized.  A p95-only gate waits until
        # >5% of Gauss points are plastic and delays softening unrealistically.
        'distributed_plastic_rho_tol': float(os.environ.get('DP_DISTRIBUTED_PLASTIC_RHO_TOL', '5e-3')),
        'distributed_plastic_p95_tol': float(os.environ.get('DP_DISTRIBUTED_PLASTIC_P95_TOL', '1e-5')),
        'distributed_plastic_max_tol': float(os.environ.get('DP_DISTRIBUTED_PLASTIC_MAX_TOL', '1e-4')),
    }
    logger.info(
        "[Softening loss config] "
        f"asm_elastic={penalty_cfg['assembled_equilibrium_weight']:.3e}, "
        f"asm_plastic={penalty_cfg['assembled_equilibrium_weight_plastic']:.3e}, "
        f"asm_first_plastic={penalty_cfg['assembled_equilibrium_weight_first_plastic']:.3e}, "
        f"dissipation_weight={penalty_cfg['softening_dissipation_weight']:.3e}, "
        f"dissipation_first_plastic={penalty_cfg['softening_dissipation_weight_first_plastic']:.3e}, "
        f"continuation_first={penalty_cfg['continuation_weight_first_plastic']:.3e}, "
        f"continuation_plastic={penalty_cfg['continuation_weight_plastic']:.3e}, "
        f"peeq_trust_first={penalty_cfg['peeq_increment_trust_weight_first_plastic']:.3e}, "
        f"peeq_trust_plastic={penalty_cfg['peeq_increment_trust_weight_plastic']:.3e}, "
        f"peeq_trust_cap_factor={penalty_cfg['peeq_increment_trust_cap_factor']:.3e}, "
        f"peeq_trust_min_cap={penalty_cfg['peeq_increment_trust_min_cap']:.3e}, "
        f"peeq_trust_lower_factor={penalty_cfg['peeq_increment_trust_lower_factor']:.3e}, "
        f"internal_first_plastic={penalty_cfg['internal_energy_weight_first_plastic']:.3e}, "
        f"internal_plastic={penalty_cfg['internal_energy_weight_plastic']:.3e}, "
        f"rho_gate={penalty_cfg['distributed_plastic_rho_tol']:.3e}, "
        f"p95_gate={penalty_cfg['distributed_plastic_p95_tol']:.3e}, "
        f"max_gate={penalty_cfg['distributed_plastic_max_tol']:.3e}"
    )
    DEM = DeepMixedMethod([snet, lr, domain, Settings, penalty_cfg])
    DEM.skip_lbfgs = LBFGS_Iteration <= 0
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
