"""
Honeybee–Drosophila Computational Model
=======================================

Clean source derived from:

    Untitled98.ipynb
    Cell 64
    Original model-core lines 1–858

This file preserves the operational model logic of
the original v13.3E implementation.

IMPORTANT:
-----------
This is a computational model. The Honeybee and
Drosophila-transfer architectures and learning rules
represent model assumptions/operational implementations;
they should not automatically be interpreted as
experimentally established biological mechanisms.

The untouched source is preserved separately in:

    v13_3E_ORIGINAL_CELL64_FULL.py
    v13_3E_ORIGINAL_MODEL_CORE_1_858.py

Do not modify the original provenance files.
"""


# ================================================================================
# v13.3E — MECHANISTIC INSTRUMENTATION
# ================================================================================
# GOAL:
#   v13.3D ke behavioural architecture × rule interaction ko
#   synaptic + KC-representation level par investigate karna.
#
# 2×2:
#   HONEYBEE__BEE
#   HONEYBEE__DROSOPHILA
#   DROSOPHILA_TRANSFER__BEE
#   DROSOPHILA_TRANSFER__DROSOPHILA
#
# SAVES:
#   1. Behavioural metrics
#   2. Behavioural profiles
#   3. PN→KC mechanistic summaries
#   4. KC→EN mechanistic summaries
#   5. KC representation summaries
#   6. Training trajectory
#   7. Full synaptic matrices in compressed NPZ
#
# IMPORTANT:
#   - Google Drive mounted
#   - Same continuum
#   - Same 100 networks
#   - Same 20 trials
#   - Same 2×2 design
#   - Correct PN→KC WRITE-BACK preserved
# ================================================================================

# ================================================================================
# 0. GOOGLE DRIVE MOUNT
# ================================================================================

from google.colab import drive

drive.mount('/content/drive')

print("Google Drive mounted successfully.")

# ================================================================================
# 1. IMPORTS
# ================================================================================

import os
import gc
import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    ttest_rel,
    wilcoxon
)

# ================================================================================
# 2. PATHS
# ================================================================================

BASE = "/content/drive/MyDrive/Honey"

CONTINUUM_FILE = os.path.join(
    BASE,
    "Honeybee_Model_v10_3_Validated_Continuum.csv"
)

OUT_PREFIX = os.path.join(
    BASE,
    "Honeybee_Model_v13_3E_Mechanistic"
)

os.makedirs(BASE, exist_ok=True)

print("\nBASE:")
print(BASE)

print("\nContinuum:")
print(CONTINUUM_FILE)

# ================================================================================
# 3. LOAD EXACT CONTINUUM
# ================================================================================

cont = pd.read_csv(CONTINUUM_FILE)

PN_COLS = [
    f"PN_{i}"
    for i in range(1, 101)
]

X = cont[PN_COLS].values.astype(np.int8)

patterns = cont["pattern"].values.astype(int)

assert X.shape == (100, 100)
assert np.all(X.sum(axis=1) == 50)

CSPLUS_PATTERN = 51
CSMINUS_PATTERN = 65

CSPLUS_IDX = int(
    np.where(patterns == CSPLUS_PATTERN)[0][0]
)

CSMINUS_IDX = int(
    np.where(patterns == CSMINUS_PATTERN)[0][0]
)

assert CSPLUS_IDX == 50
assert CSMINUS_IDX == 64

# True Hamming distance
hamming_cs = int(
    np.sum(
        X[CSPLUS_IDX] != X[CSMINUS_IDX]
    )
)

assert hamming_cs == 28

print("\n" + "=" * 80)
print("CONTINUUM AUDIT")
print("=" * 80)

print("Shape:", X.shape)
print("All stimuli have 50 active PNs:", np.all(X.sum(axis=1) == 50))
print("CS+:", CSPLUS_PATTERN, "index:", CSPLUS_IDX)
print("CS-:", CSMINUS_PATTERN, "index:", CSMINUS_IDX)
print("True CS+ ↔ CS- Hamming:", hamming_cs)

print("CONTINUUM AUDIT: PASS")

# ================================================================================
# 4. MODEL CONSTANTS
# ================================================================================

N_PN = 100
N_KC = 4000

ACTIVE_KC_FRACTION = 0.05
N_ACTIVE_KC = int(
    N_KC * ACTIVE_KC_FRACTION
)

W_INITIAL_VALUE = 0.2

W_MIN = 0.0
W_MAX = 0.4

N_TRIALS = 20
N_NETWORKS = 100

SEED_BASE = 133300

# ================================================================================
# 5. CONDITIONS
# ================================================================================

CONDITIONS = [
    "HONEYBEE__BEE",
    "HONEYBEE__DROSOPHILA",
    "DROSOPHILA_TRANSFER__BEE",
    "DROSOPHILA_TRANSFER__DROSOPHILA"
]

# ================================================================================
# 6. TRAINING SCHEDULE
# ================================================================================

# Alternating:
# R P R P R P ...
#
# R = CS+ / reward
# P = CS- / punishment

schedule = np.array(
    [
        1 if t % 2 == 0 else -1
        for t in range(N_TRIALS)
    ],
    dtype=np.int8
)

assert np.sum(schedule == 1) == 10
assert np.sum(schedule == -1) == 10

print("\nTraining schedule:")
print(
    "".join(
        "R" if s == 1 else "P"
        for s in schedule
    )
)

# ================================================================================
# 7. LEARNING-RULE PARAMETERS
# ================================================================================

# ------------------------------------------------------------------------------
# Honeybee-style
# ------------------------------------------------------------------------------

BEE_PNKC_PLUS = 0.006
BEE_PNKC_MINUS = 0.007

BEE_KCEN_PLUS = 0.006
BEE_KCEN_MINUS = 0.008

# ------------------------------------------------------------------------------
# Drosophila-inspired
# ------------------------------------------------------------------------------

DOPA_PNKC_PLUS = 0.004
DOPA_PNKC_MINUS = 0.002

DOPA_KCEN_ETA = 0.006

# ================================================================================
# 8. ARCHITECTURE GENERATOR
# ================================================================================

def make_architecture(
    seed,
    architecture
):

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------------------------
    # Honeybee architecture
    # --------------------------------------------------------------------------

    if architecture == "HONEYBEE":

        # 5–15 PN inputs per KC
        n_inputs = rng.integers(
            5,
            16,
            size=N_KC
        )

    # --------------------------------------------------------------------------
    # Drosophila-transfer operational architecture
    # --------------------------------------------------------------------------

    elif architecture == "DROSOPHILA_TRANSFER":

        # Operational transplant used in v13.3
        # fixed 10 PN inputs per KC
        n_inputs = np.full(
            N_KC,
            10,
            dtype=np.int8
        )

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    W = np.zeros(
        (N_KC, N_PN),
        dtype=np.float32
    )

    for kc in range(N_KC):

        pn_idx = rng.choice(
            N_PN,
            size=int(n_inputs[kc]),
            replace=False
        )

        W[
            kc,
            pn_idx
        ] = W_INITIAL_VALUE

    return W, n_inputs

# ================================================================================
# 9. KC ACTIVATION
# ================================================================================

def activate_kcs(
    W,
    pn_pattern
):

    kc_input = W @ pn_pattern

    # Select exactly 5% most strongly driven KCs
    active_idx = np.argpartition(
        kc_input,
        -N_ACTIVE_KC
    )[-N_ACTIVE_KC:]

    active_idx = np.sort(
        active_idx
    )

    return (
        active_idx,
        kc_input
    )

# ================================================================================
# 10. TRAINING FUNCTION
# ================================================================================

def train_condition(
    architecture,
    rule,
    network_id
):

    # --------------------------------------------------------------------------
    # Architecture
    # --------------------------------------------------------------------------

    seed = (
        SEED_BASE +
        network_id
    )

    W, kc_input_counts = make_architecture(
        seed,
        architecture
    )

    W_initial = W.copy()

    # --------------------------------------------------------------------------
    # KC → EN
    #
    # column 0 = EPLUS
    # column 1 = EMINUS
    # --------------------------------------------------------------------------

    KCEN = np.full(
        (N_KC, 2),
        W_INITIAL_VALUE,
        dtype=np.float32
    )

    KCEN_initial = KCEN.copy()

    # --------------------------------------------------------------------------
    # Training trajectory
    # --------------------------------------------------------------------------

    training_records = []

    # ==========================================================================
    # TRAINING
    # ==========================================================================

    for trial, valence in enumerate(
        schedule,
        start=1
    ):

        if valence == 1:
            stimulus = X[CSPLUS_IDX]

        else:
            stimulus = X[CSMINUS_IDX]

        active_idx, kc_input = activate_kcs(
            W,
            stimulus
        )

        # ======================================================================
        # HONEYBEE RULE
        # ======================================================================

        if rule == "BEE":

            # ------------------------------------------------------------------
            # CS+
            # ------------------------------------------------------------------

            if valence == 1:

                # CS+ depresses KC → EPLUS
                KCEN[
                    active_idx,
                    0
                ] -= BEE_KCEN_PLUS

                KCEN[
                    active_idx,
                    0
                ] = np.clip(
                    KCEN[
                        active_idx,
                        0
                    ],
                    W_MIN,
                    W_MAX
                )

            # ------------------------------------------------------------------
            # CS-
            # ------------------------------------------------------------------

            else:

                # CS- depresses KC → EMINUS
                KCEN[
                    active_idx,
                    1
                ] -= BEE_KCEN_MINUS

                KCEN[
                    active_idx,
                    1
                ] = np.clip(
                    KCEN[
                        active_idx,
                        1
                    ],
                    W_MIN,
                    W_MAX
                )

            # ------------------------------------------------------------------
            # Published-style PN→KC remodeling used in this project
            #
            # CRITICAL:
            # NumPy advanced indexing returns a COPY.
            # Therefore we explicitly WRITE BACK to W.
            # ------------------------------------------------------------------

            current_rows = W[
                active_idx,
                :
            ].copy()

            active_pn_mask = (
                stimulus.astype(bool)
            )

            inactive_pn_mask = ~active_pn_mask

            current_rows[
                :,
                active_pn_mask
            ] += BEE_PNKC_PLUS

            current_rows[
                :,
                inactive_pn_mask
            ] -= (
                0.5 *
                BEE_PNKC_MINUS
            )

            current_rows = np.clip(
                current_rows,
                W_MIN,
                W_MAX
            )

            # CRITICAL WRITE-BACK
            W[
                active_idx,
                :
            ] = current_rows

        # ======================================================================
        # DROSOPHILA-INSPIRED RULE
        # ======================================================================

        elif rule == "DROSOPHILA":

            dopamine = float(
                valence
            )

            # Reward:
            # depress EMINUS
            #
            # Punishment:
            # depress EPLUS

            target_col = (
                1 if dopamine > 0
                else 0
            )

            KCEN[
                active_idx,
                target_col
            ] -= (
                DOPA_KCEN_ETA *
                abs(dopamine)
            )

            KCEN[
                active_idx,
                target_col
            ] = np.clip(
                KCEN[
                    active_idx,
                    target_col
                ],
                W_MIN,
                W_MAX
            )

            # ------------------------------------------------------------------
            # PN→KC remodeling
            # ------------------------------------------------------------------

            current_rows = W[
                active_idx,
                :
            ].copy()

            active_pn_mask = (
                stimulus.astype(bool)
            )

            inactive_pn_mask = ~active_pn_mask

            current_rows[
                :,
                active_pn_mask
            ] += (
                DOPA_PNKC_PLUS *
                abs(dopamine)
            )

            current_rows[
                :,
                inactive_pn_mask
            ] -= (
                DOPA_PNKC_MINUS *
                abs(dopamine)
            )

            current_rows = np.clip(
                current_rows,
                W_MIN,
                W_MAX
            )

            # CRITICAL WRITE-BACK
            W[
                active_idx,
                :
            ] = current_rows

        else:

            raise ValueError(
                f"Unknown rule: {rule}"
            )

        # ----------------------------------------------------------------------
        # Training-state summary
        # ----------------------------------------------------------------------

        training_records.append({

            "network": network_id,

            "condition":
                f"{architecture}__{rule}",

            "trial": trial,

            "valence": valence,

            "active_KC_count":
                len(active_idx),

            "mean_PNKC_weight":
                float(W.mean()),

            "mean_KCEN_EPLUS":
                float(KCEN[:, 0].mean()),

            "mean_KCEN_EMINUS":
                float(KCEN[:, 1].mean())
        })

    # ==========================================================================
    # FINAL SYNAPTIC CHANGES
    # ==========================================================================

    W_delta = (
        W -
        W_initial
    )

    KCEN_delta = (
        KCEN -
        KCEN_initial
    )

    # ==========================================================================
    # PN→KC MECHANISTIC SUMMARY
    # ==========================================================================

    pnkc_abs = np.abs(
        W_delta
    )

    pnkc_summary = {

        "PNKC_mean_delta":
            float(W_delta.mean()),

        "PNKC_mean_abs_delta":
            float(pnkc_abs.mean()),

        "PNKC_std_delta":
            float(W_delta.std()),

        "PNKC_positive_fraction":
            float(
                (W_delta > 1e-9).mean()
            ),

        "PNKC_negative_fraction":
            float(
                (W_delta < -1e-9).mean()
            ),

        "PNKC_changed_fraction":
            float(
                (pnkc_abs > 1e-9).mean()
            ),

        "PNKC_max_abs_delta":
            float(
                pnkc_abs.max()
            )
    }

    # ==========================================================================
    # KC→EN MECHANISTIC SUMMARY
    # ==========================================================================

    kcen_summary = {

        "KCEN_EPLUS_mean_delta":
            float(
                KCEN_delta[:, 0].mean()
            ),

        "KCEN_EMINUS_mean_delta":
            float(
                KCEN_delta[:, 1].mean()
            ),

        "KCEN_EPLUS_mean_abs_delta":
            float(
                np.abs(
                    KCEN_delta[:, 0]
                ).mean()
            ),

        "KCEN_EMINUS_mean_abs_delta":
            float(
                np.abs(
                    KCEN_delta[:, 1]
                ).mean()
            ),

        "KCEN_EPLUS_final_mean":
            float(
                KCEN[:, 0].mean()
            ),

        "KCEN_EMINUS_final_mean":
            float(
                KCEN[:, 1].mean()
            )
    }

    # ==========================================================================
    # FINAL KC REPRESENTATIONS
    # ==========================================================================

    csplus_active, _ = activate_kcs(
        W,
        X[CSPLUS_IDX]
    )

    csminus_active, _ = activate_kcs(
        W,
        X[CSMINUS_IDX]
    )

    csplus_set = set(
        csplus_active.tolist()
    )

    csminus_set = set(
        csminus_active.tolist()
    )

    representation_records = []

    profile_records = []

    for stim_idx in range(
        len(X)
    ):

        active_idx, kc_input = activate_kcs(
            W,
            X[stim_idx]
        )

        active_set = set(
            active_idx.tolist()
        )

        overlap_csplus = (
            len(
                active_set.intersection(
                    csplus_set
                )
            )
            /
            N_ACTIVE_KC
        )

        overlap_csminus = (
            len(
                active_set.intersection(
                    csminus_set
                )
            )
            /
            N_ACTIVE_KC
        )

        # ----------------------------------------------------------------------
        # Representation record
        # ----------------------------------------------------------------------

        representation_records.append({

            "network":
                network_id,

            "condition":
                f"{architecture}__{rule}",

            "pattern":
                int(patterns[stim_idx]),

            "stimulus_index":
                stim_idx,

            "overlap_CSplus":
                overlap_csplus,

            "overlap_CSminus":
                overlap_csminus,

            "active_KC_count":
                len(active_idx)
        })

        # ----------------------------------------------------------------------
        # Output response
        # ----------------------------------------------------------------------

        EPLUS = float(
            KCEN[
                active_idx,
                0
            ].sum()
        )

        EMINUS = float(
            KCEN[
                active_idx,
                1
            ].sum()
        )

        # Behavioural convention used in v13.3:
        BV = EPLUS - EMINUS

        profile_records.append({

            "network":
                network_id,

            "condition":
                f"{architecture}__{rule}",

            "pattern":
                int(patterns[stim_idx]),

            "stimulus_index":
                stim_idx,

            "EPLUS":
                EPLUS,

            "EMINUS":
                EMINUS,

            "behavioral_valence":
                BV
        })

    # ==========================================================================
    # RETURN
    # ==========================================================================

    return {

        "W_initial":
            W_initial,

        "W_final":
            W,

        "W_delta":
            W_delta,

        "KCEN_initial":
            KCEN_initial,

        "KCEN_final":
            KCEN,

        "KCEN_delta":
            KCEN_delta,

        "pnkc_summary":
            pnkc_summary,

        "kcen_summary":
            kcen_summary,

        "representation":
            representation_records,

        "profile":
            profile_records,

        "training":
            training_records
    }
