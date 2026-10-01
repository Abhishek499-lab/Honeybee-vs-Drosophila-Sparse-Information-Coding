
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


# ================================================================================
# 11. WRITE-BACK SANITY TEST
# ================================================================================

print("\n" + "=" * 80)
print("WRITE-BACK SANITY TEST")
print("=" * 80)

test = train_condition(
    architecture="HONEYBEE",
    rule="DROSOPHILA",
    network_id=0
)

test_pnkc_change = np.abs(
    test["W_delta"]
).max()

test_kcen_change = np.abs(
    test["KCEN_delta"]
).max()

print(
    "Max PN→KC change:",
    test_pnkc_change
)

print(
    "Max KC→EN change:",
    test_kcen_change
)

assert test_pnkc_change > 0
assert test_kcen_change > 0

print("PN→KC write-back: PASS")
print("KC→EN plasticity: PASS")

# ================================================================================
# 12. FULL EXPERIMENT
# ================================================================================

print("\n" + "=" * 80)
print("FULL 2×2 EXPERIMENT")
print("=" * 80)

all_metrics = []
all_profiles = []
all_representation = []
all_training = []

# ------------------------------------------------------------------------------
# IMPORTANT MEMORY NOTE
#
# We do NOT keep all 400 full PN→KC matrices simultaneously in RAM.
# Each condition/network matrix is saved immediately into a compressed NPZ file.
# ------------------------------------------------------------------------------

matrix_file = (
    OUT_PREFIX +
    "_Synaptic_Matrices.npz"
)

# ------------------------------------------------------------------------------
# Use a temporary dictionary for compressed NPZ.
# 400 × ~1.6M float32 entries would be large but still manageable compressed.
# ------------------------------------------------------------------------------

matrix_store = {}

# ================================================================================
# RUN CONDITIONS
# ================================================================================

for condition_id, condition in enumerate(
    CONDITIONS,
    start=1
):

    architecture, rule = condition.split(
        "__"
    )

    print("\n" + "-" * 80)
    print(
        f"CONDITION {condition_id}/4:",
        condition
    )
    print("-" * 80)

    for network in range(
        N_NETWORKS
    ):

        result = train_condition(
            architecture=architecture,
            rule=rule,
            network_id=network
        )

        # ----------------------------------------------------------------------
        # Profile
        # ----------------------------------------------------------------------

        profile_df = pd.DataFrame(
            result["profile"]
        )

        bv = profile_df[
            "behavioral_valence"
        ].values

        csplus_bv = float(
            bv[CSPLUS_IDX]
        )

        csminus_bv = float(
            bv[CSMINUS_IDX]
        )

        peak_idx = int(
            np.argmax(bv)
        )

        peak_pattern = int(
            patterns[peak_idx]
        )

        peak_behavior = float(
            bv[peak_idx]
        )

        discrimination = (
            csplus_bv -
            csminus_bv
        )

        response_range = (
            float(bv.max()) -
            float(bv.min())
        )

        d_plus = abs(
            peak_pattern -
            CSPLUS_PATTERN
        )

        d_minus = abs(
            peak_pattern -
            CSMINUS_PATTERN
        )

        peak_advantage = (
            d_minus -
            d_plus
        )

        both_correct = int(
            (
                csplus_bv > 0
            )
            and
            (
                csminus_bv < 0
            )
        )

        # ----------------------------------------------------------------------
        # Network metrics
        # ----------------------------------------------------------------------

        metric_row = {

            "network":
                network,

            "condition":
                condition,

            "architecture":
                architecture,

            "rule":
                rule,

            "CSplus_behavior":
                csplus_bv,

            "CSminus_behavior":
                csminus_bv,

            "discrimination":
                discrimination,

            "response_range":
                response_range,

            "peak_stimulus":
                peak_pattern,

            "peak_behavior":
                peak_behavior,

            "distance_peak_to_CSplus":
                d_plus,

            "distance_peak_to_CSminus":
                d_minus,

            "peak_advantage":
                peak_advantage,

            "both_correct":
                both_correct,

            **result["pnkc_summary"],

            **result["kcen_summary"]
        }

        all_metrics.append(
            metric_row
        )

        # ----------------------------------------------------------------------
        # Profiles
        # ----------------------------------------------------------------------

        all_profiles.extend(
            result["profile"]
        )

        # ----------------------------------------------------------------------
        # KC representation
        # ----------------------------------------------------------------------

        all_representation.extend(
            result["representation"]
        )

        # ----------------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------------

        all_training.extend(
            result["training"]
        )

        # ----------------------------------------------------------------------
        # Mechanistic matrices
        # ----------------------------------------------------------------------

        key = (
            f"{condition}"
            f"_network_{network:03d}"
        )

        matrix_store[
            key + "__W_initial"
        ] = result["W_initial"]

        matrix_store[
            key + "__W_final"
        ] = result["W_final"]

        matrix_store[
            key + "__W_delta"
        ] = result["W_delta"]

        matrix_store[
            key + "__KCEN_initial"
        ] = result["KCEN_initial"]

        matrix_store[
            key + "__KCEN_final"
        ] = result["KCEN_final"]

        matrix_store[
            key + "__KCEN_delta"
        ] = result["KCEN_delta"]

        # ----------------------------------------------------------------------
        # Progress
        # ----------------------------------------------------------------------

        if (
            (network + 1) % 10 == 0
            or
            network == 0
        ):

            print(
                f"Network {network + 1:03d}/100 complete"
            )

    print(
        "Condition complete:",
        condition
    )

# ================================================================================
# 13. DATAFRAMES
# ================================================================================

metrics_df = pd.DataFrame(
    all_metrics
)

profiles_df = pd.DataFrame(
    all_profiles
)

representation_df = pd.DataFrame(
    all_representation
)

training_df = pd.DataFrame(
    all_training
)

print("\n" + "=" * 80)
print("DATASET AUDIT")
print("=" * 80)

print(
    "Network metrics:",
    metrics_df.shape
)

print(
    "Profiles:",
    profiles_df.shape
)

print(
    "KC representation:",
    representation_df.shape
)

print(
    "Training trajectory:",
    training_df.shape
)

assert len(metrics_df) == 400
assert len(profiles_df) == 40000
assert len(representation_df) == 40000
assert len(training_df) == 8000

assert (
    metrics_df["condition"].nunique()
    == 4
)

print("DATASET AUDIT: PASS")

# ================================================================================
# 14. SAVE CSVs
# ================================================================================

metrics_file = (
    OUT_PREFIX +
    "_Network_Metrics.csv"
)

profiles_file = (
    OUT_PREFIX +
    "_Profiles.csv"
)

representation_file = (
    OUT_PREFIX +
    "_KC_Representation.csv"
)

training_file = (
    OUT_PREFIX +
    "_Training_Trajectory.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

profiles_df.to_csv(
    profiles_file,
    index=False
)

representation_df.to_csv(
    representation_file,
    index=False
)

training_df.to_csv(
    training_file,
    index=False
)

print("\nCSV files saved.")

# ================================================================================
# 15. SAVE MATRICES
# ================================================================================

print("\nSaving compressed mechanistic matrices...")

np.savez_compressed(
    matrix_file,
    **matrix_store
)

print(
    "Matrix file saved:",
    matrix_file
)

# Free RAM
del matrix_store
gc.collect()

# ================================================================================
# 16. CONDITION SUMMARY
# ================================================================================

summary = (
    metrics_df
    .groupby("condition")
    .agg({

        "CSplus_behavior":
            "mean",

        "CSminus_behavior":
            "mean",

        "discrimination":
            "mean",

        "response_range":
            "mean",

        "peak_stimulus":
            "mean",

        "peak_behavior":
            "mean",

        "distance_peak_to_CSplus":
            "mean",

        "distance_peak_to_CSminus":
            "mean",

        "peak_advantage":
            "mean",

        "both_correct":
            "mean",

        "PNKC_mean_delta":
            "mean",

        "PNKC_mean_abs_delta":
            "mean",

        "PNKC_positive_fraction":
            "mean",

        "PNKC_negative_fraction":
            "mean",

        "PNKC_changed_fraction":
            "mean",

        "KCEN_EPLUS_mean_delta":
            "mean",

        "KCEN_EMINUS_mean_delta":
            "mean",

        "KCEN_EPLUS_mean_abs_delta":
            "mean",

        "KCEN_EMINUS_mean_abs_delta":
            "mean"
    })
    .reset_index()
)

summary_file = (
    OUT_PREFIX +
    "_Condition_Summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("\n" + "=" * 80)
print("CONDITION SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}"
    )
)

# ================================================================================
# 17. MECHANISTIC FACTORIAL DECOMPOSITION
# ================================================================================

print("\n" + "=" * 80)
print("MECHANISTIC FACTORIAL DECOMPOSITION")
print("=" * 80)

wide = metrics_df.pivot(
    index="network",
    columns="condition"
)

mechanistic_metrics = [

    "PNKC_mean_delta",
    "PNKC_mean_abs_delta",
    "PNKC_positive_fraction",
    "PNKC_negative_fraction",
    "PNKC_changed_fraction",

    "KCEN_EPLUS_mean_delta",
    "KCEN_EMINUS_mean_delta",
    "KCEN_EPLUS_mean_abs_delta",
    "KCEN_EMINUS_mean_abs_delta"
]

factor_rows = []

for metric in mechanistic_metrics:

    hb_bee = wide[
        metric
    ][
        "HONEYBEE__BEE"
    ].values

    hb_dopa = wide[
        metric
    ][
        "HONEYBEE__DROSOPHILA"
    ].values

    dt_bee = wide[
        metric
    ][
        "DROSOPHILA_TRANSFER__BEE"
    ].values

    dt_dopa = wide[
        metric
    ][
        "DROSOPHILA_TRANSFER__DROSOPHILA"
    ].values

    # Rule effects
    rule_hb = (
        hb_dopa -
        hb_bee
    )

    rule_dt = (
        dt_dopa -
        dt_bee
    )

    # Architecture effects
    arch_bee = (
        dt_bee -
        hb_bee
    )

    arch_dopa = (
        dt_dopa -
        hb_dopa
    )

    # Architecture × rule interaction
    interaction = (
        rule_dt -
        rule_hb
    )

    factor_rows.append({

        "metric":
            metric,

        "rule_effect_HB":
            float(rule_hb.mean()),

        "rule_effect_TRANSFER":
            float(rule_dt.mean()),

        "architecture_effect_BEE":
            float(arch_bee.mean()),

        "architecture_effect_DROSOPHILA":
            float(arch_dopa.mean()),

        "architecture_x_rule":
            float(interaction.mean()),

        "interaction_SD":
            float(
                interaction.std(
                    ddof=1
                )
            ),

        "interaction_paired_t_p":
            float(
                ttest_rel(
                    rule_dt,
                    rule_hb
                ).pvalue
            )
    })

mechanistic_factorial = pd.DataFrame(
    factor_rows
)

mechanistic_factorial_file = (
    OUT_PREFIX +
    "_Mechanistic_Factorial_Decomposition.csv"
)

mechanistic_factorial.to_csv(
    mechanistic_factorial_file,
    index=False
)

print(
    mechanistic_factorial.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6g}"
    )
)

# ================================================================================
# 18. KC REPRESENTATION SUMMARY
# ================================================================================

print("\n" + "=" * 80)
print("KC REPRESENTATION SUMMARY")
print("=" * 80)

rep_summary = (
    representation_df
    .groupby("condition")
    .agg({

        "overlap_CSplus":
            "mean",

        "overlap_CSminus":
            "mean",

        "active_KC_count":
            "mean"
    })
    .reset_index()
)

print(
    rep_summary.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}"
    )
)

rep_summary_file = (
    OUT_PREFIX +
    "_KC_Representation_Summary.csv"
)

rep_summary.to_csv(
    rep_summary_file,
    index=False
)

# ================================================================================
# 19. MECHANISM → BEHAVIOUR ASSOCIATIONS
# ================================================================================

print("\n" + "=" * 80)
print("MECHANISM → BEHAVIOUR ASSOCIATIONS")
print("=" * 80)

predictors = [

    "PNKC_mean_delta",
    "PNKC_mean_abs_delta",
    "PNKC_positive_fraction",
    "PNKC_negative_fraction",
    "PNKC_changed_fraction",

    "KCEN_EPLUS_mean_delta",
    "KCEN_EMINUS_mean_delta",
    "KCEN_EPLUS_mean_abs_delta",
    "KCEN_EMINUS_mean_abs_delta"
]

association_rows = []

for condition in CONDITIONS:

    sub = metrics_df[
        metrics_df["condition"]
        == condition
    ].copy()

    y = sub[
        "discrimination"
    ].values

    for predictor in predictors:

        x = sub[
            predictor
        ].values

        if np.std(x) == 0:
            continue

        pearson_r, pearson_p = pearsonr(
            x,
            y
        )

        spearman_r, spearman_p = spearmanr(
            x,
            y
        )

        association_rows.append({

            "condition":
                condition,

            "predictor":
                predictor,

            "outcome":
                "discrimination",

            "pearson_r":
                pearson_r,

            "pearson_p":
                pearson_p,

            "spearman_r":
                spearman_r,

            "spearman_p":
                spearman_p
        })

association_df = pd.DataFrame(
    association_rows
)

association_file = (
    OUT_PREFIX +
    "_Mechanism_Behaviour_Associations.csv"
)

association_df.to_csv(
    association_file,
    index=False
)

print(
    association_df.sort_values(
        "pearson_p"
    ).head(20).to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6g}"
    )
)

# ================================================================================
# 20. FINAL AUDIT
# ================================================================================

print("\n" + "=" * 80)
print("v13.3E FINAL AUDIT")
print("=" * 80)

checks = {

    "400 network metrics":
        len(metrics_df) == 400,

    "40000 profiles":
        len(profiles_df) == 40000,

    "40000 representation rows":
        len(representation_df) == 40000,

    "8000 training rows":
        len(training_df) == 8000,

    "4 conditions":
        metrics_df[
            "condition"
        ].nunique() == 4,

    "PN-KC remodeling detected":
        metrics_df[
            "PNKC_changed_fraction"
        ].mean() > 0,

    "KC-EN remodeling detected":
        (
            metrics_df[
                [
                    "KCEN_EPLUS_mean_abs_delta",
                    "KCEN_EMINUS_mean_abs_delta"
                ]
            ].values.mean()
            > 0
        ),

    "finite metrics":
        np.isfinite(
            metrics_df.select_dtypes(
                include=np.number
            ).values
        ).all()
}

for name, passed in checks.items():

    print(
        f"{name:<38}:",
        "PASS" if passed else "FAIL"
    )

assert all(
    checks.values()
)

# ================================================================================
# 21. OUTPUT LIST
# ================================================================================

print("\n" + "=" * 80)
print("SAVED FILES")
print("=" * 80)

for f in [

    metrics_file,

    profiles_file,

    representation_file,

    training_file,

    matrix_file,

    summary_file,

    mechanistic_factorial_file,

    rep_summary_file,

    association_file

]:

    print(f)

print("\n" + "=" * 80)
print("OVERALL: PASS")
print("v13.3E MECHANISTIC INSTRUMENTATION COMPLETE")
print("=" * 80)