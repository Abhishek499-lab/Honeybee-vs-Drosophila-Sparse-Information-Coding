# Honeybee–Drosophila Computational Model

## Overview

This repository contains the computational model, source
provenance, validated stimulus data, model outputs, and
reproducibility audits associated with the Honeybee–Drosophila
computational study.

The repository preserves the computational workflow and
documents the provenance of the major inputs, model components,
outputs, and audits.

---

## Model architecture

The v13.3E model uses:

- 100 projection neurons (PNs)
- 4000 Kenyon cells (KCs)
- 5% KC activation
- 200 active KCs per stimulus
- Initial synaptic weight = 0.2
- Weight range = 0.0–0.4
- 20 alternating reward/punishment training trials
- 100 independently seeded networks per condition

Honeybee architecture uses variable PN→KC connectivity of
5–15 inputs per KC.

The Drosophila-transfer architecture uses fixed 10 PN inputs
per KC.

---

## v13.3E conditions

The main v13.3E factorial conditions are:

1. HONEYBEE__BEE
2. HONEYBEE__DROSOPHILA
3. DROSOPHILA_TRANSFER__BEE
4. DROSOPHILA_TRANSFER__DROSOPHILA

These conditions should not be confused with the historical
v12.2.16 mechanistic decomposition conditions.

---

## Historical mechanistic decomposition

The v12.2.16 response-profile analysis used four plasticity
conditions:

- NONE
- PN_KC_ONLY
- KC_EN_ONLY
- FULL

These represent:

- NONE: no plasticity
- PN_KC_ONLY: PN→KC plasticity only
- KC_EN_ONLY: KC→EN plasticity only
- FULL: both plasticity mechanisms

These historical conditions are distinct from the v13.3E
factorial species/architecture conditions and should not be
directly mapped onto the v13.3E NPZ network identifiers.

---

## Stimulus provenance

The validated stimulus continuum is:

data/Honeybee_Model_v10_3_Validated_Continuum.csv

It contains 100 stimuli.

The documented CS+ stimulus is pattern 51.

The documented CS− stimulus is pattern 65.

The underlying CS+/CS− pair has:

- 50 active PNs in each pattern
- overlap = 36 active PNs
- Hamming distance = 28
- Jaccard similarity = 0.5625

### Important clarification about the historical value 44

Historical analyses report 44 strict geometric candidates among
the 100 stimuli.

This value is a downstream geometric-candidate count.

It is NOT the Hamming distance between CS+ and CS−.

The repository therefore preserves both facts separately.

---

## PI provenance

The historical v12.2.16 PI profile was calculated from KC
representations and final KC→EN states.

The historical response-profile implementation is preserved under:

docs/provenance/historical_PI_function_and_response_source.txt

The historical four-condition PI profiles are retained under:

results/summary/

The historical PI files should not be directly merged with
the v13.3E network-level NPZ matrices because their condition
identifiers represent different experimental decompositions.

---

## Repository structure

Honeybee-Drosophila/
|
|-- README.md
|-- .gitignore
|
|-- code/
|   |-- model_core.py
|   `-- audits/
|
|-- docs/
|   |-- provenance/
|   |-- methods/
|   `-- audit/
|
|-- data/
|   |-- Honeybee_Model_v10_3_Validated_Continuum.csv
|   `-- README.md
|
`-- results/
    |-- summary/
    |-- audit/
    `-- matrices/

---

## Provenance

The original computational model implementation was recovered
from the original Colab notebook source and preserved separately
from the clean operational model core.

Relevant provenance files are stored in:

docs/provenance/

The repository distinguishes between:

1. Original source
2. Clean operational code
3. Historical outputs
4. Forensic audit outputs
5. Validated input data

---

## Reproducibility

The repository preserves the model implementation, validated
stimulus input, parameter definitions, historical outputs,
current model outputs, and major computational audits.

Large NPZ matrix files in results/matrices/ contain network-level
model states and are retained as computational outputs.

---

## Audit status

Completed audits include:

- stimulus provenance
- CS+/CS− geometry
- strict geometric candidate reconstruction
- historical distance-formula comparison
- stored geometric-candidate provenance
- response-profile value provenance
- mechanistic factorial decomposition
- counterfactual mechanism decomposition
- KC→EN selective-sampling analysis
- KCEN permutation null analysis
- original source/provenance tracing
- historical PI source tracing

---

## Interpretation

The results in this repository are computational/model-level
results.

They should not be interpreted as direct evidence of biological
causation without appropriate experimental validation.

The repository preserves the original computational assumptions,
known limitations, and control limitations.

---

## Reproducibility principle

No historical value is silently replaced.

Where an apparent discrepancy was identified, the repository
records the source, reconstructs the quantity where possible,
and preserves the distinction between the originally reported
quantity and the audited interpretation.
