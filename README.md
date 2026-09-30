# Honeybee vs Drosophila: Sparse Information Coding

A computational comparison of sparse information coding and representation stability in Honeybee- and Drosophila-inspired neural models.

## Overview

This project investigates how two insect-inspired neural architectures represent and maintain information under different conditions.

The same set of stimulus patterns is presented to both models, and their representations are examined under:

- Noise
- Neuron dropout
- Temporal changes
- Increasing pattern load
- Combined stress conditions

The main aim is to understand how stable sparse neural representations remain when the system is exposed to perturbations.

## Research Questions

This project addresses the following questions:

1. How stable are sparse representations in Honeybee- and Drosophila-inspired models?
2. How does noise affect representation stability?
3. How does neuron dropout affect the representation?
4. Does the representation persist across different temporal stages?
5. How does increasing the number of stored patterns affect performance?
6. Do the two architectures respond differently to perturbations?

## Experimental Design

The computational pipeline follows this general workflow:

Stimulus Generation  
↓  
Sparse Neural Representation  
↓  
Honeybee Model / Drosophila Model  
↓  
Decoder  
↓  
Noise and Dropout Tests  
↓  
Temporal Persistence  
↓  
Capacity Analysis  
↓  
Stress Testing  
↓  
Statistical Analysis

The experiment uses multiple random seeds to reduce dependence on a single simulation.

## Stimulus Set

The current experimental design uses:

- 100 stimulus patterns
- 52 unique patterns
- 20 random seeds
- Controlled similarity between stimulus patterns

A stimulus audit is performed before the main analysis to check pattern uniqueness and pairwise similarity.

## Experiments

### 1. Noise Test

Different levels of noise are introduced into the neural representation.

Representation retention and decoding accuracy are measured at each noise level.

### 2. Neuron Dropout

A proportion of neurons is randomly removed from the representation.

The effect on representation retention and decoding is then measured.

### 3. Temporal Persistence

The representation is evaluated across multiple temporal stages to examine whether information remains stable over time.

### 4. Capacity Sweep

The number of stimulus patterns is increased to examine how the system behaves as the information load increases.

### 5. Stress Test

Noise and neuron dropout are systematically varied to evaluate representation stability under stronger perturbations.

## Current Status

The first computational pipeline has been completed, including:

- Stimulus generation
- Stimulus auditing
- Honeybee-inspired model
- Drosophila-inspired model
- Noise analysis
- Dropout analysis
- Temporal analysis
- Capacity analysis
- Stress testing
- Statistical analysis
- Figure generation

The current version is considered a **diagnostic/preliminary version**.

Before drawing biological conclusions, the decoder and control procedures are being validated and the model parameters are being finalized.

## Preliminary Results

The preliminary simulation shows that both models retain part of their representations under noise and neuron dropout.

However, the current decoding results remain close to chance-level performance across many conditions. Therefore, the present results are being treated as a diagnostic stage rather than as a final biological comparison.

The next version will focus on validating the decoding pipeline, random controls, model architecture and capacity analysis.

## Reproducibility

The project is designed to be reproducible using fixed random seeds and saved configuration parameters.

Results and figures generated from each analysis will be stored in the repository.

## Repository Structure

```text
honeybee-vs-drosophila/
│
├── README.md
├── requirements.txt
│
├── code/
│   └── master_pipeline_v1.py
│
├── results/
│   ├── stimulus_audit.csv
│   ├── noise_sweep.csv
│   ├── dropout_sweep.csv
│   ├── temporal_persistence.csv
│   ├── capacity_sweep.csv
│   ├── stress_test.csv
│   ├── random_control.csv
│   └── statistics_summary.json
│
└── figures/
