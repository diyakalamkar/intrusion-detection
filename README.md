# Unsupervised Network Intrusion Detection System

A comprehensive machine learning project analyzing network intrusion detection using unsupervised learning techniques on the NSL-KDD dataset.

## Project Overview

**Dataset:** NSL-KDD (125,973 network flows)  
**Objective:** Detect network attacks without labeled training data  
**Methods:** 7 algorithms (4 individual + 3 ensemble approaches)  
**Best Performance:** 94.3% accuracy (Autoencoder)

## Key Achievements

- **94.3% detection accuracy** using deep learning Autoencoder
- **41 → 17 features** via PCA (95% variance retained)
- **Interactive Streamlit dashboard** for visualization
- **Production-ready models** with inference pipeline

## Novel Insights

### Insight #1: Cluster Tightness Predicts Detectability
DoS attacks cluster 2.7× tighter than U2R attacks, explaining detection difficulty.

### Insight #2: Cluster Purity Analysis (82%)
K-Means achieves 82% overall purity with DoS at 97.1% pure cluster.

### Insight #3: Single Strong Algorithm > Weak Ensemble
Autoencoder (93.9% F1) outperforms weighted ensemble (91.1% F1) when ensemble includes weak components.

## Repository Structure
