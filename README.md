# UniVI

UniVI is a **multi-modal variational autoencoder (VAE)** framework for aligning and integrating single-cell modalities such as RNA, ADT (CITE-seq), and ATAC. It’s built to support experiments like:

- Joint embedding of RNA + ADT (CITE-seq)
- RNA + ATAC (Multiome) integration
- RNA + ADT + ATAC (TEA-seq)
- Independent non-paired modalities from the same tissue type
- Cross-modal reconstruction and imputation
- Data denoising
- Evaluation of alignment quality (FOSCTTM, modality mixing, label transfer, etc.)

This repository contains the core UniVI code, training scripts, parameter files, and example notebooks.

---

## Repository structure

At a high level:

```text
UniVI/
├── README.md                  # This file
├── envs/                      # Conda / environment definitions (YAML, etc.)
├── data/                      # Local data (gitignored except small metadata)
├── figures/                   # Generated plots and figures (gitignored)
├── notebooks/                 # Jupyter notebooks for experiments / tutorials
├── parameter_files/           # JSON/YAML configs for models & training
├── saved_models/              # Trained model checkpoints
├── scripts/                   # CLI scripts / training entry points
├── tutorials/                 # Example workflows / how-to guides
└── univi/                     # UniVI Python package
    ├── __init__.py
    ├── config.py              # Config dataclasses / helpers
    ├── data.py                # Dataset wrappers, loaders (MultiModalDataset, etc.)
    ├── evaluation.py          # Metrics (FOSCTTM, mixing scores, label transfer, etc.)
    ├── matching.py            # Helper functions for matching / alignment
    ├── models/                # VAE architectures and modality-specific decoders
    ├── objectives.py          # Losses (ELBO, alignment, KL annealing, etc.)
    ├── plotting.py            # Plotting helpers / evaluation visualizations
    ├── trainer.py             # UniVITrainer: training loop & logging
    └── utils/                 # Utility functions, helpers, misc tools

