# Data — interaction-networks-dmfc

This directory holds the Rajalingham et al. 2025 DMFC + RNN data release. Contents are not committed to git per `CONSTITUTION.md` (neural data handling rule).

## Layout

```
data/
├── README.md              ← this file (tracked in git)
└── dmfc/                  ← .gitignore'd; symlink or copy of the Zenodo release
    ├── all_hand_dmfc_dataset_50ms.pkl
    ├── perle_hand_dmfc_dataset_50ms.pkl
    ├── mahler_hand_dmfc_dataset_50ms.pkl
    ├── offline_rnn_neural_responses_reliable_50.pkl
    ├── valid_meta_sample_full.pkl
    ├── decode_*.pkl
    └── ...
```

## Source

- **Zenodo DOI**: 10.5281/zenodo.13952210
- **URL**: https://doi.org/10.5281/zenodo.13952210
- **Paper**: Rajalingham, R. et al. (2025). The role of mental simulation in primate physical inference abilities.
- **Companion code**: https://github.com/jazlab/MentalPong (`code/` is also bundled inside the Zenodo release).

## Local setup on this machine (2026-04-25)

`data/dmfc/` is a symlink to the local Zenodo download:

```
data/dmfc -> /Users/david/Downloads/MentalPong/data
```

The Zenodo bundle also contains `analyses/`, `code/`, and a top-level README at `~/Downloads/MentalPong/`. Only the `data/` subdirectory is exposed through this symlink.

## To recreate on another machine

```bash
# 1. Download the Zenodo release (see DOI above) and unpack to <local-dir>
# 2. Either symlink:
ln -s <local-dir>/data data/dmfc
# 3. ... or copy directly:
cp -r <local-dir>/data data/dmfc
```

A schema for each `.pkl` is documented in `SCRATCHPAD.md` under the *Milestone 1 findings* section.
