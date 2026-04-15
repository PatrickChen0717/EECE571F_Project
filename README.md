# LSTM + GAT SurgPose Trajectory Prediction

EECE 571F project for surgical instrument trajectory prediction on the SurgPose dataset.

The main workflow in this repository uses `python -m scripts.train_surgpose` for training and `python -m scripts.evaluate_surgpose` for evaluation.

## Project Summary

This project combines:

- an LSTM-based temporal encoder
- a Graph Attention Network (GAT) for spatial reasoning over keypoints
- DINOv2 visual features for image conditioning

The goal is to predict future surgical tool keypoint trajectories from observed motion.

## Data Representation

The motion input uses shape `(B, T, M, 5, 2)`:

- `B`: batch size
- `T`: time steps
- `M`: number of instruments
- `5`: keypoints per instrument
- `2`: `(delta x, delta y)` motion values

The pipeline also includes:

- a visibility flag for each keypoint
- a virtual root node for each instrument
- frame-level visual features for conditioning

## Important Symbols

- `O`: number of observed frames
- `P`: number of future frames to predict
- `M`: number of instruments in each frame

The model uses the previous `O` frames to predict the next `P` frames.

## Main Scripts

### Training

```bash
python -m scripts.train_surgpose
```

This script:

- loads SurgPose keypoint annotations
- builds windowed observation and prediction sequences
- trains the LSTM+GAT model with visual features
- saves checkpoints under `models/`

### Evaluation

```bash
python -m scripts.evaluate_surgpose
```

This script:

- loads a trained checkpoint
- runs blockwise trajectory prediction
- generates trajectory plots for the evaluation set

## Other Baseline Models

Other baseline models can be trained and evaluated using:

- `python -m scripts.train_lstm` and `python -m scripts.evaluate_lstm`
- `python -m scripts.train_transformer` and `python -m scripts.evaluate_transformer`

## Model Overview

The full pipeline is:

1. Encode motion deltas from observed keypoints.
2. Model temporal dynamics with LSTM.
3. Model spatial relations with GAT.
4. Fuse trajectory features with visual features.
5. Predict future keypoint motion.

## Training Objective

The training script uses:

- position loss
- delta loss
- direction loss
- magnitude loss

Ground-truth motion is computed from frame-to-frame position differences.

## Notes

- Update dataset paths in the SurgPose scripts if your dataset is stored in a different location.
- `scripts/train_surgpose.py` currently uses Weights & Biases logging.
- `scripts/evaluate_surgpose.py` contains a checkpoint path that should be changed to the model you want to evaluate.
