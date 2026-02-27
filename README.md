<!-- Instruction:
1. set wandb user key or set Enable_WandB == False  in train.py
wandb.login(key="")
2. O and P means using previous O frame to predict the following P
3. M is how many instruments in the image

Model Structure:
(B, T, M, 5, 2)
B = batch size
T = time steps
M = number of instruments
5 = keypoints per instrument
2 = (x, y)

1. input phi Φ embedding, linear layer
2. Vanilla LSTM
3. GAT  -->

# Trajectory Prediction of DVRK Joints By LSTM with Graph Attention
EECE 571F Project




## Instructions

### 1. Weights & Biases (W&B) Setup

Before training, either set your W&B API key in `train.py`:

wandb.login(key="YOUR_KEY")

Or disable W&B logging:

Enable_WandB = False

---

### 2. Meaning of O and P

- O = number of observed frames  
- P = number of future frames to predict  

The model uses:

Previous O frames → Predict next P frames

---

### 3. Meaning of M

M = number of surgical instruments in each frame.
- M = 2 → two instruments (e.g., left & right tools)

Each instrument contains 5 keypoints.

---

## Model Structure

Input tensor shape:

(B, T, M, 5, 2)

Where:

- B = batch size  
- T = time steps  
- M = number of instruments  
- 5 = keypoints per instrument  
- 2 = (x, y) coordinates  

## Architecture

### 1. Φ Embedding (Input Projection)

- Linear layer applied to raw keypoint positions  
- Embeds coordinates into feature space  

### 2. Temporal Modeling — Vanilla LSTM

- Processes feature sequences over time  
- Learns motion dynamics  

### 3. Spatial Modeling — Graph Attention Network (GAT)

- Models spatial relationships between:
  - Keypoints
  - Virtual root node (instrument center)
  - (If M > 1) interactions between instruments



## Other

Ground-truth delta:

Δpos_t = pos_{t+1} − pos_t

Loss: Masked MSE


## Run

python -m scripts.train

python -m scripts.evaluate
