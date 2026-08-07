# Autoregressive Layout Engine: Deep Dive

This report covers the exact mathematical and architectural foundations of the **Autoregressive Layout Engine** used in Step 4 of the PlanGen pipeline. 

Unlike traditional methods that statically size rooms and then try to pack them, or diffusion models that start with static noise, this engine uses a **Graph Neural Network (GNN)** combined with an **Autoregressive Causal Transformer**. It learns to jointly predict *what* a room is, *how big* it should be, and *where* it should go—room by room, exactly like a human architect.

---

## 1. The Core Architecture

The system is built on two primary neural networks working in tandem:

### A. Graph Neural Network (GNN) Encoder
Before the sequence generation begins, the system maps out the mathematical relationships (adjacencies) between all the requested rooms.
- **Input:** Node features (approximate sizes) and Edge features (which rooms should be next to which).
- **Processing:** A GNN passes messages between connected rooms to build a deep understanding of the floor plan's topological requirements.
- **Output:** It outputs a `256-dimensional` embedding for every single room, plus one `global` embedding representing the entire floor.

### B. Autoregressive Causal Transformer
The transformer is responsible for predicting the layout step-by-step. 
- At every layer of the transformer, it uses **Cross-Attention** to look at the GNN embeddings. This means that even when placing the very first room, the transformer is highly aware of the topological constraints of the *entire* graph.
- It uses **Causal Self-Attention**, meaning it looks at every room it has placed so far to decide where to place the next one.

---

## 2. The Token Sequence

To feed a floor plan into a transformer, the geometry must be tokenized into a 1D sequence. The maximum sequence length is `127` tokens (supporting up to 25 rooms).

The sequence is formatted as:
`[GLOBAL | R0_type, R0_cx, R0_cy, R0_w, R0_h | R1_type, R1_cx ... ]`

1. **GLOBAL Token:** A single token containing the plot dimensions, number of rooms, Vastu compliance flag, and the physical entrance angle (encoded using sine/cosine).
2. **Room Tokens (5 per room):**
   - **TYPE:** Categorical class (0-15, e.g., Master Bedroom, Kitchen).
   - **CX, CY:** The center X and center Y coordinates (continuous).
   - **W, H:** The width and height (continuous).

All continuous tokens are normalized strictly between `[0, 1]` based on the net buildable plot boundaries, and embedded using **Sinusoidal Positional Encoding** before passing into the transformer.

---

## 3. The Output Heads & Mixture of Gaussians (MoG)

When the transformer outputs a hidden state for a token, it has to convert that back into a real-world prediction.

- **For TYPE tokens:** It uses a standard Linear layer + Softmax to classify the room type.
- **For Continuous tokens (CX, CY, W, H):** It does NOT output a single number. Instead, it uses a **Mixture of Gaussians (MoG)** head.

### The Math of MoG:
Real-world architecture has multimodal distributions (e.g., a door can be on the far left wall OR the far right wall, but rarely in the dead center). If a network predicts a single number, it will average those two options and put the room in the dead center (which is wrong).
- The MoG head outputs parameters for `K=3` separate Gaussian distributions per scalar.
- It predicts:
  1. **Mixture Weights:** Which of the 3 distributions to use.
  2. **Means:** The center of the distributions.
  3. **Log Standard Deviations:** The uncertainty of the distributions.
- During inference, the engine samples from this complex distribution using a **Temperature** parameter to control creativity/randomness.

---

## 4. The Autoregressive Generation Process

Rooms are not generated randomly. The Enricher sorts them by **Generation Priority**. "Anchor" rooms like the Living Room and Master Bedroom are generated first, while service rooms like Bathrooms and Stores are generated last.

The loop looks like this:
1. Feed the `GLOBAL` token. Predict `R0_TYPE`.
2. Feed `R0_TYPE`. Predict `R0_CX`.
3. Feed `R0_CX`. Predict `R0_CY`.
4. Feed `R0_CY`. Predict `R0_W`.
5. Feed `R0_W`. Predict `R0_H`.
6. Moving to the next room...

Because it is autoregressive, if the model decides to make the Living Room exceptionally large, the hidden state passed to the Kitchen will mathematically reflect that there is less space left on the plot, and the Kitchen will naturally shrink.

---

## 5. Post-Processing: Physics and Math

Neural networks are statistical and slightly fuzzy. Once the network outputs the raw `(x, y, w, h)` coordinates, the engine applies rigorous math to lock it to physical reality.

### A. Dimension Clamping (Aspect Ratios)
If the AI predicts an absurd aspect ratio (e.g., a room that is 4 feet wide and 30 feet long), the `_clamp_rooms` function detects this. It redistributes the excess dimension into the short side to approach a standard architectural ratio (e.g. 1:2.5 for bedrooms, 1:3 for bathrooms), while enforcing hard National Building Code (NBC) minimums.

### B. Projected Gradient Descent (Overlap Resolution)
The AI might predict rooms that overlap slightly. The engine runs a physics-like simulation (Separating-Axis Theorem) for 400 iterations:
1. It calculates the penetration depth between overlapping rooms on both X and Y axes.
2. It pushes the rooms apart along the axis of minimum penetration.
3. **Inverse-Area Weighting:** The distance each room is pushed is inversely proportional to its area. A massive Living Room acts like a heavy boulder and barely moves, while a tiny Bathroom acts like a pebble and gets pushed completely out of the way.

### C. Wall-Snap Phase
After overlaps are resolved, rooms might have a tiny gap between them (e.g., 2 inches). To make the plan look like it was drawn in CAD, the engine looks for rooms that share a significant span and are within a `SNAP_DIST` (3.6 inches) of each other. It mathematically snaps their coordinates together so the walls are perfectly flush.

---

## 6. Pure NumPy Inference

Remarkably, the entire generative step during production runs **without PyTorch**. 
To make the application incredibly lightweight and fast, the PyTorch weights are exported to `.npz` files. The `LayoutTransformerNumpy` class manually implements causal self-attention, cross-attention, layer normalization, and GeLU activations using pure matrix multiplication in NumPy. This removes the massive PyTorch dependency footprint for final deployment.
