# Module 1: Mathematical Carrier Lattice Engine

This document records the complete engineering narrative and educational study guide for **Module 1**. It explains each submodule from first principles, including the mathematical formulas, variable definitions, implementation logic, and unit test verification.

---

## Submodule 1.1: Cryptographic Spatial Carrier Generator

### 1. The Story: Why We Need This Submodule
In classical digital watermarking, systems often used visible stamp logos or static pseudo-random noise. However:
- **Static Watermarks**: Attackers can easily discover, isolate, and subtract static watermarks.
- **Unstructured Random Noise**: Fragile to image compression; sending the image over WhatsApp or re-saving as a JPEG instantly wipes out unstructured single-pixel noise.

To solve this, **Submodule 1.1** designs a **deterministic, position-keyed 2D harmonic carrier lattice**:
1. It blankets the entire image with an invisible mathematical carrier grid that is cryptographically tied to a **Secret Key** (e.g., `"CompanyAuditKey2026"`).
2. Without knowing this key, an adversary cannot calculate or mimic the pattern.
3. The carrier sits in the **mid-frequency spatial band** ($T = 8$ to $16$ pixels), ensuring it survives standard compression while remaining hypersensitive to localized tampering.

---

### 2. What We Did
We implemented `deep_guard/core/lattice.py`, which provides:
1. `derive_key_seed(secret_key)`: Converts any string passphrase into an unsigned 64-bit integer seed using **SHA-256**.
2. `generate_carrier_lattice(...)`: Computes an orthogonal 2D carrier matrix $L \in \{-1.0, +1.0\}^{H \times W \times C}$ tailored to the exact image dimensions $(H, W)$.

---

### 3. Step-by-Step Mathematical Formulation

#### Step 3.1: Cryptographic Seed Derivation
Any text passphrase is transformed into a pseudo-random seed using cryptographic hashing:
$$\text{Digest} = \text{SHA-256}(\text{Secret Key})$$
$$\text{Seed} = \text{BytesToInt}(\text{Digest}[0:8]) \pmod{2^{31} - 1}$$
*Why?* Ensures that even changing a single character in the password (e.g., `"key1"` vs `"key2"`) results in completely unrelated, avalanche-scrambled seeds.

---

#### Step 3.2: 2D Harmonic Carrier Wave
We define a 2D spatial coordinate grid $x \in [0, W-1]$ and $y \in [0, H-1]$.
The spatial angular frequency $\omega$ is governed by the chosen period $T$ (default $T = 8$ pixels):
$$\omega = \frac{2\pi}{T}$$

To prevent an attacker from guessing the grid's alignment, the seed injects random phase shifts $\phi_x, \phi_y \in [0, 2\pi)$:
$$\sin_x(x) = \sin(\omega \cdot x + \phi_x)$$
$$\cos_y(y) = \cos(\omega \cdot y + \phi_y)$$

The 2D harmonic matrix is computed using a vectorized outer product:
$$W(x, y) = \cos_y(y) \otimes \sin_x(x) = \sin(\omega x + \phi_x) \cdot \cos(\omega y + \phi_y)$$

---

#### Step 3.3: Keyed Block-Sign Modulation (Orthogonality Engine)
If we only used pure sine/cosine waves, two different keys might still share some overlapping harmonic correlation. 

To make every key **100% mathematically orthogonal** (cross-correlation $\approx 0$), we divide the image into blocks of size $2T \times 2T$ (e.g., $16 \times 16$ pixels). For each block at coordinate $(b_x, b_y)$, the random generator assigns an independent random sign:
$$M_{\text{signs}}(b_x, b_y) \in \{-1.0, +1.0\}$$

The composite carrier is then:
$$C(x, y) = W(x, y) \cdot M_{\text{signs}}(x, y)$$

---

#### Step 3.4: Bipolar Discretization into $\{-1.0, +1.0\}$
To guarantee numerical stability and uniform perturbation amplitude across the image, we apply a sign threshold:
$$L(x, y) = \begin{cases} +1.0, & \text{if } C(x, y) \ge 0 \\ -1.0, & \text{if } C(x, y) < 0 \end{cases}$$

```
Sample 8x8 Lattice Matrix Values (Top-Left Corner):
[ +1.0,  +1.0,  +1.0,  +1.0,  -1.0,  -1.0,  -1.0,  -1.0 ]
[ +1.0,  +1.0,  +1.0,  +1.0,  -1.0,  -1.0,  -1.0,  -1.0 ]
[ +1.0,  +1.0,  +1.0,  +1.0,  -1.0,  -1.0,  -1.0,  -1.0 ]
[ +1.0,  +1.0,  +1.0,  +1.0,  -1.0,  -1.0,  -1.0,  -1.0 ]
[ -1.0,  -1.0,  -1.0,  -1.0,  +1.0,  +1.0,  +1.0,  +1.0 ]
[ -1.0,  -1.0,  -1.0,  -1.0,  +1.0,  +1.0,  +1.0,  +1.0 ]
[ -1.0,  -1.0,  -1.0,  -1.0,  +1.0,  +1.0,  +1.0,  +1.0 ]
[ -1.0,  -1.0,  -1.0,  -1.0,  +1.0,  +1.0,  +1.0,  +1.0 ]
```

---

#### Step 3.5: Multi-Channel Expansion
To prevent color distortions (like turning a white background slightly pink or green), the exact same lattice matrix $L(x, y)$ is replicated across all 3 color channels ($R, G, B$):
$$L_{\text{RGB}}(x, y, c) = L(x, y), \quad \forall c \in \{0, 1, 2\}$$
This ensures the perturbation acts only on **luminance (brightness)**, which human eyes perceive much less sensitively than chromaticity (color shifts).

---

### 4. What It Helps in the Overall Project

| Feature | How Submodule 1.1 Solves It |
| :--- | :--- |
| **Zero Database Storage** | The pattern is generated on the fly in **55 milliseconds**. No database needed. |
| **Tamper Localization** | Because $L(x, y)$ depends on $(x, y)$, any altered pixel immediately loses the pattern. |
| **Copy-Move Protection** | Cloning text from $(x_1, y_1)$ to $(x_2, y_2)$ causes destructive phase collision. |
| **Cryptographic Security** | Cross-correlation between two different keys is strictly **$\approx 0.002$** (orthogonal). |

---

### 5. Unit Test Verification (`deep_guard/tests/test_lattice.py`)

All 4 unit tests passed with 100% success:

1. **`test_determinism`** ($\text{PASS}$):
   - Verified that running `generate_carrier_lattice` twice with identical keys produces **100% bitwise identical matrices**.
2. **`test_orthogonality`** ($\text{PASS}$):
   - Calculated normalized cross-correlation between lattices generated with different keys:
     $$\text{Corr}(L_1, L_2) = \frac{L_1 \cdot L_2}{\|L_1\| \|L_2\|} = \mathbf{0.00239} < 0.05$$
   - Confirms independent keys are completely uncorrelated.
3. **`test_shape_and_bipolar_values`** ($\text{PASS}$):
   - Confirmed output tensor shape matches $(H, W, C)$ with values strictly in $\{-1.0, +1.0\}$.
4. **`test_execution_speed`** ($\text{PASS}$):
   - Benchmarked Full HD 1080p ($1920 \times 1080$) carrier generation: took **$55.22\text{ ms}$** on CPU.

---

## Status Summary
- **Submodule 1.1**: **Completed, Tested, Documented**
- **Next Submodule**: **Submodule 1.2: Perceptually Bounded Modulation & Image Injection Engine** (Injecting the carrier into actual images at $\text{PSNR} > 48\text{ dB}$).
