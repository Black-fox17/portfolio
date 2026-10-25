# Multi-Level Lumbar Spine Degenerative Disease Grading on Volumetric MRI

## Introduction: The Radiologist's Burden & The 25-Target Problem

Low back pain is the single leading cause of physical disability worldwide. When patients present with severe radiculopathy, sciatica, or debilitating spinal compression, magnetic resonance imaging (MRI) is the clinical gold standard for determining whether conservative physical therapy or urgent neurosurgical decompression is warranted.

However, interpreting a complete lumbar spine MRI exam is one of the most cognitively demanding tasks in diagnostic radiology. A standard study comprises hundreds of DICOM slices acquired across multiple pulse sequences and spatial planes: **Sagittal T1**, **Sagittal T2 / STIR**, and **Axial T2**. 

Radiologists must examine every intervertebral level—from $L_1/L_2$ down to $L_5/S_1$—and independently evaluate three distinct structural conditions:
1. **Spinal Canal Stenosis**: Narrowing of the central vertebral canal compressing the thecal sac and cauda equina.
2. **Left & Right Neural Foraminal Narrowing**: Constriction of the lateral exit canals through which exiting spinal nerve roots pass.
3. **Left & Right Subarticular Stenosis**: Impingement of the lateral recess beneath the superior articular facet before the nerve enters the foramen.

```
       Lumbar Intervertebral Levels:
       [ L1 / L2 ] ── Central Canal · Left Foramen · Right Foramen · Left Subarticular · Right Subarticular
       [ L2 / L3 ] ── Central Canal · Left Foramen · Right Foramen · Left Subarticular · Right Subarticular
       [ L3 / L4 ] ── Central Canal · Left Foramen · Right Foramen · Left Subarticular · Right Subarticular
       [ L4 / L5 ] ── Central Canal · Left Foramen · Right Foramen · Left Subarticular · Right Subarticular
       [ L5 / S1 ] ── Central Canal · Left Foramen · Right Foramen · Left Subarticular · Right Subarticular
       ─────────────────────────────────────────────────────────────────────────────────────────────────
       Total: 5 Anatomical Levels × 5 Pathological Targets = 25 Simultaneous Multi-Class Decisions
```

Every single one of these 25 targets must be graded across three severity tiers: **Normal/Mild**, **Moderate**, and **Severe**. 

In the **RSNA 2024 Lumbar Spine Degenerative Classification Challenge**, our goal was to engineer an end-to-end deep learning framework that automates this entire diagnostic matrix directly from raw multi-series DICOM volumes. This article dissects how I structured the data ingestion, 3D coordinate sorting, feature extraction backbones with Generalized Mean (GeM) pooling, and asymmetric severe-penalized loss optimization.

---

## Clinical Series Mapping: Why Single-View Vision Fails

A critical mistake made by computer vision engineers new to medical imaging is treating all MRI slices as interchangeable images. In radiology, different tissue properties and anatomical planes highlight completely different pathologies:

| Degenerative Condition | Primary Series | Anatomical Plane | Diagnostic Focus |
| :--- | :--- | :--- | :--- |
| **Spinal Canal Stenosis** | `Sagittal T2` & `Axial T2` | Sagittal / Axial | Cerebrospinal fluid (CSF) hyperintensity surrounding the spinal cord. Narrowing obliterates the bright CSF buffer. |
| **Neural Foraminal Narrowing** | `Sagittal T1` | Parasagittal | Fat pad preservation within the intervertebral neural foramen. Loss of perineural fat signals nerve root compression. |
| **Subarticular Stenosis** | `Axial T2` | Transverse Axial | Lateral recess dimension between the posterior vertebral margin and the facet joint. |

Training a monolithic network on an unorganized mix of slices leads to gradient interference: features optimal for detecting neural foraminal fat pads on T1 slices actively degrade spinal canal CSF boundary detection on T2 slices.

To address this, our architecture enforces **series-aware routing**, directing sagittal views to foraminal and central canal heads, and axial views to lateral recess and facet subarticular heads.

---

## The DICOM Ingestion Pipeline: Coordinates & VOI LUTs

Medical DICOM files are fundamentally different from standard PNG or JPEG images. Raw DICOM files do not arrive pre-sorted in orderly sequential frames, and naive loading introduces two catastrophic failure modes:

### 1. Spatial Slice Inversion & Sorting
Slices stored in DICOM folders are arbitrarily ordered by file system timestamps or acquisition identifiers. To reconstruct a coherent anatomical volume, slices must be mathematically projected and sorted along the acquisition axis using the DICOM metadata vector `ImagePositionPatient`:

```python
import pydicom
import numpy as np

def get_slice_position(dcm: pydicom.Dataset) -> float:
    """Extract spatial slice position along the patient acquisition axis."""
    pos = dcm.ImagePositionPatient
    orient = dcm.ImageOrientationPatient
    
    # Calculate normal vector to the acquisition plane
    normal = np.cross(orient[:3], orient[3:])
    
    # Project 3D position onto the normal axis
    return float(np.dot(pos, normal))
```

Sorting slices along this projected normal axis guarantees that slice $i$ and slice $i+1$ are spatially contiguous physical neighbors.

### 2. VOI LUT Transformations & Grayscale Correction
Medical scanners record pixel intensities in high-dynamic-range Hounsfield units or vendor-specific sensor ranges ($12$ to $16$-bit depths). Furthermore, some scanners output `MONOCHROME1` (where the minimum pixel value is white) instead of standard `MONOCHROME2` (where minimum is black). 

Failing to apply Value of Interest (VOI) Look-Up Tables (LUT) or photometric inversion causes soft tissue contrast (such as disk protrusions into the thecal sac) to be crushed into pure black or blown out:

```python
from pydicom.pixel_data_handlers.util import apply_voi_lut

def load_and_preprocess_dicom(path: str) -> np.ndarray:
    dcm = pydicom.dcmread(path)
    
    # Apply scanner-calibrated VOI LUT windowing
    try:
        pixel_array = apply_voi_lut(dcm.pixel_array, dcm)
    except Exception:
        pixel_array = dcm.pixel_array
        
    # Invert MONOCHROME1 to standard visual polarity
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixel_array = np.amax(pixel_array) - pixel_array
        
    # Min-max scale to standard [0, 255] uint8 representation
    p_min, p_max = pixel_array.min(), pixel_array.max()
    if p_max > p_min:
        norm = (pixel_array - p_min) / (p_max - p_min) * 255.0
    else:
        norm = np.zeros_like(pixel_array)
        
    return norm.astype(np.uint8)
```

---

## 2.5D Volumetric Stacking vs. Native 3D Convolutions

When handling volumetric scans, the naive approach is to deploy a 3D Convolutional Network (such as 3D ResNet-50 or MedicalNet). In practice, full 3D convolutions introduce severe drawbacks:
* Exponential memory complexity: Storing 3D intermediate feature activations for backpropagation exhausts GPU VRAM, restricting batch sizes to $1$ or $2$ volumes and breaking Batch Normalization stability.
* Inconsistent axial slice depths: Across different hospitals and MRI protocols, a series may contain anywhere from $8$ to $42$ slices.

Instead, we adopted an optimized **2.5D uniform volumetric sampling strategy**:
1. For every MRI series, exactly $K = 16$ equidistant physical slices are sampled across the patient's anatomical volume.
2. The sampled slices are passed through 2D backbones initialized from high-capacity vision pretraining (**EfficientNetV2-S** and **ConvNeXt-Tiny**).
3. The slice-level features are aggregated using **Generalized Mean (GeM) Pooling** before multi-task level grading.

```
Raw DICOM Series (Variable Slices: 12 - 40)
                  │
                  ▼ [ Spatial Projection & Uniform 16-Slice Sampling ]
         [ S₁ , S₂ , S₃ , ... , S₁₆ ]  (Shape: 16 × 1 × 256 × 256)
                  │
                  ▼ [ High-Capacity 2D Feature Backbone ]
                  │   (ConvNeXt / EfficientNetV2 Feature Maps)
                  │
                  ▼ [ Generalized Mean (GeM) Pooling ]
                  │   (Aggregates localized focal compression hotspots)
                  │
                  ▼ [ Multi-Task Spinal Classifier Heads ]
         ┌────────┬────────┬────────┬────────┬────────┐
       L1/L2    L2/L3    L3/L4    L4/L5    L5/S1
       (3-cls)  (3-cls)  (3-cls)  (3-cls)  (3-cls)
```

---

## Generalized Mean (GeM) Pooling vs. Global Average Pooling

Standard classification architectures use Global Average Pooling (GAP) to collapse spatial feature maps:

$$f_{\text{GAP}} = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i,j}$$

In lumbar spine imaging, GAP introduces a fatal failure mode: **pathology dilution**. A critical neural foraminal stenosis or disc herniation may only occupy a tiny fraction ($5\%$) of an MRI slice. When the entire slice is averaged, the small high-intensity compression signal gets completely diluted by the large background area of surrounding muscle, vertebral body bone, and subcutaneous fat.

Conversely, Global Max Pooling ($p \to \infty$) is hypersensitive to solitary imaging artifacts or gradient noise.

We resolve this by parameterizing feature aggregation via **Generalized Mean (GeM) Pooling**:

$$f_{\text{GeM}} = \left( \frac{1}{|X|} \sum_{x \in X} x^p \right)^{\frac{1}{p}}$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeMPooling(nn.Module):
    """Generalized Mean (GeM) Pooling with learnable power parameter p."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamped power pooling over spatial dimensions
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (1, 1)
        ).pow(1.0 / self.p).flatten(1)
```

When $p = 1.0$, GeM is standard average pooling. As $p$ increases ($p \approx 3.0$), the mathematical operation shifts attention toward the highest-magnitude activations corresponding to focal spinal stenosis and nerve root impingement, without ignoring anatomical context.

---

## Multi-Task Heads & The Asymmetric RSNA Log Loss

To grade all five lumbar levels simultaneously, the pooled representation branches into 5 parallel multi-task classifier heads, each outputting 3 logits corresponding to `Normal/Mild`, `Moderate`, and `Severe`:

```python
class RSNASpineClassifier(nn.Module):
    def __init__(self, backbone_name="efficientnet_v2_s", in_features=1280, num_levels=5, num_classes=3):
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        self.pool = GeMPooling(p=3.0)
        self.dropout = nn.Dropout(p=0.2)
        
        # Parallel classifier heads for L1/L2, L2/L3, L3/L4, L4/L5, L5/S1
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 256),
                nn.SiLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
            for _ in range(num_levels)
        ])

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.dropout(self.pool(feats))
        logits = [head(pooled) for head in self.heads]
        return torch.stack(logits, dim=1) # (Batch, 5, 3)
```

### The Evaluation Metric: Severe-Penalized Multi-Class Log Loss
The competition evaluation metric mirrors real clinical risk. Missed diagnoses of severe disease have catastrophic clinical consequences, so the loss weights each severity tier asymmetrically:
* **Normal / Mild**: Relative weight = $1.0$
* **Moderate**: Relative weight = $2.0$
* **Severe**: Relative weight = $4.0$

$$\mathcal{L}_{\text{RSNA}} = - \frac{1}{\sum_{i} w_{y_i}} \sum_{i=1}^{N} w_{y_i} \log(p_{i, y_i})$$

To further prevent the network from hedging toward the dominant `Normal/Mild` class, we engineered an **auxiliary any-severe condition loss** ($\mathcal{L}_{\text{aux}}$):

```python
class AnySevereSpineLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=2.0, label_smoothing=0.05):
        super().__init__()
        weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
        self.register_buffer("weights", weights)
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        b, l, c = logits.shape
        flat_logits = logits.view(-1, c)
        flat_targets = targets.view(-1)
        
        # Base weighted cross-entropy
        base_loss = F.cross_entropy(
            flat_logits, flat_targets, 
            weight=self.weights.to(logits.device),
            label_smoothing=self.label_smoothing
        )
        
        # Auxiliary severe penalty: checks if ANY level has severe pathology (class index 2)
        probs = F.softmax(logits, dim=-1)
        severe_probs = probs[..., 2].max(dim=-1).values
        severe_targets = (targets == 2).any(dim=-1).float()
        
        aux_loss = F.binary_cross_entropy(
            severe_probs.clamp(1e-6, 1.0 - 1e-6),
            severe_targets
        )
        
        return self.alpha * base_loss + self.beta * aux_loss
```

---

## Validation Strategy & Experimental Benchmarks

### 1. Leakage-Free Patient Stratification
A single patient study contains multiple series and slice volumes. If slices from patient $P$ appear in both the training and validation sets, the model learns patient-specific anatomical quirks rather than generalizable radiological patterns.

We enforced strict **GroupKFold on `study_id`** ($K = 5$), ensuring zero patient leakage between folds.

### 2. Empirical Ablation Progression

Across iterative experiments, moving from single-slice baselines to volumetric GeM ensembles produced dramatic performance leaps:

| Iteration | Model Architecture | Slices per Study | Image Resolution | Multi-Class Log Loss | Severe Disease Recall |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **01** | ResNet-34 Single-Slice Baseline | 1 | $224 \times 224$ | 0.684 | 64.2% |
| **02** | ResNet-50 Multi-Task Volume | 8 | $224 \times 224$ | 0.542 | 71.8% |
| **03** | ConvNeXt-Tiny + GeM ($p=3.0$) | 16 | $256 \times 256$ | 0.495 | 78.4% |
| **04** | **EfficientNetV2-S + ConvNeXt Ensemble + TTA** | **16** | **$256 \times 256$** | **0.441** | **83.6%** |

```
Log Loss Reduction Progression:
Baseline (0.684) ───► Volume 8-Slice (0.542) ───► GeM Pooling (0.495) ───► Ensemble (0.441)
Severe Recall Improvement:
Baseline (64.2%) ───► Volume 8-Slice (71.8%) ───► GeM Pooling (78.4%) ───► Ensemble (83.6%)
```

---

## Key Takeaways for Production Medical AI

1. **DICOM Orientation is Ground Truth**: Never assume raw image slices are sequential or oriented properly. Projecting `ImagePositionPatient` onto the normal plane is essential for valid 3D reasoning.
2. **GeM Pooling Preserves Focal Pathology**: Standard global average pooling dilutes localized lesions. Generalized Mean Pooling with $p \approx 3.0$ sharpens attention on focal cord and nerve root impingements.
3. **Loss Functions Must Mirror Clinical Consequence**: In clinical diagnostic workflows, false negatives on severe compression are far more dangerous than false positives on mild cases. Sample-weighting loss matrices combined with auxiliary severe detection penalties ensures networks prioritize high-risk pathologies.
4. **Series-Aware Multi-Task Decoupling**: Rather than forcing a single model to process all MRI orientations, matching pulse sequence physics (T1 vs. T2 vs. STIR) to specific anatomical conditions eliminates inter-task feature interference.

The open-source PyTorch pipeline, configuration matrices, and inference scripts are available on GitHub:  
👉 [**rsna-lumbar-spine-classification-**](https://github.com/Black-fox17/rsna-lumbar-spine-classification-)
