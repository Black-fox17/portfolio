# Segmenting Solar Filaments in H-Alpha Images: What Actually Happened When We Built the Pipeline

*Full code, configuration, unit tests, and research notes for this project are available on GitHub: [Black-fox17/solar-filament-instance-segmentation](https://github.com/Black-fox17/solar-filament-instance-segmentation).*

---

If you have never stared at a full-disk H-alpha image of the Sun, picture a glowing, textured sphere covered in dark, winding cracks. Those dark ribbons are **solar filaments** — massive arches of relatively cool, dense plasma suspended above the solar chromosphere by magnetic fields.

In the **Solar Filament Segmentation Challenge 2026** (built on the `MAGFiLO 1.0` dataset), the objective sounds refreshingly simple on paper: take a `2048 × 2048` grayscale solar image, find every individual filament on the disk, and output a separate, non-overlapping COCO RLE mask for each one.

Sounds like a weekend project, right? Download the COCO annotations, run `train_test_split`, fine-tune a pretrained Mask R-CNN for a dozen epochs, run inference, and upload `submission.csv`.

Well, almost every single step of that textbook plan broke — or handed us a nasty surprise — the moment we actually looked at the data.

This article is the honest, start-to-finish walkthrough of what we found, what failed, what we threw away, what actually worked, and the weaknesses that are still sitting in our baseline pipeline. Everything here comes directly from our measured runs; nothing is polished after the fact.

---

## First: What Are We Actually Looking At?

The competition archive (`filament-segmentation-2026`, 671 MB) unpacks into `MAGFiLO_1.0_Kaggle_2026/` with:
- **707 training images** (`train/train_images/*.jpeg`)
- **180 test images** (`test/test_images/*.jpeg`)
- **1 COCO annotation file** (`MAGFiLO_1.0_Annotations_kaggle2026_train.json`, 46.36 MB)

Every single image is a `2048 × 2048`, 8-bit single-channel grayscale (`L`) JPEG from the National Solar Observatory's GONG H-alpha network. Sampling 20 training images gave us a quick feel for the pixel distribution:
- Average min / max: `0.00` / `239.50`
- Average mean: `83.09` (`0.326` when scaled to `[0, 1]`)
- Average std: `55.40` (`0.217` when scaled to `[0, 1]`)

Keep those two numbers — `0.326` and `0.217` — in the back of your mind. They come back when we configure the model.

![Full-disk GONG H-alpha solar images alongside their reconstructed COCO instance masks](/blog/solar-filament/solar_samples_and_masks.png)

---

## Surprise #1: 1,154 COCO Images… That Are Only 707 Physical Photos

When we loaded the training JSON with `pycocotools`, COCO reported **1,154 image records** and **8,199 polygon annotations**.

Wait a second. There are only **707** `.jpeg` files in `train/train_images/`. How do you get 1,154 image records out of 707 files?

Let's see why. When we grouped COCO records by their `file_name` field, **296 physical solar images showed up more than once**:

| Annotation Records per Physical Image | Physical Images |
|---|---:|
| 1 record | 411 |
| 2 records | 145 |
| 3 records | 151 |
| **Total** | **707 unique images (1,154 records)** |

Each duplicate record has a different COCO `id` prefix — for example `050101-20111116063134Lh`, `050102-20111116063134Lh`, and `050103-20111116063134Lh` — all pointing to the exact same JPEG file. They are independent annotator passes over the same solar observation.

![Three independent annotator passes overlaid on the exact same physical solar image (20111116063134Lh.jpeg)](/blog/solar-filament/multi_annotator_comparison.png)

Now, why is that a big deal? Two reasons.

### 1. Silent Data Leakage

Our very first `train_test_split` cell split on the 1,154 COCO IDs (980 train / 174 validation). Had we left it there, the exact same solar disk images would have appeared in both training and validation under different annotator IDs — inflating validation scores while hiding overfitting.

Once we spotted it, we threw out the record-level split and split on the **707 unique physical filenames** instead (85/15, `random_state=42`):
- **Training**: `600` unique physical images (`992` annotation records)
- **Validation**: `107` unique physical images (`162` annotation records)
- **Physical filename overlap**: `0` (explicitly asserted in [`annotations.py`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/src/solar_filament/data/annotations.py))

### 2. Annotators Disagree Heavily

Even more interesting: when we compared the foreground union masks of different annotators on the *same* image, they were not minor 1-pixel disagreements.

- On `20111116063134Lh.jpeg`, all three annotators labeled 11 filaments, yet their total foreground pixel counts were `28,884`, `29,875`, and `45,522`.
- On `20131007075314Th.jpeg`, the pairwise union IoU between the three annotators was only **`0.4778`**, **`0.4009`**, and **`0.3604`**.

If you dump all 1,154 records into the same training epoch, the model sees the exact same solar image two or three times per epoch with noticeably different target masks. To avoid feeding contradictory supervision in the same pass, our [`SolarFilamentDataset`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/src/solar_filament/data/dataset.py) implements **epoch-level single-annotator sampling**: at the start of every training epoch (`set_epoch(epoch)`), it selects **one** annotation record per physical image for the entire epoch (and deterministically picks `record_ids[0]` during validation).

---

## Surprise #2: Filaments Are Tiny, Thin, and Mostly Empty Boxes

Next, we audited all `8,199` annotations across the four COCO categories:
- `Left` (ID 1): `2,535` (`30.92%`)
- `Right` (ID 2): `2,590` (`31.59%`)
- `Unidentifiable` (ID 3): `3,074` (`37.49%`)
- `Ambiguous` (ID 4): `0` (`0.00%`)

Because the Kaggle competition evaluates class-agnostic filament instance segmentation (`filament_id`, `segmentation_rle`), we collapse categories 1–3 into a single foreground class (`1`).

![Original solar disk image alongside color-coded filament instances and chirality categories](/blog/solar-filament/instance_segmentation_overlay.png)

What really jumped out wasn't the class balance — it was the **shape** of the targets:

| Metric | Median | 75th Percentile | 90th Percentile | Max |
|---|---:|---:|---:|---:|
| Mask area (px) | `1,228.0` | — | `4,684.6` | `37,739.0` |
| BBox width (px) | `74.52` | `126.21` | `202.48` | `1,067.31` |
| BBox height (px) | `69.61` | `116.59` | `186.51` | `709.81` |
| Aspect ratio | `1.66` | `2.36` | `3.23` | `12.78` |
| **BBox occupancy (`mask / bbox`)** | **`24.30%`** | `36.15%` | `47.79%` | `87.54%` |

Pause on two numbers there:
1. A median filament is `1,228 px` inside a `4,194,304 px` image — **0.029% of the frame**.
2. Its mask fills less than a **quarter** (`24.3%`) of its own axis-aligned bounding box.

If you resize a `2048 × 2048` solar disk down to `512 × 512` to feed it into a standard detector, you crush a thin `1,228 px` ribbon down by 16× in area. You lose the very structure you're trying to segment.

We also checked whether ground-truth instances overlap inside a single record (all 1,154 records / 8,199 masks):
- Empty masks: `0`
- Invalid masks: `0`
- Records with *any* overlapping pixels: only `5` out of `1,154`
- Pairs with IoU ≥ 0.50 in the entire dataset: **just 1** (`010403-20160810234034Lh`, two `Right` polygons with IoU `0.7242`, almost certainly an annotator duplicate).

So in the ground truth, filaments are effectively mutually exclusive — matching the competition's strict zero-overlap submission rule.

---

## Tiling — And the "Smart" Idea We Audited and Threw Away

Since downsampling destroys small filaments, we kept native pixel resolution and sliced each `2048 × 2048` image into overlapping patches.

We compared three patch sizes with 25% overlap (`overlap = patch_size // 4`) and edge-anchored final coordinates (so the last tile always ends cleanly at pixel `2048`):

| Patch Size | Overlap | Stride | Patches / Image | Fully Contained Instances | Containment Rate |
|---:|---:|---:|---:|---:|---:|
| `512 × 512` | `128` | `384` | `25` | `7,448 / 8,199` | `90.84%` |
| **`768 × 768`** | **`192`** | **`576`** | **`16`** | **`7,944 / 8,199`** | **`96.89%`** |
| `1024 × 1024` | `256` | `768` | `9` | `8,095 / 8,199` | `98.73%` |

`768 × 768` with `192` overlap (`positions = [0, 576, 1152, 1280]`, 16 patches per image) gave us `96.89%` full-instance containment while keeping GPU memory modest.

Now, here is where an intuitively "clean" idea almost wrecked our training set.

Our first dataset prototype thought: *"If a patch boundary slices a filament in half, that's a truncated ground-truth target. What if we simply exclude any patch that cuts across an instance?"*

Before training on it, we wrote an audit script (preserved in [`patches.py`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/src/solar_filament/data/patches.py)) to measure what strict filtering actually did across all 707 images:

| Patch Config | Valid Patches Kept | Instances with Clean Coverage | Images Left with 0 Valid Patches |
|---|---:|---:|---:|
| `768 × 768` (overlap `192`) | `6,269 / 11,312` (`55.42%`) | `4,070 / 8,199` (**`49.64%`**) | `4` |
| `1024 × 1024` (overlap `256`) | `3,405 / 6,363` (`53.51%`) | `5,163 / 8,199` (**`62.97%`**) | `46` |

Dropping boundary-crossing patches threw away **half of all filaments** in the dataset.

So we ditched strict filtering completely. The final dataset keeps all 16 patches per image (`600 × 16 = 9,600` train patches, `107 × 16 = 1,712` val patches), clips any intersecting mask to the patch window, and recomputes a tight `[x1, y1, x2, y2]` bounding box around the clipped pixels. On a 200-patch random sample of the training set, `71.0%` of patches contained at least one filament (median `2.0`, max `7`) and `29.0%` were clean background negatives.

---

## Adapting Mask R-CNN — And Why FP16 Blew Up

For the model, we started from Torchvision's COCO-pretrained `maskrcnn_resnet50_fpn` (`43,916,123` total parameters; `43,703,131` trainable under Torchvision's default `trainable_backbone_layers=3`) and adapted it cleanly in [`mask_rcnn.py`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/src/solar_filament/models/mask_rcnn.py):

```python
# 1. Convert 3-channel RGB stem to 1-channel grayscale by averaging pretrained RGB weights
new_conv = nn.Conv2d(
    in_channels=1,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=False,
)
with torch.no_grad():
    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
model.backbone.body.conv1 = new_conv

# 2. Replace box & mask predictors for 2 classes (0: background, 1: filament)
model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes=2)
model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes=2)

# 3. Match our GONG H-alpha grayscale statistics and disable internal resizing
model.transform.image_mean = [0.326]
model.transform.image_std = [0.217]
```

Then came the next surprise.

We ran single-batch memory smoke tests with `torch.autocast("cuda", dtype=torch.float16)` on an NVIDIA A100-SXM4-80GB:
- Batch size `1`: `1.17 GB` peak VRAM
- Batch size `2`: `2.01 GB` peak VRAM
- Batch size `4`: `3.76 GB` peak VRAM

Everything looked great — until we ran full training under FP16, which **produced non-finite gradients**.

On an 80 GB A100, FP32 at batch size 4 only used ~`4.2 GB` of VRAM anyway. So instead of wrestling with mixed-precision scaling quirks in Mask R-CNN's RPN/ROI heads, we switched to **pure FP32** (`use_amp: false`), added gradient clipping (`max_norm = 1.0`, `error_if_nonfinite = True`), and wrapped every training batch with explicit finite-loss and finite-gradient checks (aborting if more than `1%` of batches are skipped).

One subtle Torchvision detail worth mentioning: `maskrcnn_resnet50_fpn` only returns a loss dictionary when `model.train()` is active. If you blindly call `model.train()` during validation, every BatchNorm layer updates its running statistics on validation data. In [`trainer.py`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/src/solar_filament/training/trainer.py), our validation function runs `model.train()` inside `torch.no_grad()` while temporarily forcing every `_BatchNorm` submodule to `.eval()` and restoring it in a `finally` block.

---

## Training Results: Rock-Solid Stability, Immediate Overfitting

We trained on Google Colab (`NVIDIA A100-SXM4-80GB`, Python `3.13.16`, PyTorch `2.14.1+cu130`, Torchvision `0.29.1+cu130`) with:
- Optimizer: `AdamW(lr=1e-4, weight_decay=1e-4)`
- Scheduler: `CosineAnnealingLR(T_max=12, eta_min=1e-6)`
- Batch size: `4` (`2,400` train batches and `428` val batches per epoch)
- Early stopping: `patience=4`, `min_delta=0.001` on validation loss

Here are the actual epoch-by-epoch numbers (`0.57 hours` total, `0` skipped batches across the entire run):

| Epoch | LR | Train Loss | Val Loss | Train Mask | Val Mask | Train Obj | Val Obj | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | `1.000e-4` | `0.45668` | **`0.41655`** | `0.23219` | `0.21231` | `0.03172` | `0.03409` | `4.14 GB` |
| 2 | `9.831e-5` | `0.38394` | `0.43285` | `0.20751` | `0.21268` | `0.02310` | `0.03572` | `4.25 GB` |
| 3 | `9.337e-5` | `0.34356` | `0.43748` | `0.19647` | `0.20681` | `0.01818` | `0.05534` | `4.18 GB` |
| 4 | `8.550e-5` | `0.31131` | `0.43854` | `0.18521` | `0.21548` | `0.01340` | `0.04631` | `4.23 GB` |
| 5 | `7.525e-5` | `0.28526` | `0.46518` | `0.17484` | `0.21891` | `0.01005` | `0.06855` | `4.40 GB` |

Early stopping triggered after Epoch 5 and restored the **Epoch 1** checkpoint (`val_loss = 0.41655`).

Now, let's be honest about what this table is telling us: **the model started overfitting right after Epoch 1.** Training loss dropped smoothly every epoch, while validation RPN objectness loss literally doubled (`0.03409 → 0.06855`).

Three factors are staring right at us:
1. **Zero data augmentation** on only `600` unique physical training images.
2. **`lr=1e-4` with AdamW** on `43.7M` unfrozen parameters (no warmup, no backbone freezing).
3. **We only measured validation *loss*, not COCO mask AP or Dice.** Look closely at `Val Mask` in the table above: it actually ticked *down* to `0.20681` at Epoch 3 even while total validation loss climbed because of RPN objectness. Without a true instance-segmentation metric on the full-resolution validation split, we cannot claim Epoch 1 actually had higher mask AP than Epoch 2 or 3.

---

## Stitching 16 Patches Back Into Non-Overlapping Full-Disk Masks

Inference is where tiled instance segmentation gets genuinely tricky. Because adjacent `768 × 768` patches overlap by `192` pixels, the same filament often gets detected two, three, or four times in neighboring crops. Meanwhile, the competition requires **zero overlapping pixels** between any two predicted instances in an image.

Our reconstruction pipeline in [`postprocess.py`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/src/solar_filament/inference/postprocess.py) and [`predictor.py`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/src/solar_filament/inference/predictor.py) works in four stages:

1. **Tiled FP32 inference**: Extract all 16 `768 × 768` patches and run the model in batches of 4 under `torch.inference_mode()`.
2. **Per-patch filtering**: Keep detections with `score >= 0.50`, binarize masks at `>= 0.50`, drop masks smaller than `20 px`, and project cropped masks back into global `2048 × 2048` coordinates.
3. **Union-Find duplicate merging**: Group any two detections whose bounding boxes satisfy `bbox_iou >= 0.30` OR `bbox_containment >= 0.50` (`intersection / min(area_a, area_b)`). Each connected component is combined via pixelwise logical OR and inherits the maximum confidence score in the group.
4. **Confidence-ordered exclusive overlap resolution**: Sort merged instances descending by score. Each instance claims its pixels; lower-confidence instances have already-claimed pixels subtracted (`mask & ~claimed`) and are dropped if their remaining area falls below `20 px`.

Finally, every exclusive mask is encoded as Fortran-contiguous `uint8` COCO RLE via `pycocotools.mask.encode`, written to `submission.csv`, and then **decoded back to `2048 × 2048`** by our validator ([`validator.py`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/src/solar_filament/submission/validator.py)) to guarantee zero overlap and valid dimensions.

Running the Epoch 1 checkpoint across all **180 test images** produced:
- **Raw patch detections**: `6,599`
- **Merged instances**: `1,689`
- **Final exclusive instances**: `1,689` (`submission.csv` shape `(1689, 2)`)
- **Unique predicted test images**: `179 / 180` (`1` test image had zero detections above `0.50`)
- **Overlapping instances after full RLE decode audit**: **`0`**

---

## One Postprocessing Flaw You Should Watch Out For

Remember earlier when we discovered that solar filaments fill a median of only **`24.3%`** of their bounding boxes?

That comes back to bite bounding-box-based Union-Find merging. Two completely separate curved or diagonal filaments sitting near each other can easily have bounding boxes with `bbox_iou >= 0.30` or `containment >= 0.50` without touching at a single foreground pixel — and Union-Find will happily merge them into one instance. Even worse, Union-Find transitivity can chain neighboring detections together.

We kept the exact bounding-box merging logic in the repository so the baseline matches our actual experimental run, but replacing it with **pixel-mask IoU / mask containment** is the single cleanest postprocessing upgrade we would make next.

---

## What We Would Fix Next

If you clone the repo and want to push this from a clean, verified baseline to a stronger competition entry, here are the highest-leverage next steps:

1. **Evaluate real instance-segmentation metrics on validation** (COCO mask AP, Dice, or competition IoU matching at full `2048 × 2048` resolution) instead of selecting checkpoints by patch-level loss alone.
2. **Add data augmentation** (flips, 90° rotations, mild contrast/gamma jitter) and tune learning rate / warmup to stop the Epoch 2+ overfitting.
3. **Switch duplicate merging from bounding-box IoU to pixel-mask IoU** so nearby curved filaments aren't falsely merged.
4. **Handle multi-annotator disagreement more deliberately** — for example, evaluating validation against all available annotators per image rather than only `record_ids[0]`, or experimenting with consensus masks.

---

## Explore the Code & Reproduce the Pipeline

Everything described in this post has been extracted out of the original exploratory notebook into a clean, tested Python package:

- **Repository**: [https://github.com/Black-fox17/solar-filament-instance-segmentation](https://github.com/Black-fox17/solar-filament-instance-segmentation)
- **Original exploratory notebook** (preserved verbatim with all A100 outputs): [`notebooks/01_original_investigation.ipynb`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/notebooks/01_original_investigation.ipynb)
- **Clean reproducible pipeline notebook**: [`notebooks/02_reproducible_pipeline.ipynb`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/notebooks/02_reproducible_pipeline.ipynb)
- **Detailed research notes**: [`docs/dataset_analysis.md`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/docs/dataset_analysis.md), [`docs/methodology.md`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/docs/methodology.md), and [`docs/experiments_and_limitations.md`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/docs/experiments_and_limitations.md)

To get started locally (after downloading `MAGFiLO_1.0_Kaggle_2026` under the competition's terms):

```bash
git clone https://github.com/Black-fox17/solar-filament-instance-segmentation.git
cd solar-filament-instance-segmentation
pip install -r requirements.txt
pip install -e .

# Audit annotations & patch coverage
python scripts/audit_dataset.py --config configs/default.yaml --dataset-root data/MAGFiLO_1.0_Kaggle_2026

# Train in FP32
python scripts/train.py --config configs/default.yaml --dataset-root data/MAGFiLO_1.0_Kaggle_2026

# Run tiled inference & validate submission.csv
python scripts/predict.py --config configs/default.yaml --dataset-root data/MAGFiLO_1.0_Kaggle_2026 --checkpoint checkpoints/best_model.pth
```

*(Note: Trained `.pth` weights from the ephemeral Colab workspace are not committed to Git; see [`checkpoints/README.md`](https://github.com/Black-fox17/solar-filament-instance-segmentation/blob/main/checkpoints/README.md) for checkpoint format details and placement.)*
