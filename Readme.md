# MRI Degradation Simulator for Low-Resource Imaging

This pipeline leverages `torchio` to degrade high-quality MRI scans, replicating the imaging artifacts and limitations prevalent in low- and middle-income country (LMIC) settings. It simulates the effects of outdated hardware, suboptimal calibration, and operational constraints, enabling robust testing of image analysis algorithms under realistic conditions. Below are the implemented degradation transforms:

## Degradation Transforms

### Low Resolution
To emulate the reduced spatial resolution of legacy LMIC scanners:
- **Resample**: `tio.Resample()` randomly adjusts voxel sizes to `[2, 3, 4, 5, 6]` mm, reflecting coarse sampling.
- **Anisotropy**: `tio.RandomAnisotropy()` downsamples a single axis (`[0, 1, 2]`) by a factor of 2–6, producing anisotropic voxels (e.g., 1x1x5mm), typical of rapid acquisitions.
- **Universal Application**: One of these is applied to all images to ensure consistent resolution degradation.

### Magnetic Field Inhomogeneity (MFI)
To replicate contrast distortions from poorly shimmed magnets:
- **Bias Field**: `tio.RandomBiasField` introduces intensity variations (coefficients 0.3–0.7) with a 30% probability, simulating field non-uniformity common in aging systems.

### Noise Artifacts
To model signal corruption from inadequate shielding or power instability:
- **Gaussian Noise**: `tio.RandomNoise` applies Gaussian noise (std 0.05–0.15).
- **Rician Noise**: `RandomRicianNoise` (std 0.05–0.15), weighted 2:1, reflects MRI-specific noise profiles.
- **Probability**: Applied to 20% of images, balancing realism and frequency.

### Motion Artifacts
To simulate patient motion during prolonged scan times:
- **Motion**: `tio.RandomMotion` (degrees & translation 3–7, weighted 2:1) introduces displacement artifacts.
- **Blur**: `tio.RandomBlur` (std 0.3–0.7, 30% internal probability) mimics motion-induced softening.
- **Occurrence**: Affects 10% of images, consistent with clinical motion prevalence.

### Compression Artifacts
To approximate data loss from aggressive compression:
- **Spikes**: `tio.RandomSpike` (5–15 spikes, intensity 0.1–0.3) introduces block-like distortions, applied to 5% of images.

### MRI Ghosting
To emulate aliasing from scanner instability or physiological motion:
- **Ghosting**: `tio.RandomGhosting` (intensity 0.1–0.3, random axis `[0, 1, 2]`) affects 1% of images, reflecting rare but impactful artifacts.

### Limited Field of View (FOV)
To simulate incomplete anatomical coverage:
- **Crop**: `RandomCrop` uniformly reduces each axis by 20–50% (retaining 50% anatomy), applied to 10% of images, mimicking hardware or positioning limitations.

## Implementation

The degradation pipeline is implemented as a `tio.Compose` sequence:

```python
tio.Compose([
    RandomResample(),  # Resolution degradation
    tio.RandomBiasField(coefficients=(lambda: np.random.uniform(0.3, 0.7))(), p=0.3),  # MFI
    tio.OneOf({RandomRicianNoise(std=(0.05, 0.15)): 2, tio.RandomNoise(mean=0, std=(lambda: np.random.uniform(0.05, 0.15))()): 1}, p=0.2),  # Noise
    tio.OneOf({tio.RandomMotion(degrees=(lambda: np.random.uniform(3, 7))(), translation=(lambda: np.random.uniform(3, 7))(), num_transforms=2): 2, tio.RandomBlur(std=(lambda: np.random.uniform(0.3, 0.7))(), p=0.3): 1}, p=0.1),  # Motion
    tio.RandomSpike(num_spikes=(lambda: np.random.randint(5, 15))(), intensity=(lambda: np.random.uniform(0.1, 0.3))(), p=0.05),  # Compression
    tio.RandomGhosting(intensity=(lambda: np.random.uniform(0.1, 0.3))(), axes=(lambda: np.random.randint(0, 3))(), p=0.01),  # Ghosting
    RandomCrop(p=0.1)  # Limited FOV
])
```

## Setup
```bash
git clone https://github.com/yourusername/LowQualityMRISimulation.git
cd LMICsQ-Sim
pip install -r requirements.txt


### Example Usage from Terminal
**Single Image**:
Single Image:
Command: python cli_degrade.py -i input.nii -o degraded.nii
Loads input.nii, applies the degradation pipeline, saves to degraded.nii, and writes transform history to degraded_history.json.


```bash
python LMICsQ-Sim.py -i sample_data/high_quality_mri.nii.gz -o output/degraded.nii --single

Output: `output/degraded_mri1.nii.gz`, and `output/degraded_history.json`.

**Batch Image**:
Command: python cli_degrade.py -i input_dir -o output_dir --batch
Processes all .nii or .nii.gz files in input_dir, saves degraded versions to output_dir (e.g., degraded_file1.nii), and writes a single batch_transform_history.json for all images.

python LMICsQ-Sim.py -i sample_data -o output_batch --batch

Assuming `sample_data` has `mri1.nii.gz` and `mri2.nii.gz`
Output: `output_batch/degraded_mri1.nii.gz`, `output_batch/degraded_mri2.nii.gz`, and `output_batch/batch_transform_history.json`.



