# Low-Quality MRI Simulation

Simulate low-quality MRI scans from LMICs with `torchio`. Randomly resamples to 2mm or 3mm and tracks transform history.

## Features
- **Random Resample**: 2mm or 3mm voxel sizes.
- **Transform History**: Saved to JSON per image/batch.
- **CLI**: Single or batch degradation from the terminal.
- **Interactive Demo**: Visualize effects in Colab/Kaggle.

## Setup
```bash
git clone https://github.com/yourusername/LowQualityMRISimulation.git
cd LMICsQ-Sim
pip install -r requirements.txt
```


## Example Usage from Terminal
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



