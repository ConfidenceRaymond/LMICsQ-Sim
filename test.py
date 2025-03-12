# Imports
import torch
import torchio as tio
import matplotlib.pyplot as plt
from degrade_mri import get_degradation_pipeline, get_individual_transforms, degrade_mri, save_transform_history

# Load a sample MRI (adjust to a NIfTI file from the dataset)
mri_path = 'Sample_Data/MNI152_T1_2mm_brain.nii.gz'  # Example NIfTI path, adjust as needed
mri = tio.ScalarImage(mri_path)

# Visualize function
def show_slice(mri, title, ax):
    slice_idx = mri.shape[1] // 2
    ax.imshow(mri.data[0, slice_idx, :, :], cmap='gray')
    ax.set_title(title)
    ax.axis('off')

# Demo individual transforms and full pipeline
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()
show_slice(mri, 'Original', axes[0])

individual_transforms = get_individual_transforms()
transformed_mris = []
for i, (name, transform) in enumerate(individual_transforms.items(), 1):
    print(name)
    degraded_mri, _ = degrade_mri(mri_path, f'{name}_degraded.nii', tio.Compose([transform]))
    show_slice(degraded_mri, name, axes[i])
    transformed_mris.append(degraded_mri)