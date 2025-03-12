import torch
import torchio as tio
import numpy as np
import json
from torchio.utils import history_collate

# Custom Transforms
class RandomCrop(tio.Transform):
    def apply_transform(self, subject):
        d, h, w = subject.get_first_image().shape[1:]
        return tio.Crop((np.random.randint(5, 15), np.random.randint(5, 15),
                         np.random.randint(10, 20), np.random.randint(10, 20),
                         np.random.randint(10, 20), np.random.randint(10, 20)))(subject)

class RandomPatchDegradation(tio.Transform):
    def __init__(self, num_patches=5, intensity_range=(0.1, 0.3)):
        super().__init__()
        self.num_patches, self.intensity_range = num_patches, intensity_range

    def apply_transform(self, subject):
        data = subject.get_first_image().data.clone()
        d, h, w = data.shape[1:]
        for _ in range(self.num_patches):
            d_start, h_start, w_start = [np.random.randint(0, s // 2) for s in (d, h, w)]
            patch_size = np.random.randint(10, 30)
            intensity = np.random.uniform(*self.intensity_range)
            data[:, d_start:d_start+patch_size, h_start:h_start+patch_size, w_start:w_start+patch_size] += intensity
        subject.get_first_image().data = torch.clamp(data, 0, 1)
        return subject

class LimitedOverlapTransform(tio.Transform):
    def __init__(self, p_apply=0.75, max_effects=3, weights=(0.4, 0.4, 0.2)):
        super().__init__()
        self.p_apply, self.max_effects = p_apply, min(max_effects, 3)
        self.weights = weights / np.sum(weights)
        self.effects = [
            tio.RandomNoise(mean=0, std=lambda: np.random.uniform(0.05, 0.15)),
            tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(2, 5)),
            tio.RandomBiasField(coefficients=lambda: np.random.uniform(0.3, 0.7))
        ]

    def apply_transform(self, subject):
        if np.random.rand() < self.p_apply:
            num_effects = np.random.choice([1, 2, 3], p=self.weights[:self.max_effects])
            chosen_indices = np.random.choice([0, 1, 2], size=num_effects, replace=False)
            for idx in chosen_indices:
                subject = self.effects[idx](subject)
        return subject

class RandomResample(tio.Transform):
    def apply_transform(self, subject):
        voxel_size = np.random.choice([2, 3])  # Randomly choose 2mm or 3mm
        return tio.Resample(voxel_size)(subject)

# Core Functions
def get_degradation_pipeline():
    """Return the full degradation pipeline with history logging."""
    return tio.Compose([
        RandomResample(),
        LimitedOverlapTransform(p_apply=0.75, max_effects=3, weights=(0.4, 0.4, 0.2)),
        tio.RandomBlur(std=lambda: np.random.uniform(0.3, 0.7), p=0.3),
        tio.OneOf({
            tio.RandomMotion(degrees=lambda: np.random.uniform(3, 7), translation=lambda: np.random.uniform(3, 7), num_transforms=2): 1,
            tio.RandomSpike(num_spikes=lambda: np.random.randint(5, 15), intensity=lambda: np.random.uniform(0.1, 0.3)): 2,
            tio.RandomGhosting(intensity=lambda: np.random.uniform(0.1, 0.3), axis=np.random.randint(0, 3)): 2
        }, p=0.7),
        tio.OneOf({
            RandomCrop(): 1,
            RandomPatchDegradation(num_patches=5, intensity_range=(0.1, 0.3)): 1
        }, p=0.5),
    ], log_history=True)

def get_individual_transforms():
    """Return a dictionary of individual transforms for demonstration."""
    return {
        "Resample": RandomResample(),
        "Noise": tio.RandomNoise(mean=0, std=0.1),
        "Anisotropy": tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(2, 5)),
        "BiasField": tio.RandomBiasField(coefficients=0.5),
        "Blur": tio.RandomBlur(std=0.5),
        "Motion": tio.RandomMotion(degrees=5, translation=5, num_transforms=2),
        "Spike": tio.RandomSpike(num_spikes=10, intensity=0.2),
        "Ghosting": tio.RandomGhosting(intensity=0.2),
        "Crop": RandomCrop(),
        "PatchDegradation": RandomPatchDegradation(num_patches=5, intensity_range=(0.1, 0.3))
    }

def save_transform_history(subjects, output_json='transform_history.json'):
    """Save the transform history of subjects to a JSON file."""
    history = history_collate(subjects)
    with open(output_json, 'w') as f:
        json.dump(history, f, indent=4)
    return output_json

def degrade_mri(input_path, output_path, pipeline=None):
    """Degrade a single MRI and save with history."""
    mri = tio.ScalarImage(input_path)
    pipeline = pipeline or get_degradation_pipeline()
    degraded_mri = pipeline(mri)
    degraded_mri.save(output_path)
    history_path = output_path.replace('.nii', '_history.json')
    save_transform_history([degraded_mri], history_path)
    return degraded_mri, history_path