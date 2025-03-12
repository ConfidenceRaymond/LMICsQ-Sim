import torch, os
import torchio as tio
import nibabel as nib
import numpy as np
import json
from torchio.utils import history_collate

# Custom Transforms
class RandomCrop(tio.Transform):
    def __init__(self, crop_percentage_range=(0.2, 0.5), min_fov_percentage=0.5, p=0.1):
        super().__init__()
        self.crop_percentage_range = crop_percentage_range
        self.min_fov_percentage = min_fov_percentage
        self.p = p

    def apply_transform(self, subject):
        if np.random.rand() < self.p:
            d, h, w = subject.get_first_image().shape[1:]
            min_dim = min(d, h, w)
            max_crop_percentage = 1.0 - self.min_fov_percentage
            crop_percentage = np.random.uniform(max(self.crop_percentage_range[0], 0.0),
                                            min(self.crop_percentage_range[1], max_crop_percentage))
            crop_total = int(min_dim * crop_percentage)
            crop_total = crop_total if crop_total % 2 == 0 else crop_total + 1
            d_front = d_back = min(crop_total // 2, d // 2)
            h_front = h_back = min(crop_total // 2, h // 2)
            w_front = w_back = min(crop_total // 2, w // 2)
            return tio.Crop((d_front, d_back, h_front, h_back, w_front, w_back))(subject)
        return subject

class RandomPatchDegradation(tio.Transform):
    def __init__(self, num_patches=5, intensity_range=(0.1, 0.3)):
        super().__init__()
        self.num_patches, self.intensity_range = num_patches, intensity_range

    def apply_transform(self, subject):
        data = subject.get_first_image().data.clone().to(torch.float32)
        d, h, w = data.shape[1:]
        for _ in range(self.num_patches):
            d_start, h_start, w_start = [np.random.randint(0, s // 2) for s in (d, h, w)]
            patch_size = np.random.randint(10, 30)
            intensity = np.random.uniform(*self.intensity_range)
            d_end = min(d_start + patch_size, d)
            h_end = min(h_start + patch_size, h)
            w_end = min(w_start + patch_size, w)
            data[:, d_start:d_end, h_start:h_end, w_start:w_end] += intensity
        data = torch.clamp(data, 0, 1)
        subject.get_first_image().data = data
        return subject

class SparseSpatialTransform(tio.Transform):
    def __init__(self, p_apply=0.75, max_effects=3, weights=(0.3, 0.5, 0.2)):
        super().__init__()
        self.p_apply, self.max_effects = p_apply, min(max_effects, 3)
        self.weights = weights / np.sum(weights)
        self.effects = [
            tio.RandomNoise(mean=0, std=(lambda: np.random.uniform(0.05, 0.15))()),  # RandomNoise -0.2, RandomAnisotropy -0.5 RandomBiasField - 0.2
            tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(2, 5)),
            tio.RandomBiasField(coefficients=(lambda: np.random.uniform(0.3, 0.7))())
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
        voxel_size = np.random.choice([2, 3])
        return tio.Resample(voxel_size)(subject)

# Core Functions
def get_degradation_pipeline():
    return tio.Compose([
        RandomResample(),
        SparseSpatialTransform(p_apply=0.6, max_effects=3, weights=(0.2, 0.5, 0.3)),  ## RandomNoise -0.2, RandomAnisotropy -0.5 RandomBiasField - 0.2
        tio.RandomBlur(std=(lambda: np.random.uniform(0.3, 0.7))(), p=0.3),
        tio.OneOf({
            tio.RandomMotion(degrees=(lambda: np.random.uniform(3, 7))(), translation=(lambda: np.random.uniform(3, 7))(), num_transforms=2): 1,
            tio.RandomSpike(num_spikes=(lambda: np.random.randint(5, 15))(), intensity=(lambda: np.random.uniform(0.1, 0.3))()): 2,
            tio.RandomGhosting(intensity=(lambda: np.random.uniform(0.1, 0.3))(), axes=(lambda: np.random.randint(0, 3))()): 2
        }, p=0.5),
        RandomCrop(p=0.1)  # Uniform crop for limited FOV
        #RandomPatchDegradation(num_patches=5, intensity_range=(0.1, 0.3)): 1
    ])

def get_individual_transforms():
    return {
        "Resample": RandomResample(),
        "Noise": tio.RandomNoise(mean=0, std=(lambda: np.random.uniform(0.05, 0.15))()),
        "Anisotropy": tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(2, 5)),
        "BiasField": tio.RandomBiasField(coefficients=(lambda: np.random.uniform(0.3, 0.7))()),
        "Blur": tio.RandomBlur(std=(lambda: np.random.uniform(0.3, 0.7))()),
        "Motion": tio.RandomMotion(degrees=(lambda: np.random.uniform(3, 7))(), translation=(lambda: np.random.uniform(3, 7))(), num_transforms=2),
        "Spike": tio.RandomSpike(num_spikes=(lambda: np.random.randint(5, 15))(), intensity=(lambda: np.random.uniform(0.1, 0.3))()),
        "Ghosting": tio.RandomGhosting(intensity=(lambda: np.random.uniform(0.1, 0.3))(), axes=(lambda: np.random.randint(0, 3))()),
        "Crop": RandomCrop(),  # Uniform crop for limited FOV
        "PatchDegradation": RandomPatchDegradation(num_patches=5, intensity_range=(0.1, 0.3))
    }

def save_transform_history(subjects, output_json='transform_history.json'):
    history = subjects.get_composed_history()
    print(history)
    with open(output_json, 'w') as f:
        json.dump(history, f, indent=4)
    return output_json

def degrade_mri(input_path, output_path, pipeline=None):
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        nifti_img = nib.load(input_path)
        original_data = nifti_img.get_fdata()
        mri = tio.Subject(
            img = tio.ScalarImage(tensor=original_data[..., np.newaxis]),
        )
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}. Please ensure the path is correct.")
        exit()  # Exit if file is not found.
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        exit()
        
    pipeline = pipeline or get_degradation_pipeline()
    degraded_mri = pipeline(mri)
    #mri = tio.SubjectsDataset(mri, transform=get_degradation_pipeline)
    history_path = output_path.replace('.nii', '_history.json')
    save_transform_history(mri, history_path)
    degraded_mri.save(output_path)
    
    return degraded_mri, history_path