import argparse
import os
import torchio as tio
from degrade_mri import get_degradation_pipeline, save_history_to_json, degrade_mri, remove_nifti_extensions
import glob
from tqdm import tqdm 


verbose = False

def degrade_single(input_path, output_path):
    """Degrade a single MRI and save with transform history."""
    # Load image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found")
    
    # Apply pipeline
    
    filename = remove_nifti_extensions(input_path)
    img_output_path = os.path.join(output_path, f"degraded_{filename}")
    history_path = os.path.join(output_path, f"{filename}_history.json")
    
    degraded_mri, _ = degrade_mri(input_path, img_output_path, history_path)
    
    print(f"Degraded MRI saved to {img_output_path}")
    print(f"Transform history saved to {history_path}")

def degrade_batch(input_dir, output_dir):
    
    input_files = glob.glob(os.path.join(input_dir, '*.nii')) + glob.glob(os.path.join(input_dir, '*.nii.gz'))
    if not input_files:
        raise FileNotFoundError(f"No NIfTI files found in {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "Batch_history.json")
    for input_path in tqdm(input_files, desc="Degrading MRI files"):
        pipeline = get_degradation_pipeline()
        filename = remove_nifti_extensions(input_path)
        output_path = os.path.join(output_dir, f"degraded_{filename}")
        degraded_mri, _ = degrade_mri(input_path, output_path, history_path)
        
        if verbose:
            print(f"Degraded MRI saved to {output_path}")
    
    print(f"Finished processing {len(input_files)} MRI files.")


def main():
    parser = argparse.ArgumentParser(description="Degrade MRI scans to simulate low-quality LMIC images.")
    parser.add_argument('-i', '--input', required=True, help="Input NIfTI file or directory")
    parser.add_argument('-o', '--output', type=str, default="outputs/", required=False, help="Output file path (single) or directory (batch)")
    parser.add_argument('--batch', action='store_true', help="Process as batch")
    
    args = parser.parse_args()
    
    if args.batch:
        degrade_batch(args.input, args.output)
    else:
        degrade_single(args.input, args.output)
        print(f"Degraded MRI saved to {args.output}")
        print(f"Transform history saved to {args.output.replace('.nii', '_history.json')}")

if __name__ == "__main__":
    main()