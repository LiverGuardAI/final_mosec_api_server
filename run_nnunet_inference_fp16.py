#!/usr/bin/env python
"""
Standalone nnU-Net inference script
Run as separate process to avoid multiprocessing daemon limitations
"""
import sys
import os
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main():
    if len(sys.argv) != 7:
        print("Usage: run_nnunet_inference.py <model_folder> <input_file> <output_dir> <folds> <checkpoint> <device>")
        sys.exit(1)

    model_folder = sys.argv[1]
    input_file = sys.argv[2]
    output_dir = sys.argv[3]
    folds = tuple(map(int, sys.argv[4].split(',')))
    checkpoint_name = sys.argv[5]
    device_str = sys.argv[6]

    # Convert device string to torch.device object
    device = torch.device(device_str)

    print(f"Initializing nnU-Net predictor...")
    print(f"Model folder: {model_folder}")
    print(f"Device: {device}")
    print(f"Device type: {device.type}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,  # Disable TTA for faster inference (8x speedup)
        perform_everything_on_device=(device.type == 'cuda'),  # Fixed: use device.type
        device=device,
        verbose=True,  # Enable verbose to see progress
        verbose_preprocessing=True,  # Enable preprocessing logs
        allow_tqdm=True,
    )

    # Load model
    print(f"Loading model from {model_folder}...")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name=checkpoint_name
    )

    print(f"Running inference on {input_file}...")

    # Use predict_single_npy_array to avoid multiprocessing issues
    import SimpleITK as sitk
    import numpy as np

    # Load input image
    print("Loading input image...")
    img = sitk.ReadImage(input_file)
    img_data = sitk.GetArrayFromImage(img)

    # Add channel dimension: (D, H, W) -> (C, D, H, W)
    if len(img_data.shape) == 3:
        img_data = img_data[None]  # Add channel dimension at the beginning

    # Get spacing and properties
    original_spacing = np.array(img.GetSpacing())[::-1]  # Reverse for numpy order

    print(f"Input shape: {img_data.shape}, spacing: {original_spacing}")

    # Run prediction (this handles preprocessing internally)
    print("Running prediction...")
    use_fp16 = (device.type == "cuda")

    print(f"Using FP16 AMP: {use_fp16}")

    with torch.cuda.amp.autocast(enabled=use_fp16):
        predicted_probabilities = predictor.predict_single_npy_array(
            img_data,
            {'spacing': original_spacing},
            None,
            None,
            False  # Don't save probabilities
        )

    # Get segmentation (argmax over classes)
    print("Processing prediction results...")
    if len(predicted_probabilities.shape) == 4:  # (C, D, H, W)
        segmentation = np.argmax(predicted_probabilities, axis=0).astype(np.uint8)
    else:
        segmentation = predicted_probabilities.astype(np.uint8)

    print(f"Segmentation shape: {segmentation.shape}")

    # Create output image
    output_img = sitk.GetImageFromArray(segmentation)
    output_img.SetSpacing(img.GetSpacing())
    output_img.SetOrigin(img.GetOrigin())
    output_img.SetDirection(img.GetDirection())

    # Save output
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    print(f"Saving segmentation to {output_file}...")
    sitk.WriteImage(output_img, output_file)

    print("Inference completed successfully")


if __name__ == '__main__':
    main()