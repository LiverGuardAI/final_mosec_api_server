#!/usr/bin/env python3
"""
Minimal nnU-Net v2 + ONNX Runtime inference script
Flow:
1) Load nnU-Net predictor
2) Export ONNX with dummy input (once)
3) Load ONNX Runtime session
4) Preprocess input with nnU-Net
5) Run ONNX Runtime inference
6) Postprocess with nnU-Net and save
"""

import os
import sys
import argparse

import torch
import numpy as np
import SimpleITK as sitk
import onnxruntime as ort

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)

if not hasattr(torch, "GradScaler"):
    try:
        torch.GradScaler = torch.cuda.amp.GradScaler
    except AttributeError:
        torch.GradScaler = torch.amp.GradScaler


class ORTModule(torch.nn.Module):
    def __init__(
        self,
        session: ort.InferenceSession,
        input_name: str,
        output_name: str,
        fp16: bool,
        num_classes: int,
        use_io_binding: bool,
    ):
        super().__init__()
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.fp16 = fp16
        self.num_classes = num_classes
        self.use_io_binding = use_io_binding

    def load_state_dict(self, *args, **kwargs):  # no-op for ORT
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = torch.float16 if self.fp16 else torch.float32
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        if self.use_io_binding and x.is_cuda:
            np_dtype = np.float16 if self.fp16 else np.float32
            device_id = x.device.index or 0
            io_binding = self.session.io_binding()
            io_binding.bind_input(
                self.input_name,
                "cuda",
                device_id,
                np_dtype,
                tuple(x.shape),
                x.data_ptr(),
            )
            output_shape = (x.shape[0], self.num_classes, *x.shape[2:])
            out = torch.empty(output_shape, device=x.device, dtype=target_dtype)
            io_binding.bind_output(
                self.output_name,
                "cuda",
                device_id,
                np_dtype,
                output_shape,
                out.data_ptr(),
            )
            self.session.run_with_iobinding(io_binding)
            return out

        np_dtype = np.float16 if self.fp16 else np.float32
        x_np = x.detach().cpu().numpy().astype(np_dtype, copy=False)
        out = self.session.run(None, {self.input_name: x_np})[0]
        return torch.from_numpy(out.astype(np_dtype, copy=False))


def export_onnx(model: torch.nn.Module, dummy_shape, onnx_path: str, fp16: bool):
    model_device = next(model.parameters()).device
    dummy_dtype = torch.float16 if fp16 else torch.float32
    dummy = torch.randn(*dummy_shape, device=model_device, dtype=dummy_dtype)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
            do_constant_folding=True,
        )

def export_and_simplify_onnx(model: torch.nn.Module, dummy_shape, onnx_path: str, onnx_simp_path: str, fp16: bool):
    import onnx
    from onnxsim import simplify

    export_onnx(model, dummy_shape, onnx_path, fp16)
    onnx_model = onnx.load(onnx_path)
    onnx_model_simp, check = simplify(onnx_model)
    if not check:
        raise RuntimeError("onnxsim simplify check failed")
    onnx.save(onnx_model_simp, onnx_simp_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model_folder")
    p.add_argument("input_file")
    p.add_argument("output_dir")
    p.add_argument("folds", help="e.g. 0 or 0,1,2")
    p.add_argument("checkpoint")
    p.add_argument("device", help="cuda or cpu")
    p.add_argument("--onnx", default="nnunet_fp16.onnx")
    p.add_argument("--onnx_simp", default="nnunet_fp16_sim.onnx")
    p.add_argument("--rebuild", action="store_true", help="Force ONNX export")
    p.add_argument("--dummy", default="128,128,128", help="Dummy D,H,W for ONNX export")
    p.add_argument("--fp16", action="store_true", help="Export ONNX in FP16")
    return p.parse_args()


def main():
    args = parse_args()

    folds = tuple(map(int, args.folds.split(",")))
    device = torch.device(args.device)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True,
    )

    print(f"Loading model from: {args.model_folder}")
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        use_folds=folds,
        checkpoint_name=args.checkpoint,
    )

    model = predictor.network.eval().to(device)
    if args.fp16:
        model = model.half()

    dummy_shape = tuple(map(int, args.dummy.split(",")))
    desired_shape = (1, 1, *dummy_shape)

    if args.rebuild or (not os.path.exists(args.onnx_simp)):
        print("Exporting nnU-Net network to ONNX...")
        export_and_simplify_onnx(model, desired_shape, args.onnx, args.onnx_simp, args.fp16)

    available_providers = ort.get_available_providers()
    if device.type == "cuda" and "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    onnx_to_load = args.onnx_simp if os.path.exists(args.onnx_simp) else args.onnx
    print(f"Loading ONNX Runtime session: {onnx_to_load}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_to_load, sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Loading input image: {args.input_file}")
    img = sitk.ReadImage(args.input_file)
    img_data = sitk.GetArrayFromImage(img)
    if img_data.ndim == 3:
        img_data = img_data[None]

    original_spacing = np.array(img.GetSpacing())[::-1]
    print(f"Input numpy shape (C,D,H,W): {img_data.shape}, spacing(D,H,W): {original_spacing}")

    if hasattr(predictor, "preprocess_single_npy_array"):
        data, _, properties = predictor.preprocess_single_npy_array(
            img_data,
            {"spacing": original_spacing},
        )
    else:
        ppa = PreprocessAdapterFromNpy(
            [img_data],
            [None],
            [{"spacing": original_spacing}],
            [None],
            predictor.plans_manager,
            predictor.dataset_json,
            predictor.configuration_manager,
            num_threads_in_multithreaded=1,
            verbose=predictor.verbose,
        )
        dct = next(ppa)
        data = dct["data"]
        properties = dct["data_properties"]

    if isinstance(data, torch.Tensor):
        data = data.cpu()
    data = torch.as_tensor(data, dtype=torch.float32)

    use_io_binding = device.type == "cuda" and "CUDAExecutionProvider" in providers
    predictor.network = ORTModule(
        session,
        input_name,
        output_name,
        args.fp16,
        predictor.label_manager.num_segmentation_heads,
        use_io_binding,
    )
    predictor.network.eval()

    if args.fp16:
        data = data.half()
    predicted_logits = predictor.predict_logits_from_preprocessed_data(data)
    if isinstance(predicted_logits, torch.Tensor):
        predicted_logits = predicted_logits.cpu().numpy()
    if predicted_logits.ndim == 5 and predicted_logits.shape[0] == 1:
        predicted_logits = predicted_logits[0]

    segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.label_manager,
        properties,
    ).astype(np.uint8)

    out_img = sitk.GetImageFromArray(segmentation)
    out_img.SetSpacing(img.GetSpacing())
    out_img.SetOrigin(img.GetOrigin())
    out_img.SetDirection(img.GetDirection())

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, os.path.basename(args.input_file))
    print(f"Saving: {out_path}")
    sitk.WriteImage(out_img, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
