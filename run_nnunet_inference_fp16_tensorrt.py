#!/usr/bin/env python3
"""
Minimal nnU-Net v2 + TensorRT FP16 inference script
Flow:
1) Load nnU-Net predictor
2) Export ONNX with dummy input (once) + simplify
3) Build TensorRT engine
4) Load engine
5) Preprocess input with nnU-Net
6) Run TensorRT inference via nnU-Net sliding window
7) Postprocess with nnU-Net and save
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
import SimpleITK as sitk
import tensorrt as trt

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


def export_and_simplify_onnx(model: torch.nn.Module, dummy_shape, onnx_path: str, onnx_simp_path: str):
    import onnx
    from onnxsim import simplify

    model_device = next(model.parameters()).device
    dummy = torch.randn(*dummy_shape, device=model_device, dtype=torch.float32)

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

    onnx_model = onnx.load(onnx_path)
    onnx_model_simp, check = simplify(onnx_model)
    if not check:
        raise RuntimeError("onnxsim simplify check failed")
    onnx.save(onnx_model_simp, onnx_simp_path)


def build_trt_engine_fp16(onnx_simp_path: str, engine_path: str, workspace_gb: int = 8):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_simp_path, "rb") as f:
        onnx_bytes = f.read()

    if not parser.parse(onnx_bytes):
        print("ERROR: ONNX parsing failed. Parser errors:", file=LOG_STREAM, flush=True)
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=LOG_STREAM, flush=True)
        raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb) * (1 << 30)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    config.set_flag(trt.BuilderFlag.FP16)

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Failed to build TensorRT serialized engine")
        with open(engine_path, "wb") as f:
            f.write(serialized)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())


class TRTInferencer:
    def __init__(self, engine_path: str, device: torch.device):
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1 or len(self.output_names) != 1:
            raise RuntimeError("Expected 1 input and 1 output tensor")

        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        self.input_tensor = None
        self.output_tensor = None

    def _ensure_buffer(self, name: str, shape: tuple, dtype: np.dtype):
        if name == "input":
            current = self.input_tensor
        else:
            current = self.output_tensor
        needs_alloc = (
            current is None
            or tuple(current.shape) != tuple(shape)
            or current.dtype != torch.from_numpy(np.empty((), dtype=dtype)).dtype
        )
        if needs_alloc:
            torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)
            if name == "input":
                self.input_tensor = tensor
            else:
                self.output_tensor = tensor

    def infer(self, input_np: np.ndarray) -> np.ndarray:
        if input_np.dtype != self.input_dtype:
            input_np = input_np.astype(self.input_dtype, copy=False)

        input_shape = tuple(input_np.shape)
        if any(dim < 0 for dim in self.input_shape) or input_shape != self.input_shape:
            if not self.context.set_input_shape(self.input_name, input_shape):
                raise RuntimeError(f"Failed to set input shape: {input_shape}")
            self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        else:
            self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        self._ensure_buffer("input", input_shape, self.input_dtype)
        self._ensure_buffer("output", self.output_shape, self.output_dtype)

        input_torch = torch.from_numpy(input_np).to(device=self.device)
        self.input_tensor.copy_(input_torch)
        self.context.set_tensor_address(self.input_name, self.input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ok = self.context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()

        return self.output_tensor.cpu().numpy()


class TRTModule(torch.nn.Module):
    def __init__(self, trt_engine: TRTInferencer):
        super().__init__()
        self.trt_engine = trt_engine

    def load_state_dict(self, *args, **kwargs):  # no-op for TRT
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.trt_engine.input_dtype
        x_np = x.detach().cpu().numpy().astype(dtype, copy=False)
        out_np = self.trt_engine.infer(x_np)
        return torch.from_numpy(out_np.astype(self.trt_engine.output_dtype, copy=False))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model_folder")
    p.add_argument("input_file", nargs="?")
    p.add_argument("output_dir", nargs="?")
    p.add_argument("folds", help="e.g. 0 or 0,1,2")
    p.add_argument("checkpoint")
    p.add_argument("device", help="cuda or cpu (TensorRT requires cuda)")
    p.add_argument("--engine", default="nnunet_fp16.trt")
    p.add_argument("--onnx", default="nnunet_fp16.onnx")
    p.add_argument("--onnx_simp", default="nnunet_fp16_sim.onnx")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild ONNX + TRT engine")
    p.add_argument("--dummy", default="128,128,128", help="Dummy D,H,W for fixed ONNX, e.g. 128,128,128")
    p.add_argument("--workspace_gb", type=int, default=8)
    p.add_argument("--server", action="store_true", help="Run in server mode with JSON lines over stdin/stdout")
    return p.parse_args()


def initialize_predictor(args):
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

    print(f"Loading model from: {args.model_folder}", file=LOG_STREAM, flush=True)
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        use_folds=folds,
        checkpoint_name=args.checkpoint,
    )

    model = predictor.network.eval().to(device)

    dummy_shape = tuple(map(int, args.dummy.split(",")))
    desired_shape = (1, 1, *dummy_shape)

    if args.rebuild or (not os.path.exists(args.onnx_simp)):
        print("Exporting nnU-Net network to ONNX...", file=LOG_STREAM, flush=True)
        export_and_simplify_onnx(model, desired_shape, args.onnx, args.onnx_simp)

    if args.rebuild or (not os.path.exists(args.engine)):
        print("Building TensorRT FP16 engine...", file=LOG_STREAM, flush=True)
        build_trt_engine_fp16(args.onnx_simp, args.engine, workspace_gb=args.workspace_gb)

    print(f"Loading TensorRT engine: {args.engine}", file=LOG_STREAM, flush=True)
    trt_engine = TRTInferencer(args.engine, device)

    predictor.network = TRTModule(trt_engine)
    predictor.network.eval()

    return predictor, trt_engine


def run_single_inference(predictor, input_file: str, output_dir: str) -> str:
    print(f"Loading input image: {input_file}", file=LOG_STREAM, flush=True)
    img = sitk.ReadImage(input_file)
    img_data = sitk.GetArrayFromImage(img)
    if img_data.ndim == 3:
        img_data = img_data[None]

    original_spacing = np.array(img.GetSpacing())[::-1]
    print(
        f"Input numpy shape (C,D,H,W): {img_data.shape}, spacing(D,H,W): {original_spacing}",
        file=LOG_STREAM,
        flush=True,
    )

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

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(input_file))
    print(f"Saving: {out_path}", file=LOG_STREAM, flush=True)
    sitk.WriteImage(out_img, out_path)
    print("Done.", file=LOG_STREAM, flush=True)
    return out_path


def run_server(predictor, out_stream):
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            input_file = payload.get("input_file")
            output_dir = payload.get("output_dir")
            if not input_file or not output_dir:
                raise ValueError("input_file and output_dir are required")
            output_file = run_single_inference(predictor, input_file, output_dir)
            response = {"ok": True, "output_file": output_file}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        out_stream.write(json.dumps(response) + "\n")
        out_stream.flush()


def main():
    global LOG_STREAM
    args = parse_args()
    out_stream = sys.stdout
    if args.server:
        sys.stdout = sys.stderr
        LOG_STREAM = sys.stderr
    else:
        LOG_STREAM = sys.stdout
    if not args.server and (not args.input_file or not args.output_dir):
        raise SystemExit("input_file and output_dir are required unless --server is set")

    predictor, _trt_engine = initialize_predictor(args)

    if args.server:
        run_server(predictor, out_stream)
    else:
        run_single_inference(predictor, args.input_file, args.output_dir)


if __name__ == "__main__":
    LOG_STREAM = sys.stdout
    main()
