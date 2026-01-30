import os
import sys
import tempfile
import shutil
import logging
import json
from typing import List, Dict
import zipfile
import torch
import requests
import SimpleITK as sitk
import numpy as np
from mosec import Server, Worker
from dotenv import load_dotenv

# Load local environment variables if present.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Orthanc 서버 설정
ORTHANC_BASE_URL = os.getenv('ORTHANC_URL', '')

class DICOMSegmentationWorker(Worker):
    """
    DICOM Series에서 Segmentation을 수행하는 Worker
    1. Orthanc에서 DICOM series 로드
    2. DICOM을 NIfTI로 변환
    3. nnU-Net inference 수행
    4. Segmentation mask를 DICOM으로 변환
    5. Orthanc에 저장
    """
    
    def __init__(self):
        super().__init__()
        # Model configuration
        self.model_folder = "HCC-TACE-Seg-nnU-Net-LiTS/nnUNet_results/Dataset001_HCCSeg/nnUNetTrainer__nnUNetPlans__3d_fullres"
        self.folds = (0,)
        self.checkpoint_name = "checkpoint_final.pth"
        self.timeout = 7200

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Temporary directory (will be created in forward_setup)
        self.temp_dir = None
        self.verbose = os.getenv("MOSEC_VERBOSE", "0") == "1"
        self.infer_proc = None

    def _log_detail(self, message: str, *args) -> None:
        if self.verbose:
            logger.info(message, *args)

    def _start_inference_process(self) -> None:
        import subprocess

        if self.infer_proc and self.infer_proc.poll() is None:
            return

        script_path = os.path.join(os.path.dirname(__file__), 'run_nnunet_inference_fp16_tensorrt.py')
        folds_str = ','.join(map(str, self.folds))
        device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'

        engine_path = os.path.join(os.path.dirname(__file__), 'nnunet_fp16.trt')
        onnx_path = os.path.join(os.path.dirname(__file__), 'nnunet_fp16.onnx')
        onnx_simp_path = os.path.join(os.path.dirname(__file__), 'nnunet_fp16_sim.onnx')

        cmd = [
            sys.executable,
            "-u",
            script_path,
            self.model_folder,
            "-",
            "-",
            folds_str,
            self.checkpoint_name,
            device_str,
            '--engine', engine_path,
            '--onnx', onnx_path,
            '--onnx_simp', onnx_simp_path,
            '--dummy', '128,128,128',
            '--workspace_gb', '4',
            '--server',
        ]

        if not os.path.exists(engine_path) or os.getenv("TRT_FORCE_REBUILD") == "1":
            cmd.append('--rebuild')
            self._log_detail("   TensorRT engine not found or rebuild forced, will build on this run")
        else:
            self._log_detail("   Using cached TensorRT engine")

        env = os.environ.copy()
        env['PATH'] = '/usr/local/cuda-12.6/bin:' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = (
            '/usr/lib/wsl/lib:'
            '/usr/local/cuda-12.6/lib64:'
            '/usr/local/cuda-12.6/targets/x86_64-linux/lib:'
            '/home/syw/final_project/final_mosec_api_server/venv/lib/python3.10/site-packages/tensorrt_libs'
        )
        env['CUDA_HOME'] = '/usr/local/cuda-12.6'
        env['PYTHONUNBUFFERED'] = '1'

        self._log_detail("   Starting persistent TensorRT subprocess...")
        self.infer_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )

    def _ensure_inference_process(self) -> None:
        if self.infer_proc is None or self.infer_proc.poll() is not None:
            self._start_inference_process()

    def _stop_inference_process(self) -> None:
        if not self.infer_proc:
            return
        try:
            self.infer_proc.terminate()
            self.infer_proc.wait(timeout=5)
        except Exception:
            try:
                self.infer_proc.kill()
            except Exception:
                pass
        finally:
            self.infer_proc = None

    def _request_inference(self, nifti_path: str, output_dir: str) -> str:
        self._ensure_inference_process()
        if not self.infer_proc or not self.infer_proc.stdin or not self.infer_proc.stdout:
            raise Exception("Inference subprocess is not available")

        payload = json.dumps({
            "input_file": nifti_path,
            "output_dir": output_dir,
        })
        try:
            self.infer_proc.stdin.write(payload + "\n")
            self.infer_proc.stdin.flush()
        except Exception:
            self._stop_inference_process()
            raise

        response_line = self.infer_proc.stdout.readline()
        if not response_line:
            self._stop_inference_process()
            raise Exception("Inference subprocess exited unexpectedly")

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            self._stop_inference_process()
            raise Exception(f"Failed to parse inference response: {response_line.strip()}") from exc
        if not response.get("ok"):
            raise Exception(response.get("error", "Inference subprocess failed"))
        return response.get("output_file")
    
    def forward_setup(self):
        """Initialize worker (predictor will be initialized in subprocess)"""
        logger.info("="*70)
        logger.info("🔧 Initializing DICOM Segmentation Worker")
        logger.info("="*70)
        sys.stdout.flush()

        # Check CUDA availability
        self._log_detail("PyTorch version: %s", torch.__version__)
        self._log_detail("CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            self._log_detail("CUDA version: %s", torch.version.cuda)
            self._log_detail("GPU count: %s", torch.cuda.device_count())
            self._log_detail("Current GPU: %s", torch.cuda.current_device())
            self._log_detail("GPU name: %s", torch.cuda.get_device_name(0))
            self._log_detail(
                "GPU memory: %.2f GB",
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )

        logger.info(f"Selected device: {self.device}")
        sys.stdout.flush()

        # Create temporary directory (prefer tmpfs to reduce disk I/O)
        tmp_root = os.getenv("MOSEC_TMPDIR")
        if not tmp_root and os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
            tmp_root = "/dev/shm"
        self.temp_dir = tempfile.mkdtemp(prefix="dicom_seg_", dir=tmp_root)
        self._log_detail("Temporary directory: %s", self.temp_dir)

        # Note: Predictor will be initialized in subprocess to avoid daemon process issues
        self._log_detail("Note: nnU-Net predictor will be initialized in subprocess for each inference")
        logger.info("="*70)
        logger.info("✅ Worker initialization complete!")
        logger.info("="*70)
        sys.stdout.flush()
    
    def forward(self, data: List[Dict]) -> List[Dict]:
        """
        Process DICOM segmentation requests

        Args:
            data: List of dicts containing series_id

        Returns:
            List of dicts with segmentation results
        """
        # Ensure initialization (in case forward_setup wasn't called)
        if self.temp_dir is None:
            logger.warning("WARNING: forward_setup was not called, initializing now...")
            sys.stdout.flush()
            self.forward_setup()

        results = []

        for idx, item in enumerate(data):
            import time
            start_time = time.time()
            dicom_dir = None

            try:
                series_id = item.get('series_id')
                if not series_id:
                    results.append({
                        "error": "Missing 'series_id' field in request",
                        "status": "failed"
                    })
                    continue

                logger.info("="*70)
                logger.info("🚀 Starting segmentation for series: %s", series_id)
                logger.info("="*70)
                sys.stdout.flush()

                # 1. Download DICOM series from Orthanc
                step_start = time.time()
                self._log_detail("📥 Step 1/5: Downloading DICOM from Orthanc...")
                sys.stdout.flush()
                dicom_dir = self.download_series_from_orthanc(series_id, idx)
                if not dicom_dir or not os.path.exists(dicom_dir):
                    raise Exception(f"Failed to download DICOM series: {series_id}")
                step_elapsed = time.time() - step_start
                self._log_detail("   ✓ Completed in %.2fs", step_elapsed)
                sys.stdout.flush()

                # 2. Convert DICOM to NIfTI
                step_start = time.time()
                self._log_detail("🔄 Step 2/5: Converting DICOM to NIfTI...")
                sys.stdout.flush()
                nifti_path = self.dicom_to_nifti(dicom_dir, idx)
                if not nifti_path or not os.path.exists(nifti_path):
                    raise Exception("Failed to convert DICOM to NIfTI")
                step_elapsed = time.time() - step_start
                self._log_detail("   ✓ Completed in %.2fs", step_elapsed)
                sys.stdout.flush()

                # 3. Run nnU-Net inference
                step_start = time.time()
                self._log_detail("🧠 Step 3/5: Running nnU-Net inference (this may take a while)...")
                sys.stdout.flush()
                mask_nifti_path = self.run_inference(nifti_path, idx)
                if not mask_nifti_path or not os.path.exists(mask_nifti_path):
                    raise Exception("Failed to run nnU-Net inference")
                step_elapsed = time.time() - step_start
                self._log_detail("   ✓ Completed in %.2fs", step_elapsed)
                sys.stdout.flush()

                # 4. Convert mask to DICOM
                step_start = time.time()
                self._log_detail("🔄 Step 4/5: Converting mask to DICOM...")
                sys.stdout.flush()
                mask_dicom_dir = self.nifti_to_dicom(mask_nifti_path, dicom_dir, idx)
                if not mask_dicom_dir or not os.path.exists(mask_dicom_dir):
                    raise Exception("Failed to convert mask to DICOM")
                step_elapsed = time.time() - step_start
                self._log_detail("   ✓ Completed in %.2fs", step_elapsed)
                sys.stdout.flush()

                # 5. Upload mask DICOM to Orthanc
                step_start = time.time()
                self._log_detail("📤 Step 5/5: Uploading mask to Orthanc...")
                sys.stdout.flush()
                mask_series_id = self.upload_to_orthanc(mask_dicom_dir)
                if not mask_series_id:
                    raise Exception("Failed to get series ID from Orthanc after upload")
                step_elapsed = time.time() - step_start
                self._log_detail("   ✓ Completed in %.2fs", step_elapsed)
                sys.stdout.flush()

                total_elapsed = time.time() - start_time
                logger.info("="*70)
                logger.info("✅ Segmentation completed successfully!")
                logger.info("   Mask series ID: %s", mask_series_id)
                logger.info("   Total time: %.2fs (%.2f min)", total_elapsed, total_elapsed / 60)
                logger.info("="*70)
                sys.stdout.flush()

                results.append({
                    "status": "success",
                    "original_series_id": series_id,
                    "mask_series_id": mask_series_id,
                    "message": "Segmentation completed successfully"
                })
                
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                logger.error(f"✗ Error processing request for series {item.get('series_id', 'unknown')}:")
                logger.error(f"  Error: {str(e)}")
                logger.error(f"  Traceback:\n{error_traceback}")
                sys.stdout.flush()
                results.append({
                    "status": "failed",
                    "error": str(e),
                    "series_id": item.get('series_id', 'unknown')
                })
            finally:
                if dicom_dir and os.path.exists(dicom_dir):
                    shutil.rmtree(dicom_dir, ignore_errors=True)
        
        return results
    
    def download_series_from_orthanc(self, series_id: str, idx: int) -> str:
        """Download DICOM series from Orthanc"""
        output_dir = os.path.join(self.temp_dir, f"dicom_input_{idx}")
        os.makedirs(output_dir, exist_ok=True)

        archive_url = f'{ORTHANC_BASE_URL}/series/{series_id}/archive'
        archive_path = os.path.join(output_dir, f"{series_id}.zip")
        try:
            with requests.get(archive_url, auth=(os.environ.get("ORTHANC_USER_NAME"), os.environ.get("ORTHANC_PASSWORD")), stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(archive_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        except Exception as exc:
            raise Exception(f"Failed to download series archive: {series_id}") from exc

        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        except Exception as exc:
            raise Exception(f"Failed to extract series archive: {series_id}") from exc
        finally:
            if os.path.exists(archive_path):
                os.remove(archive_path)

        # Find actual DICOM root in case the archive has nested folders.
        for root, _dirs, files in os.walk(output_dir):
            if any(name.lower().endswith('.dcm') for name in files):
                return root

        return output_dir
    
    def dicom_to_nifti(self, dicom_dir: str, idx: int) -> str:
        """Convert DICOM series to NIfTI format"""
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Save as NIfTI (uncompressed to reduce CPU + I/O)
        nifti_path = os.path.join(self.temp_dir, f"input_{idx}.nii")
        sitk.WriteImage(image, nifti_path)
        
        return nifti_path
    
    def run_inference(self, nifti_path: str, idx: int) -> str:
        """Run nnU-Net TensorRT inference via persistent subprocess"""
        output_dir = os.path.join(self.temp_dir, f"output_{idx}")
        os.makedirs(output_dir, exist_ok=True)
        self._log_detail("   Sending inference request to persistent subprocess...")
        output_file = self._request_inference(nifti_path, output_dir)
        if not output_file:
            raise Exception("Inference subprocess did not return output path")

        if not os.path.exists(output_file):
            raise Exception(f"Prediction file not generated: {output_file}")
        self._log_detail("   Prediction file size: %s bytes", os.path.getsize(output_file))
        if os.getenv("MOSEC_DEBUG_PRED", "0") == "1":
            try:
                pred_image = sitk.ReadImage(output_file)
                pred_array = sitk.GetArrayFromImage(pred_image)
                unique_values, unique_counts = np.unique(pred_array, return_counts=True)
                logger.info(
                    "   Prediction voxels: shape=%s dtype=%s min=%s max=%s unique(sample up to 20)=%s",
                    pred_array.shape,
                    pred_array.dtype,
                    pred_array.min() if pred_array.size else None,
                    pred_array.max() if pred_array.size else None,
                    unique_values[:20].tolist(),
                )
                logger.info(
                    "   Prediction voxel counts (sample up to 20): %s",
                    list(zip(unique_values[:20].tolist(), unique_counts[:20].tolist()))
                )
                if self.verbose:
                    print(
                        "Prediction voxels:",
                        f"shape={pred_array.shape}",
                        f"dtype={pred_array.dtype}",
                        f"min={pred_array.min() if pred_array.size else None}",
                        f"max={pred_array.max() if pred_array.size else None}",
                        f"unique(sample up to 20)={unique_values[:20].tolist()}",
                        f"counts(sample up to 20)={list(zip(unique_values[:20].tolist(), unique_counts[:20].tolist()))}",
                        flush=True,
                    )
            except Exception as exc:
                logger.warning("   Failed to read prediction voxels for debug: %s", exc)

        self._log_detail("   TensorRT inference completed, output saved to: %s", output_file)
        sys.stdout.flush()

        return output_file
    
    def nifti_to_dicom(self, mask_nifti_path: str, reference_dicom_dir: str, idx: int) -> str:
        """Convert NIfTI mask to DICOM segmentation"""
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from datetime import datetime

        # Load mask
        self._log_detail("Loading mask from: %s", mask_nifti_path)
        mask_image = sitk.ReadImage(mask_nifti_path)
        mask_array = sitk.GetArrayFromImage(mask_image)
        # Reverse slice order to match DICOM orientation
        self._log_detail("Mask shape: %s", mask_array.shape)
        unique_values, unique_counts = np.unique(mask_array, return_counts=True)
        self._log_detail(
            "Mask unique values (sample up to 20): %s",
            unique_values[:20].tolist()
        )
        self._log_detail(
            "Mask voxel counts (sample up to 20): %s",
            list(zip(unique_values[:20].tolist(), unique_counts[:20].tolist()))
        )
        if self.verbose:
            print(
                f"Mask unique values (sample up to 20): {unique_values[:20].tolist()}",
                f"Mask voxel counts (sample up to 20): {list(zip(unique_values[:20].tolist(), unique_counts[:20].tolist()))}",
                flush=True
            )
        sys.stdout.flush()

        # Get reference DICOM files
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(reference_dicom_dir)
        self._log_detail("Found %s reference DICOM files", len(dicom_names))
        sys.stdout.flush()

        if len(dicom_names) == 0:
            raise Exception(f"No DICOM files found in reference directory: {reference_dicom_dir}")

        # Create output directory
        output_dir = os.path.join(self.temp_dir, f"mask_dicom_{idx}")
        os.makedirs(output_dir, exist_ok=True)

        # Generate a single SeriesInstanceUID for all slices in this mask series
        mask_series_uid = pydicom.uid.generate_uid()
        self._log_detail("Generated mask series UID: %s", mask_series_uid)
        sys.stdout.flush()

        # Build reference mapping (InstanceNumber order)
        ref_items = []
        for idx, ref_dicom_path in enumerate(dicom_names):
            try:
                ref_ds = pydicom.dcmread(ref_dicom_path, stop_before_pixels=True)
                inst_num = int(ref_ds.get('InstanceNumber', idx + 1))
            except Exception:
                inst_num = idx + 1
            ref_items.append((inst_num, idx, ref_dicom_path))

        ref_items.sort(key=lambda item: item[0])

        # Create DICOM files for each slice (InstanceNumber aligned)
        for order_index, (inst_num, src_index, ref_dicom_path) in enumerate(ref_items):
            ref_ds = pydicom.dcmread(ref_dicom_path)

            # Create new dataset based on reference
            ds = FileDataset(
                None, {},
                file_meta=ref_ds.file_meta,
                preamble=b"\0" * 128
            )

            # Copy important tags from reference
            ds.PatientName = ref_ds.get('PatientName', 'Anonymous')
            ds.PatientID = ref_ds.get('PatientID', '000000')
            ds.StudyInstanceUID = ref_ds.StudyInstanceUID
            ds.SeriesInstanceUID = mask_series_uid  # Use same UID for all slices in series
            ds.SOPInstanceUID = pydicom.uid.generate_uid()  # Unique per instance
            ds.SOPClassUID = ref_ds.SOPClassUID
            
            # Set modality and description
            ds.Modality = 'SEG'
            ds.SeriesDescription = 'AI Segmentation Mask'
            ds.SeriesNumber = str(int(ref_ds.get('SeriesNumber', '1')) + 1000)
            ds.InstanceNumber = str(inst_num)
            
            # Copy image metadata
            ds.Rows = ref_ds.Rows
            ds.Columns = ref_ds.Columns
            ds.ImagePositionPatient = ref_ds.ImagePositionPatient
            ds.ImageOrientationPatient = ref_ds.ImageOrientationPatient
            ds.PixelSpacing = ref_ds.PixelSpacing
            ds.SliceThickness = ref_ds.get('SliceThickness', '1.0')
            
            # Set pixel data
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            
            # Convert mask slice to uint16
            if src_index >= mask_array.shape[0]:
                raise Exception(
                    f"Mask slice index out of range: {src_index} (mask depth {mask_array.shape[0]})"
                )
            mask_slice = mask_array[src_index].astype(np.uint16)  # Scale for visibility
            ds.PixelData = mask_slice.tobytes()
            
            # Save DICOM file
            output_index = inst_num if inst_num is not None else (order_index + 1)
            output_path = os.path.join(output_dir, f"mask_{str(output_index).zfill(4)}.dcm")
            pydicom.dcmwrite(
                output_path,
                ds,
                enforce_file_format=True,
            )

        self._log_detail("Created %s DICOM mask files in %s", len(dicom_names), output_dir)
        sys.stdout.flush()

        return output_dir

    def upload_to_orthanc(self, dicom_dir: str) -> str:
        """Upload DICOM files to Orthanc and return series ID"""
        series_id = None

        # Check if directory exists and has files
        if not os.path.exists(dicom_dir):
            raise Exception(f"DICOM directory does not exist: {dicom_dir}")

        dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        self._log_detail("Found %s DICOM files to upload from %s", len(dicom_files), dicom_dir)
        sys.stdout.flush()

        if len(dicom_files) == 0:
            raise Exception(f"No DICOM files found in directory: {dicom_dir}")

        archive_path = os.path.join(dicom_dir, "mask_upload.zip")
        try:
            with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                for filename in sorted(dicom_files):
                    filepath = os.path.join(dicom_dir, filename)
                    zipf.write(filepath, arcname=filename)

            self._log_detail("Uploading %s DICOM files as ZIP to Orthanc...", len(dicom_files))
            sys.stdout.flush()

            with open(archive_path, 'rb') as f:
                response = requests.post(
                    f'{ORTHANC_BASE_URL}/instances',
                    data=f,
                    headers={'Content-Type': 'application/zip'},
                    auth=(os.environ.get("ORTHANC_USER_NAME"), os.environ.get("ORTHANC_PASSWORD")),
                    timeout=60
                )
            response.raise_for_status()

            result = response.json()
            instance_id = None
            if isinstance(result, dict):
                instance_id = result.get('ID')
                series_id = result.get('ParentSeries') or series_id
            elif isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    instance_id = first.get('ID')
                    series_id = first.get('ParentSeries') or series_id
                else:
                    instance_id = first

            self._log_detail("First instance uploaded with ID: %s", instance_id)
            sys.stdout.flush()

            if not instance_id:
                raise Exception("Failed to get instance ID from Orthanc ZIP upload response")

            if not series_id:
                instance_info = requests.get(
                    f'{ORTHANC_BASE_URL}/instances/{instance_id}',
                    auth=(os.environ.get("ORTHANC_USER_NAME"), os.environ.get("ORTHANC_PASSWORD")),
                    timeout=10
                ).json()
                series_id = instance_info.get('ParentSeries')
            self._log_detail("Got series ID: %s", series_id)
            sys.stdout.flush()

            if not series_id:
                raise Exception(f"Failed to get ParentSeries from instance {instance_id}")
        finally:
            if os.path.exists(archive_path):
                os.remove(archive_path)

        logger.info(
            "Successfully uploaded %s files to Orthanc with series ID: %s",
            len(dicom_files),
            series_id
        )
        sys.stdout.flush()
        return series_id
    
    def __del__(self):
        """Cleanup temporary directory"""
        self._stop_inference_process()
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Set nnUNet environment variables
    os.environ['nnUNet_results'] = os.path.abspath('HCC-TACE-Seg-nnU-Net-LiTS/nnUNet_results')
    os.environ['nnUNet_raw'] = os.path.abspath('HCC-TACE-Seg-nnU-Net-LiTS/nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = os.path.abspath('HCC-TACE-Seg-nnU-Net-LiTS/nnUNet_preprocessed')

    # Set Mosec configuration
    os.environ.setdefault('MOSEC_PORT', '8001')
    os.environ.setdefault('MOSEC_TIMEOUT', '7200000')  # 2 hour HTTP timeout (milliseconds!)

    server = Server()
    server.append_worker(
        DICOMSegmentationWorker,
        num=1,
        max_batch_size=2,
        max_wait_time=10,  # Dynamic batching wait time (milliseconds)
        timeout=7200,  # 2 hour timeout for forward processing (seconds)
        route="/ai/mosec/nnU-Net-Seg"
    )
    server.run()
