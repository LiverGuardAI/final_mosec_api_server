import os
import sys
import tempfile
import shutil
import logging
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
    
    def forward_setup(self):
        """Initialize worker (predictor will be initialized in subprocess)"""
        logger.info("="*70)
        logger.info("🔧 Initializing DICOM Segmentation Worker")
        logger.info("="*70)
        sys.stdout.flush()

        # Check CUDA availability
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"Current GPU: {torch.cuda.current_device()}")
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        logger.info(f"Selected device: {self.device}")
        sys.stdout.flush()

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="dicom_seg_")
        logger.info(f"Temporary directory: {self.temp_dir}")

        # Note: Predictor will be initialized in subprocess to avoid daemon process issues
        logger.info("Note: nnU-Net predictor will be initialized in subprocess for each inference")
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
                logger.info(f"🚀 Starting segmentation for series: {series_id}")
                logger.info("="*70)
                sys.stdout.flush()

                # 1. Download DICOM series from Orthanc
                step_start = time.time()
                logger.info("📥 Step 1/5: Downloading DICOM from Orthanc...")
                sys.stdout.flush()
                dicom_dir = self.download_series_from_orthanc(series_id, idx)
                if not dicom_dir or not os.path.exists(dicom_dir):
                    raise Exception(f"Failed to download DICOM series: {series_id}")
                step_elapsed = time.time() - step_start
                logger.info(f"   ✓ Completed in {step_elapsed:.2f}s")
                sys.stdout.flush()

                # 2. Convert DICOM to NIfTI
                step_start = time.time()
                logger.info("🔄 Step 2/5: Converting DICOM to NIfTI...")
                sys.stdout.flush()
                nifti_path = self.dicom_to_nifti(dicom_dir, idx)
                if not nifti_path or not os.path.exists(nifti_path):
                    raise Exception("Failed to convert DICOM to NIfTI")
                step_elapsed = time.time() - step_start
                logger.info(f"   ✓ Completed in {step_elapsed:.2f}s")
                sys.stdout.flush()

                # 3. Run nnU-Net inference
                step_start = time.time()
                logger.info("🧠 Step 3/5: Running nnU-Net inference (this may take a while)...")
                sys.stdout.flush()
                mask_nifti_path = self.run_inference(nifti_path, idx)
                if not mask_nifti_path or not os.path.exists(mask_nifti_path):
                    raise Exception("Failed to run nnU-Net inference")
                step_elapsed = time.time() - step_start
                logger.info(f"   ✓ Completed in {step_elapsed:.2f}s")
                sys.stdout.flush()

                # 4. Convert mask to DICOM
                step_start = time.time()
                logger.info("🔄 Step 4/5: Converting mask to DICOM...")
                sys.stdout.flush()
                mask_dicom_dir = self.nifti_to_dicom(mask_nifti_path, dicom_dir, idx)
                if not mask_dicom_dir or not os.path.exists(mask_dicom_dir):
                    raise Exception("Failed to convert mask to DICOM")
                step_elapsed = time.time() - step_start
                logger.info(f"   ✓ Completed in {step_elapsed:.2f}s")
                sys.stdout.flush()

                # 5. Upload mask DICOM to Orthanc
                step_start = time.time()
                logger.info("📤 Step 5/5: Uploading mask to Orthanc...")
                sys.stdout.flush()
                mask_series_id = self.upload_to_orthanc(mask_dicom_dir)
                if not mask_series_id:
                    raise Exception("Failed to get series ID from Orthanc after upload")
                step_elapsed = time.time() - step_start
                logger.info(f"   ✓ Completed in {step_elapsed:.2f}s")
                sys.stdout.flush()

                total_elapsed = time.time() - start_time
                logger.info("="*70)
                logger.info("✅ Segmentation completed successfully!")
                logger.info(f"   Mask series ID: {mask_series_id}")
                logger.info(f"   Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
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
            with requests.get(archive_url, stream=True, timeout=60) as response:
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
        
        # Save as NIfTI
        nifti_path = os.path.join(self.temp_dir, f"input_{idx}.nii.gz")
        sitk.WriteImage(image, nifti_path)
        
        return nifti_path
    
    def run_inference(self, nifti_path: str, idx: int) -> str:
        """Run nnU-Net inference using subprocess to avoid daemon limitations"""
        import subprocess

        output_dir = os.path.join(self.temp_dir, f"output_{idx}")
        os.makedirs(output_dir, exist_ok=True)

        # Get the path to the standalone inference script
        script_path = os.path.join(os.path.dirname(__file__), 'run_nnunet_inference_fp16.py')

        # Prepare arguments for subprocess
        folds_str = ','.join(map(str, self.folds))
        device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'

        logger.info(f"   Running subprocess inference script...")
        logger.info(f"   Script: {script_path}")
        logger.info(f"   Model: {self.model_folder}")
        logger.info(f"   Device: {device_str}")
        logger.info(f"   Folds: {folds_str}")
        sys.stdout.flush()

        # Run inference as separate process (bypasses daemon process limitation)
        # Real-time output enabled by not capturing stdout/stderr
        try:
            logger.info(f"   Starting subprocess (output will be shown in real-time)...")
            sys.stdout.flush()

            result = subprocess.run([
                sys.executable,  # Python interpreter
                script_path,
                self.model_folder,
                nifti_path,
                output_dir,
                folds_str,
                self.checkpoint_name,
                device_str
            ],
            timeout=3600  # 1 hour timeout
            )

            if result.returncode != 0:
                error_msg = f"Inference subprocess failed with return code {result.returncode}"
                logger.error(f"   {error_msg}")
                sys.stdout.flush()
                raise Exception(error_msg)

        except subprocess.TimeoutExpired:
            raise Exception("Inference subprocess timed out after 1 hour")
        except Exception as e:
            raise Exception(f"Failed to run inference subprocess: {str(e)}")

        # Get output file path
        filename = os.path.basename(nifti_path)
        output_file = os.path.join(output_dir, filename)

        if not os.path.exists(output_file):
            raise Exception(f"Prediction file not generated: {output_file}")

        logger.info(f"   Inference completed, output saved to: {output_file}")
        sys.stdout.flush()

        return output_file
    
    def nifti_to_dicom(self, mask_nifti_path: str, reference_dicom_dir: str, idx: int) -> str:
        """Convert NIfTI mask to DICOM segmentation"""
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from datetime import datetime

        # Load mask
        logger.info(f"Loading mask from: {mask_nifti_path}")
        mask_image = sitk.ReadImage(mask_nifti_path)
        mask_array = sitk.GetArrayFromImage(mask_image)
        # Reverse slice order to match DICOM orientation
        mask_array = mask_array[::-1]
        logger.info(f"Mask shape: {mask_array.shape} (reversed)")
        sys.stdout.flush()

        # Get reference DICOM files
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(reference_dicom_dir)
        logger.info(f"Found {len(dicom_names)} reference DICOM files")
        sys.stdout.flush()

        if len(dicom_names) == 0:
            raise Exception(f"No DICOM files found in reference directory: {reference_dicom_dir}")

        # Create output directory
        output_dir = os.path.join(self.temp_dir, f"mask_dicom_{idx}")
        os.makedirs(output_dir, exist_ok=True)

        # Generate a single SeriesInstanceUID for all slices in this mask series
        mask_series_uid = pydicom.uid.generate_uid()
        logger.info(f"Generated mask series UID: {mask_series_uid}")
        sys.stdout.flush()

        # Create DICOM files for each slice
        for i, ref_dicom_path in enumerate(dicom_names):
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
            ds.InstanceNumber = str(i + 1)
            
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
            mask_slice = mask_array[i].astype(np.uint16) * 1000  # Scale for visibility
            ds.PixelData = mask_slice.tobytes()
            
            # Save DICOM file
            output_path = os.path.join(output_dir, f"mask_{str(i+1).zfill(4)}.dcm")
            ds.save_as(output_path, write_like_original=False)

        logger.info(f"Created {len(dicom_names)} DICOM mask files in {output_dir}")
        sys.stdout.flush()

        return output_dir

    def upload_to_orthanc(self, dicom_dir: str) -> str:
        """Upload DICOM files to Orthanc and return series ID"""
        series_id = None

        # Check if directory exists and has files
        if not os.path.exists(dicom_dir):
            raise Exception(f"DICOM directory does not exist: {dicom_dir}")

        dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        logger.info(f"Found {len(dicom_files)} DICOM files to upload from {dicom_dir}")
        sys.stdout.flush()

        if len(dicom_files) == 0:
            raise Exception(f"No DICOM files found in directory: {dicom_dir}")

        archive_path = os.path.join(dicom_dir, "mask_upload.zip")
        try:
            with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                for filename in sorted(dicom_files):
                    filepath = os.path.join(dicom_dir, filename)
                    zipf.write(filepath, arcname=filename)

            logger.info(f"Uploading {len(dicom_files)} DICOM files as ZIP to Orthanc...")
            sys.stdout.flush()

            with open(archive_path, 'rb') as f:
                response = requests.post(
                    f'{ORTHANC_BASE_URL}/instances',
                    data=f,
                    headers={'Content-Type': 'application/zip'},
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

            logger.info(f"First instance uploaded with ID: {instance_id}")
            sys.stdout.flush()

            if not instance_id:
                raise Exception("Failed to get instance ID from Orthanc ZIP upload response")

            if not series_id:
                instance_info = requests.get(
                    f'{ORTHANC_BASE_URL}/instances/{instance_id}',
                    timeout=10
                ).json()
                series_id = instance_info.get('ParentSeries')
            logger.info(f"Got series ID: {series_id}")
            sys.stdout.flush()

            if not series_id:
                raise Exception(f"Failed to get ParentSeries from instance {instance_id}")
        finally:
            if os.path.exists(archive_path):
                os.remove(archive_path)

        logger.info(f"Successfully uploaded {len(dicom_files)} files to Orthanc with series ID: {series_id}")
        sys.stdout.flush()
        return series_id
    
    def __del__(self):
        """Cleanup temporary directory"""
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
