# MOSEC AI Model Server

nnU-Net v2와 TensorRT 기반 간암 세그멘테이션 추론 서버입니다. Orthanc의 DICOM series를 입력으로 받아 마스크 DICOM을 생성하고 다시 Orthanc에 업로드합니다.

## 구성

- **MOSEC**: 고성능 ML 모델 서빙
- **nnU-Net v2**: Medical Image Segmentation
- **TensorRT**: FP16 가속 추론
- **PyTorch**: 딥러닝 프레임워크
- **CUDA 12.6**: GPU 가속

## 사전 준비

- NVIDIA GPU (CUDA 12.6)
- TensorRT 10.4
- Orthanc 서버 접근 가능
- 아래 모델/엔진 파일 준비

## 필수 파일

- `HCC-TACE-Seg-nnU-Net-LiTS/` (학습 결과 포함)
- `nnunet_fp16.trt`
- `nnunet_fp16.onnx`
- `nnunet_fp16_sim.onnx`

## 실행 (Docker)

```bash
cd /home/syw/final_project/final_mosec_api_server
docker build -t liverguard-mosec .
docker run --gpus all -p 8001:8001 \
  -v $(pwd)/HCC-TACE-Seg-nnU-Net-LiTS:/app/HCC-TACE-Seg-nnU-Net-LiTS:ro \
  -v $(pwd)/nnunet_fp16.trt:/app/nnunet_fp16.trt:ro \
  -v $(pwd)/nnunet_fp16.onnx:/app/nnunet_fp16.onnx:ro \
  -v $(pwd)/nnunet_fp16_sim.onnx:/app/nnunet_fp16_sim.onnx:ro \
  liverguard-mosec
```

## 실행 (로컬)

```bash
cd /home/syw/final_project/final_mosec_api_server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app_fp16_tensorrt.py
```

## API

- **Endpoint**: `POST /ai/mosec/nnU-Net-Seg`
- **Default Port**: `8001`

요청 예시:

```bash
curl -X POST http://localhost:8001/ai/mosec/nnU-Net-Seg \
  -H "Content-Type: application/json" \
  -d '[{"series_id":"your-orthanc-series-id"}]'
```

응답 예시:

```json
[
  {
    "status": "success",
    "original_series_id": "your-orthanc-series-id",
    "mask_series_id": "generated-mask-series-id",
    "message": "Segmentation completed successfully"
  }
]
```
