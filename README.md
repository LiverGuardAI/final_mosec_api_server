# MOSEC AI Model Server

nnU-Net 기반 간암 세그멘테이션 추론 서버

## 기술 스택

- **MOSEC**: 고성능 ML 모델 서빙 프레임워크
- **nnU-Net v2**: Medical Image Segmentation
- **TensorRT**: FP16 가속 추론
- **PyTorch**: 딥러닝 프레임워크
- **CUDA 12.3**: GPU 가속

## Docker로 실행

### 전제조건
- NVIDIA GPU with CUDA 12.3 support
- Docker with NVIDIA Container Toolkit
- TensorRT 엔진 파일 (`nnunet_fp16.trt`)
- 학습된 모델 파일 (`HCC-TACE-Seg-nnU-Net-LiTS/`)

### 빌드 및 실행

```bash
# Docker Compose로 실행 (권장)
cd /home/syw/final_project
docker-compose up -d mosec

# 로그 확인
docker-compose logs -f mosec

# 개별 빌드 및 실행
cd final_mosec_api_server
docker build -t liverguard-mosec .
docker run --gpus all -p 8080:8080 \
  -v $(pwd)/HCC-TACE-Seg-nnU-Net-LiTS:/app/HCC-TACE-Seg-nnU-Net-LiTS:ro \
  -v $(pwd)/nnunet_fp16.trt:/app/nnunet_fp16.trt:ro \
  -e ORTHANC_URL=http://34.67.62.238/orthanc \
  liverguard-mosec
```

## API 엔드포인트

### Health Check
```bash
curl http://localhost:8080/health
```

### Inference
```bash
curl -X POST http://localhost:8080/inference \
  -H "Content-Type: application/json" \
  -d '{"series_id": "your-orthanc-series-id"}'
```

## 환경 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `ORTHANC_URL` | `http://34.67.62.238/orthanc` | Orthanc PACS 서버 URL |
| `CUDA_VISIBLE_DEVICES` | `0` | 사용할 GPU 번호 |

## 볼륨 마운트

- `/app/HCC-TACE-Seg-nnU-Net-LiTS` - nnU-Net 학습 모델
- `/app/nnunet_fp16.trt` - TensorRT FP16 엔진
- `/app/nnunet_fp16_sim.onnx` - 간단화된 ONNX 모델
- `/app/temp` - 임시 파일 처리 디렉토리

## 성능

- **추론 속도**: ~2-5초 (256³ 볼륨 기준, TensorRT FP16)
- **메모리 사용**: ~4-6GB GPU VRAM
- **동시 요청**: 1개 (GPU 메모리 제약)

## 문제 해결

### GPU 인식 안됨
```bash
# NVIDIA Container Toolkit 설치 확인
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

### TensorRT 엔진 로드 실패
- 엔진 파일이 올바른 위치에 있는지 확인
- CUDA 버전 호환성 확인 (12.3)
- 엔진을 현재 GPU에서 재빌드

### 메모리 부족
- `CUDA_VISIBLE_DEVICES` 환경 변수로 GPU 선택
- 다른 GPU 프로세스 종료
- 배치 크기 조정