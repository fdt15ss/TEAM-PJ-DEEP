# FastAPI + Streamlit 실행 가이드

## 1. 프로젝트 폴더 구조

    project_root/
    │
    ├─ backend/
    │  └─ fastapi_main.py        # FastAPI 서버 코드
    │
    ├─ streamlit/
    │  ├─ model_efficiNetB4.py
    │  ├─ model_tester_convnext.py
    │  └─ streamlit_main.py      # Streamlit UI 실행 파일
    │
    ├─ best_model/ # 구글 드라이브에서 5개 파일 전부 다운로드할 것
    │  ├─ best_model_efficiB4.pth
    │  ├─ best_model_ConvNext.pth
    │  ├─ best_model_resnet50.pth
    │  ├─ multiLabel_best_model.pt
    │  └─ document_best_model.pt
    │
    ├─ data/
    │  └─ multilabel_classes.json
    │
    └─ requirements.txt

------------------------------------------------------------------------

# 2. 가상환경 실행

### Windows

    .venv\Scripts\activate

### Linux / Mac

    source .venv/bin/activate

------------------------------------------------------------------------

# 3. 필요한 라이브러리 설치

## 방법 1 --- pip 사용

    pip install fastapi uvicorn streamlit torch torchvision transformers pillow pytorch-grad-cam

또는

    pip install -r requirements.txt

------------------------------------------------------------------------

## 방법 2 --- uv 사용 (권장 ⚡)

### 패키지 설치

    uv add fastapi
    uv add uvicorn
    uv add streamlit
    uv add torch
    uv add torchvision
    uv add transformers
    uv add pillow
    uv add pytorch-grad-cam

### requirements.txt 기반 설치

    uv pip install -r requirements.txt

### 현재 환경 기반 requirements.txt 생성

    uv pip freeze > requirements.txt

------------------------------------------------------------------------

# 4. FastAPI 서버 실행

프로젝트 **루트 폴더에서 실행합니다.**

    uvicorn backend.fastapi_main:app --host 0.0.0.0 --port 8000

개발 환경에서는 자동 리로드 옵션을 사용할 수 있습니다.

    uvicorn backend.fastapi_main:app --host 0.0.0.0 --port 8000 --reload

------------------------------------------------------------------------

# 5. Streamlit UI 실행

Streamlit 테스트 UI는 다음 명령어로 실행합니다.

    uv run streamlit run streamlit/streamlit_main.py

실행 후 브라우저에서 아래 주소로 접속할 수 있습니다.

    http://localhost:8501

------------------------------------------------------------------------

# 6. API 문서 확인

FastAPI 서버 실행 후 아래 주소에서 API 테스트가 가능합니다.

### Swagger UI

    http://localhost:8000/docs

------------------------------------------------------------------------

# 7. 실행 로그 예시

    ========== 서버 시작: 모델 로드 ==============
    필드 분석 모델 로드 중...
      - EfficientNet-B4... ✓
      - ConvNeXt... ✓
      - ResNet50... ✓
    텍스트 분류 모델(ROBERTA) 로드 중... ✓
    문서 분류 모델(ConvNeXt) 로드 중... ✓
    사용 디바이스: cuda
    ========== 모델 로드 완료 ==============
