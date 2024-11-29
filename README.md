# 🔍 VisionDefectToolkit

VisionDefectToolkit은 머신비전 기반의 결함 검출 및 분석을 위한 종합 도구 모음입니다. PyQt6 기반의 직관적인 UI를 통해 이미지 전처리부터 결함 검출까지 원스톱으로 처리할 수 있습니다.

## 📁 프로젝트 구조

```
VisionDefectToolkit/
├── FilterApplicationTool/
│   ├── filters/
│   │   ├── base_filter.py
│   │   ├── edge_filters.py
│   │   └── frequency_filters.py
│   ├── model.py
│   ├── view.py
│   └── controller.py
├── ImageViewerTool/
│   ├── ImageViewer_preprocess_v0.1.py
│   └── ImageViewer_simple.py
└── requirements.txt
```

## 🛠 주요 기능

### 1. FilterApplicationTool
고급 이미지 필터링 도구로, 다양한 필터를 실시간으로 적용하고 비교할 수 있습니다.

#### 핵심 기능:
- **다중 필터 미리보기**: 10가지 필터를 동시에 비교
- **실시간 강도 조절**: 슬라이더를 통한 필터 강도 실시간 조절
- **필터 블렌딩**: 원본과 필터링된 이미지의 자연스러운 블렌딩

```python
# 필터 강도 조절 예시
def blend_with_original(self, original, filtered, intensity):
    if len(filtered.shape) == 2:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(original, 1 - intensity, filtered, intensity, 0)
```

### 2. ImageViewerTool
YOLO 기반 결함 검출 및 이미지 전처리 도구입니다.

#### 핵심 기능:
- **실시간 결함 검출**: YOLO 모델을 통한 실시간 결함 감지
- **다양한 전처리 옵션**: CLAHE, Gaussian Blur, Canny Edge 등
- **Ground Truth 비교**: 예측 결과와 실제 라벨 비교 분석

## 🎯 사용 예시

### FilterApplicationTool
![Filter Application Example](filter_application_example.png)
*다양한 필터를 적용한 결함 이미지 분석 예시*

필터 종류:
- Original
- Bandpass Filter
- Gabor Filter
- Laplacian
- Sobel (X/Y)
- Scharr (X/Y)
- Prewitt
- Canny Edge

## 💻 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/username/VisionDefectToolkit.git
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. 실행
```bash
python FilterApplicationTool/main.py
# 또는
python ImageViewerTool/ImageViewer_preprocess_v0.1.py
```
