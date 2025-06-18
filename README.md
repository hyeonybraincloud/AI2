# American Sign Language(ASL) 실시간 번역 AI 모델

# 1. Details
## 가. Motivation
https://www.udemy.com/course/computer-vision-game-development/

Udemy(위 링크 참조)에서 '웹캠으로 손가락 모션을 인지하고, 검지손가락 끝에 의사 캐릭터를 배치하여, 화면 상에 임의로 나타나는 바이러스 캐릭터와 충돌하면 점수를 얻는 게임' 실습을 하였다. 해당 실습에서, 웹캠 상에 비춰진 손가락의 움직임을 인지하는 것을 보면서, 수어(手語)를 실시간으로 번역해보는 모델을 만들어보면 어떨까 싶었다.

## 나. 실현 가능성
Google에 검색을 해보니, 이미 내 아이디어를 고도화한 버전이 존재했다. 다시 말해, 기술적으로 불가능한 아이디어가 아니라는 것을 반증하기도 한다. 그들과 비교했을 때, 데이터셋과 기술적인 측면에서 차이가 있겠으나, 간단한 버전이라도 구현해보고자 했다. 데이터셋은 Kaggle에서 발췌해왔으며, 링크는 다음과 같다.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data

데이터셋은, A, B, C, ..., Z, nothing, del, space, 총 29 개의 클래스로 구성되며, 클래스마다 3,000 장의 이미지가 있다. 각 이미지는 영어 알파벳에 대응되는 수어를 나타낸다.

# 2. Configuration
## 가. Fine_tuning_a_model_with_ASL_Dataset.py
### 1) Flow
**① 구글 드라이브 마운트 및 데이터 압축 해제**  
- Colab에서 Drive를 연결하고, `asl_alphabet_train` 폴더를 `/tmp`로 압축 해제(빠른 처리를 위해 로컬로 옮기는 것임)

**② 작업 디렉터리 및 하이퍼파라미터 설정**  
- 원본 데이터 경로(`SRC_ROOT`), 증강 결과 경로(`AUG_ROOT`), 오버레이 결과 경로(`DST_ROOT`) 지정

- 증강본 개수(`NUM_AUG`), 배치 크기, 학습률, 에포크 수 등 선언

**③ 증강 파이프라인 정의**  
- `RandomHorizontalFlip`(좌우 반전), `RandomRotation`(±15° 회전), `ColorJitter`(색상 변경) 등을 묶어 `aug_transform` 생성

**④ 데이터 폴더 구조 준비**  
- 클래스별 하위 폴더를 `AUG_ROOT`와 `DST_ROOT`에 미리 생성

**⑤ 데이터 증강 → 랜드마크 오버레이 → 저장**  
- 각 클래스 폴더 내 원본 이미지 순회  
   
- 원본 이미지에 대해 `NUM_AUG`개의 증강본 생성  

- 증강본마다 Mediapipe Hands로 손 랜드마크 검출  

- 검출된 경우 관절점·연결선 오버레이 후 `DST_ROOT`에 저장

**※ 왜 증강 후에 랜드마크 오버레이를 하였는가?**

증강 전에 랜드마크 오버레이가 되었다고 가정하겠다. 증강 단계에서 밝기·대비·채도 등이 무작위로 변경되면, 미리 그려둔 랜드마크의 색상 등이 예측 불가능하게 변형되어 모델이 일관된 시각적 패턴을 학습하기 어려워진다. 그래서 일관된 시각적 신호르 제공하기 위해, **원본 → 증강 → 오버레이** 순서로 했다.

**⑥ PyTorch 데이터셋 및 데이터로더 구성**  
- `DST_ROOT`의 오버레이된 이미지를 `ImageFolder`로 불러와 전체 데이터셋 생성

- 80:20 비율로 훈련/검증 세트 분할

- `Resize → ToTensor → Normalize` 변환 적용

- `DataLoader`를 통해 배치 처리, 셔플, 멀티프로세스 로딩 설정

**⑦ 모델 준비 및 학습 설정**  
- 사전학습된 `MobileNetV2`을 불러와 최종 분류기 레이어를 29개 클래스로 교체  

- 손실 함수(`CrossEntropyLoss`), 옵티마이저(`AdamW`), 학습률 스케줄러, AMP용 `GradScaler` 설정

**⑧ 훈련·검증 루프 → 모델 저장**  
- **훈련 모드**: forward → backward → optimizer/스케일러 업데이트

- **검증 모드**: no_grad 상태에서 손실·정확도 평가
  
- 손실, 정확도, 소요 시간 출력  

- 학습 종료 후 `asl_model_local.pth`를 Drive에 저장

## 나. realtime_demo.py
### 1) Flow
**① 라이브러리 임포트 및 옵션 파싱**  
- OpenCV, PyTorch, torchvision transforms, Mediapipe, PIL 등 필요한 모듈 불러오기  

- `--model` 인자로 학습된 `.pth` 파일 경로 입력 받기

**② 모델 로드 및 전처리 설정**  
- 지정된 경로에서 `torch.load`로 가중치 불러와 `model.load_state_dict()`  

- `model.eval()`로 평가 모드 전환  

- 입력 영상용 전처리 파이프라인(`Resize → ToTensor → Normalize`) 정의

**③ Mediapipe Hands 초기화**  
- `mp.solutions.hands.Hands` 객체 생성 (정적 이미지 모드 비활성화하여 실시간 처리)  

- 관절점·연결선을 그릴 `mp.solutions.drawing_utils` 준비

**④ 웹캠 스트림 열기**  
- `cv2.VideoCapture(0)`으로 기본 카메라 연결  

- 프레임 너비·높이 가져오기

**⑤ 프레임별 처리 루프**  
- **프레임 읽기**: `cap.read()` 로 BGR 이미지 획득  

- **색상 변환**: BGR → RGB  

- **손 검출**: `hands.process()` 로 랜드마크 위치 추출  

- **ROI(손 영역) 추출**: 랜드마크 좌표로 최소 경계 박스 계산, 해당 영역을 `224×224` 크기로 리사이즈  

- **모델 추론**: PIL 이미지로 변환 후 전처리 적용, 배치 차원 추가하여 모델에 입력 → 예측 결과(`argmax`) 추출, 최근 N프레임 예측을 다수결로 안정화  

**⑥ 결과 시각화**  
- 원본 프레임에 Mediapipe 관절점·연결선 오버레이  

- 예측된 알파벳 라벨을 `cv2.putText` 로 출력  

- `cv2.imshow` 로 실시간 창에 렌더링

**⑦ 종료 조건 처리**  
- 키 입력(`ESC`)을 감지하면 루프 탈출

**⑧ 자원 해제**  
- `cap.release()` 및 `cv2.destroyAllWindows()` 로 카메라·윈도우 자원 정리

### 2) Execution
![N_asl](https://github.com/user-attachments/assets/1d5753e7-576b-4b40-b542-c1c58b78b24c)
![D_asl](https://github.com/user-attachments/assets/a922c6c5-6a75-47c8-bac9-ef9346a13bbe)
![A_asl](https://github.com/user-attachments/assets/b961f4a2-c6ed-4043-b78f-3790594ab68f)
