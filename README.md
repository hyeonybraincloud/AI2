# American Sign Language(ASL) 실시간 번역 AI 모델

# 1. Details
## 가. Inspiration
https://www.udemy.com/course/computer-vision-game-development/

Udemy(위 링크 참조)에서 '웹캠으로 손가락 모션을 인지하고, 검지손가락 끝에 의사 캐릭터를 배치하여, 화면 상에 임의로 나타나는 바이러스 캐릭터와 충돌하면 점수를 얻는 게임' 실습을 하였다. 해당 실습에서, 웹캠 상에 비춰진 손가락의 움직임을 인지하는 것을 보면서, 수어(手語)를 실시간으로 번역해보는 모델을 만들어보면 어떨까 싶었다.

## 나. 실현 가능성
Google에 검색을 해보니, 이미 내 아이디어를 고도화한 버전이 존재했다. 다시 말해, 기술적으로 불가능한 아이디어가 아니라는 것을 반증하기도 한다. 그들과 비교했을 때, 데이터셋과 기술적인 측면에서 차이가 있겠으나, 간단한 버전이라도 구현해보고자 했다. 데이터셋은 Kaggle에서 발췌해왔으며, 링크는 다음과 같다.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data

데이터셋은, A, B, C, ..., Z, nothing, del, space, 총 29 개의 클래스로 구성되며, 클래스마다 3,000 장의 이미지가 있다. 각 이미지는 영어 알파벳에 대응되는 수어를 나타낸다.

# 2. Configuration
## 가. Fine_tuning_a_model_with_ASL_Dataset.ipynb
### 1) Flow
**① 라이브러리 및 환경 설정**
- `torch`, `torchvision`, `matplotlib`, `PIL` 등 필요한 패키지 `import`

- `CUDA` 사용 가능 여부 체크(A100)

데이터셋 로딩 및 전처리

ASL 이미지 데이터셋 로딩 (경로: ./asl_dataset)

transforms.Compose를 이용한 이미지 전처리 (리사이즈, 센터크롭, 텐서화 등)

훈련 및 검증 데이터셋 생성 (ImageFolder 활용)

DataLoader를 통해 배치 단위로 데이터 제공

사전 학습된 모델 로딩 및 수정

resnet18 모델 불러오기 (pretrained=True)

마지막 fc layer를 ASL 클래스 수(29개)로 변경

모델을 GPU로 전송

손실 함수 및 최적화 도구 설정

손실 함수: CrossEntropyLoss

옵티마이저: SGD (with momentum)

모델 학습

에폭 단위 반복

각 에폭마다 학습 및 검증 수행

정확도와 손실 출력

가장 좋은 모델을 저장 (best_model_wts)

모델 저장

학습 완료 후 최적의 모델을 파일로 저장 (asl_resnet18.pth)


asl_model_local.pth
## 나. realtime_demo.py
