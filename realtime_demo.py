# ───────────── realtime_demo.py ─────────────
import argparse
import cv2
import torch
import torchvision.transforms as T
import mediapipe as mp
from collections import deque
from PIL import Image

# ---------- 옵션 ----------
parser = argparse.ArgumentParser(description="ASL 알파벳 실시간 인식 데모")
parser.add_argument(
    "--model",
    default="asl_model_local.pth",
    help="학습된 .pth 파일 경로 (예: asl_model_local.pth)"
)
args = parser.parse_args()

# ---------- 클래스 리스트 ----------
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DEL", "NOTHING"]
num_classes = len(labels)

# ---------- 장치 설정 및 모델 로드 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torchvision hub로 mobilenet_v2 로드 (weights=None: 사용자 지정 가중치 사용)
model = torch.hub.load("pytorch/vision", "mobilenet_v2", weights=None)
# 마지막 분류기 레이어를 알파벳 클래스 수에 맞게 교체
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

# GPU 사용 시에만 PyTorch 2.0 컴파일 활성화
if device.type == 'cuda':
    model = torch.compile(model)

# 사용자 지정 .pth 가중치 로드
raw_state = torch.load(args.model, map_location=device)
state_dict = {}
for k, v in raw_state.items():
    new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
    state_dict[new_key] = v
model.load_state_dict(state_dict)
model.to(device).eval()

# ---------- 전처리 파이프라인 ----------
prep = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])

# ---------- Mediapipe 설정 ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# 예측 스무딩을 위한 히스토리
history = deque(maxlen=5)

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 찾을 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # 결과 초기화
    label = ""
    conf_text = ""

    # Mediapipe로 손 검출
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # 랜드마크 그리기
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

    # 손 영역 예측
    if results.multi_hand_landmarks:
        pts = [(int(l.x * w), int(l.y * h)) for l in results.multi_hand_landmarks[0].landmark]
        x1 = max(min(p[0] for p in pts) - 20, 0)
        y1 = max(min(p[1] for p in pts) - 20, 0)
        x2 = min(max(p[0] for p in pts) + 20, w)
        y2 = min(max(p[1] for p in pts) + 20, h)
        roi = frame[y1:y2, x1:x2]

        if roi.size != 0:
            # ROI 리사이즈 및 컬러 변환
            img = cv2.cvtColor(cv2.resize(roi, (224, 224)), cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            # 모델 추론 및 확률 계산
            with torch.no_grad():
                x = prep(img).unsqueeze(0).to(device)
                probs = torch.softmax(model(x), dim=1)[0]
                idx = int(probs.argmax())
                conf = float(probs[idx])

            # 신뢰도 기준 스무딩
            if conf > 0.7:
                history.append(idx)
            if history:
                idx = max(set(history), key=history.count)
                label = labels[idx]
                # 최종 확률은 현재 frame의 확률로 표시
                conf_text = f"{label} ({probs[idx]*100:.1f}%)"

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 인식된 라벨 + 확률 화면 출력
    if conf_text:
        cv2.putText(
            frame, conf_text, (40, 120), cv2.FONT_HERSHEY_SIMPLEX,
            2.5, (0, 0, 255), 5, cv2.LINE_AA
        )

    cv2.imshow("ASL Realtime", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
