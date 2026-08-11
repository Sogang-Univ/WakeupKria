#!/usr/bin/env python3
"""02 · 웨이크 워드 → 제스처 → 모델 교체 + cam1 리모컨 (스크립트 판)

`02_wake_gesture_switch.ipynb` 를 옮긴 것이다. 노트북이 아직 완성 전이라 셀 경계를
`# ==== 셀 N ====` 로 남겨 두었다 — 노트북이 바뀌면 해당 구간만 통째로 덮어쓰면 된다.
다시 쓴 곳은 셀 12(표시 루프) 하나뿐이다. 위젯 대신 cv2.imshow 를 쓴다.
이유·근거·재동기화 절차는 NOTEBOOK_TO_PY.md 를 볼 것.

    python3 02_wake_gesture_switch.py               # ssh -X 로 창이 뜬다
    python3 02_wake_gesture_switch.py --no-display  # 헤드리스. stdout 만
    python3 02_wake_gesture_switch.py --self-test   # 상태 기계만. 하드웨어 불필요

창에서: q=종료, w=수동 wake, 숫자키=제스처 주입 (마이크·손 없이 시연 경로를 탄다)
"""

import sys
import argparse

# --self-test 는 상태 기계만 본다. pynq_dpu·sounddevice 는 보드에만 있으므로,
# dev PC 에서도 점검이 돌도록 무거운 import 를 건너뛴다. 이 파일에만 있는 장치다.
SELF_TEST = "--self-test" in sys.argv

# ============================================================ 셀 2 · 설정
import os
import re
import json
import time
import queue
import threading

import cv2
import numpy as np

# 제스처 수학(디코딩·시퀀스·게이트)은 전부 gesture_live.py 에 있다. 여기서 다시 쓰지 않는다.
if not SELF_TEST:
    from gesture_live import (
        Runner, GestureClassifier, SequenceBuffer,
        build_sequence, is_active, decode, draw,
        SHAPE_CLASSES, GESTURE_CLASSES,
    )
else:
    SHAPE_CLASSES = ["open_palm", "fist", "thumb_up", "thumb_down"]   # 점검용 사본

# ---- 카메라 ----
CAM_A_DEV = "/dev/cam0"      # 사용자 — 제스처를 본다
CAM_B_DEV = "/dev/cam1"      # 대상   — 두 번째 모델이 보고, 제스처로 화면을 조종한다
WIDTH, HEIGHT = 640, 480     # 00번 검증 스펙이자 제스처 학습 때의 프레임 비율
FOURCC = "MJPG"              # 고정. YUYV 는 이 카메라에서 30fps 가 안 나온다
TARGET_FPS = 30

DISPLAY_SCALE = 0.6
DISPLAY_FPS = 10
DISP_WH = (int(WIDTH * DISPLAY_SCALE), int(HEIGHT * DISPLAY_SCALE))   # 크롭해도 창이 안 튄다

# ---- 마이크 (01_wakeword 3장 값 그대로) ----
# None = 시스템 기본 입력. results/01_wakeword.json 이 기록한 장치 이름이 그대로 "default"다.
# 인덱스(18)를 박지 않는 이유: PortAudio 인덱스는 Pa_Initialize() 때 만들어지는 ALSA PCM
# 목록의 위치일 뿐이라 세션마다 밀린다. PulseAudio 에 못 붙는 커널에서는 pulse 항목이
# 통째로 빠지면서 뒤가 한 칸씩 당겨져 18 이 사라지고 "Error querying device 18" 로 죽는다.
MIC_INDEX = None
SAMPLE_RATE, CHANNELS, DTYPE, CHUNK = 16000, 1, "int16", 1280
WAKEWORD_MODEL = "./hey_kria.onnx"   # 커스텀. 예측 딕셔너리 키는 파일 stem "hey_kria"
# 임계값 단일 출처는 학습 산출물 threshold.json 이다 (0.85 = recall 0.905 · 오검출 0.9회/h).
# 현장에서 안 깨어나면 0.05 씩 내린다 — 그 지점의 오검출률은 threshold.json 의 sweep 에 있다.
WAKE_THRESHOLD = (json.load(open("threshold.json"))["threshold"]
                  if os.path.exists("threshold.json") else 0.5)
WAKE_REFRACTORY = 2.0

# ---- DPU ----
BIT_PATH = "dpu.bit"
GESTURE_XMODEL = "./gesture_stage1_kv260.xmodel"
GRU_PATH = "./gesture_stage2_gru.tflite"

# ---- 두 번째 모델 (cam1) ----
# 쓰레기 분류 YOLOv8 9클래스. 제스처 xmodel 과 같은 래퍼(YOLOv8HeadIncludedWrapper)로
# export 한 같은 구조라 클래스 수만 다르다. dpu_fingerprint 도 같아서
# (DPUCZDX8G_ISA1_B4096) 같은 비트스트림 위에서 교체된다.
TASK_XMODEL = "./task_ai.xmodel"
# 순서 = tempModel/data.yaml 의 인덱스. cv2.putText 에 한글 폰트가 없어 로마자로 적는다
# (유리병류 · 도기류 · 형광등 · 전자제품 · 의류 · 비닐류 · 캔류 · 플라스틱류 · 종이류)
TASK_CLASSES = ["glass", "ceramic", "fluorescent", "electronics", "clothing",
                "vinyl", "can", "plastic", "paper"]
TASK_CONF = 0.35             # 시연하며 맞춘다. 안 잡히면 0.25, 오검출이 많으면 0.5
TASK_MAX_DETS = 10           # 제스처는 손 하나면 되지만 쓰레기는 한 화면에 여러 개다

# ---- 제스처 (gesture_live.py 기본값) ----
CONF = 0.15                  # 학습 시퀀스 추출값. 바꾸면 특징 분포가 어긋난다
WINDOW_S = 1.0
GRU_EVERY = 5
GESTURE_THRES = 0.70
GESTURE_COOLDOWN = 1.0       # 1회 발화 후 이 시간 동안 예측 정지 + 시퀀스 버퍼 폐기.
                             # 복귀동작이 다음 창에 섞여 재발화하는 것을 막는다

# ---- cam1 디지털 PTZ ----
# 시연하면서 맞추는 값이다. 한 번에 얼마나 움직이고 얼마나 당길지는 손에 붙어야 정해진다.
PAN_STEP = 0.15              # 화면 폭 대비 1회 이동량 (줌 배율로 나눠 쓴다)
ZOOM_STEP = 1.25             # 1회 줌 배율
ZOOM_MIN, ZOOM_MAX = 1.0, 4.0
PAN_INVERT = False           # "left 제스처 = 화면이 왼쪽으로" 가 반대로 느껴지면 True


def dev_index(path):
    real = os.path.realpath(path)
    m = re.search(r"(\d+)$", real)
    if not m:
        raise ValueError(f"video 인덱스를 찾을 수 없다: {path} -> {real}")
    return int(m.group(1))


try:
    CAM_A_ID, CAM_B_ID = dev_index(CAM_A_DEV), dev_index(CAM_B_DEV)
except (ValueError, OSError) as e:
    print(f"[WARN] 심볼릭 링크 해석 실패 ({e}). 인덱스를 직접 지정한다.")
    CAM_A_ID, CAM_B_ID = 0, 2


def print_config():
    print(f"CAM_A = {CAM_A_DEV} -> /dev/video{CAM_A_ID}  (제스처)")
    print(f"CAM_B = {CAM_B_DEV} -> /dev/video{CAM_B_ID}  (두 번째 모델 · PTZ 대상)")
    print(f"캡처 {WIDTH}x{HEIGHT} {FOURCC} @{TARGET_FPS}")
    print(f"제스처 xmodel : {GESTURE_XMODEL}  conf={CONF}  클래스 {SHAPE_CLASSES}")
    print(f"작업   xmodel : {TASK_XMODEL}  conf={TASK_CONF}  max_det={TASK_MAX_DETS}")
    print(f"               클래스 {TASK_CLASSES}")
    if not os.path.exists(TASK_XMODEL):
        print(f"[WARN] 작업 xmodel 이 없다: {TASK_XMODEL}")
    print(f"PTZ           : pan {PAN_STEP} · zoom ×{ZOOM_STEP} ({ZOOM_MIN}~{ZOOM_MAX})")
    print(f"웨이크 워드   : {WAKEWORD_MODEL}  threshold={WAKE_THRESHOLD}")


# ============================================================ 셀 4 · 캡처 스레드
# 카메라마다 스레드를 하나씩 둔다. 두 소비자(화면·DPU 워커)가 각자 snapshot() 으로
# 최신 프레임 한 장만 가져가므로 큐가 없고 지연이 쌓이지 않는다.

def set_v4l2_ctrl(dev_id, name, value):
    import subprocess
    return subprocess.run(
        ["v4l2-ctl", "-d", f"/dev/video{dev_id}", "-c", f"{name}={value}"],
        capture_output=True, text=True,
    ).returncode == 0


class CamStream(threading.Thread):
    """카메라 한 대에서 계속 읽으며 최신 프레임과 실효 fps를 유지한다."""

    def __init__(self, dev_id, label):
        super().__init__(daemon=True)
        self.dev_id = dev_id
        self.label = label
        self.frame = None
        self.fps = 0.0
        self.reads = 0
        self.fails = 0
        self._lock = threading.Lock()
        self._running = False
        self.cap = None

    def open(self):
        cap = cv2.VideoCapture(self.dev_id, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False

        # FOURCC를 해상도보다 먼저 설정해야 한다
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 저조도에서 카메라가 프레임 주기를 2배로 늘리는 것을 막는다
        set_v4l2_ctrl(self.dev_id, "exposure_dynamic_framerate", 0)

        self.cap = cap
        return True

    def run(self):
        self._running = True
        n, t0 = 0, time.monotonic()
        while self._running:
            ok, f = self.cap.read()
            if ok and f is not None:
                with self._lock:
                    self.frame = f
                self.reads += 1
                n += 1
                if n >= 15:
                    now = time.monotonic()
                    self.fps = n / (now - t0)
                    n, t0 = 0, now
            else:
                self.fails += 1
                time.sleep(0.005)

    def snapshot(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self._running = False
        self.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ============================================================ 셀 6 · 상태 기계
# 파이프라인 전체의 규칙이 apply_event() 하나에 들어 있다. 하드웨어 없이 검증된다.

INIT_STATE = {"on": False, "dpu": "gesture", "cx": 0.5, "cy": 0.5, "zoom": 1.0}
PAN = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
DPU_BY_EVENT = {"open": "task"}                      # dpu: gesture | task


def _clamp(st):
    half = 0.5 / st["zoom"]                        # 뷰포트가 프레임 밖으로 못 나간다
    st["cx"] = min(1 - half, max(half, st["cx"]))
    st["cy"] = min(1 - half, max(half, st["cy"]))
    return st


def apply_event(st, ev):
    """상태 딕셔너리를 이벤트 하나로 갱신해 **새** 딕셔너리를 돌려준다. 순수 함수.

    낮은 배율에서는 한 스텝이 남은 여백보다 커서 한 번에 가장자리까지 간다.
    PAN_STEP 을 배율로 나누는 이유가 그것이고, 그래도 모자라면 PAN_STEP 을 줄인다.
    """
    st = dict(st)
    if ev == "wake":
        st["on"], st["dpu"] = True, "gesture"
        return st
    if not st["on"]:
        return st                                  # 꺼져 있으면 wake 말고는 아무것도 안 먹는다
    if ev == "close":                              # 끄는 것은 close 뿐이다. bye 는 안 묶는다 —
        st["on"] = False                           # 정지한 손이 GRU 에서 bye 로 읽히기 때문이다
    elif ev in DPU_BY_EVENT:
        st["dpu"] = DPU_BY_EVENT[ev]
    elif ev in ("thumbs up", "thumbs down"):
        z = st["zoom"] * (ZOOM_STEP if ev == "thumbs up" else 1 / ZOOM_STEP)
        st["zoom"] = min(ZOOM_MAX, max(ZOOM_MIN, z))
    elif ev in PAN:
        dx, dy = PAN[ev]
        s = -PAN_STEP / st["zoom"] if PAN_INVERT else PAN_STEP / st["zoom"]
        st["cx"] += dx * s
        st["cy"] += dy * s
    return _clamp(st)


# --- 뷰포트 ---
# 표시 루프와 DPU 워커가 둘 다 쓴다. 워커는 잘라낸 영역을 DPU 에 넣고, 표시 루프는
# 같은 영역을 화면에 띄운다. 상태에서만 나오는 순수 함수라 여기 둔다.

def crop_rect(w, h, st):
    """뷰포트 사각형 (x, y, cw, ch). zoom=1.0 이면 프레임 전체."""
    if st["zoom"] <= 1.0:
        return 0, 0, w, h
    cw, ch = int(w / st["zoom"]), int(h / st["zoom"])
    x = min(w - cw, max(0, int(st["cx"] * w - cw / 2)))
    y = min(h - ch, max(0, int(st["cy"] * h - ch / 2)))
    return x, y, cw, ch


def crop_view(frame, st):
    """cam1 디지털 PTZ. zoom=1.0 이면 원본 그대로 돌려준다."""
    h, w = frame.shape[:2]
    x, y, cw, ch = crop_rect(w, h, st)
    return frame if (cw, ch) == (w, h) else frame[y:y + ch, x:x + cw]


def self_test():
    """노트북 셀 6 의 자체 점검. 하드웨어 불필요."""
    s = INIT_STATE
    assert apply_event(s, "left") == s                          # 1) 꺼져 있으면 제스처 무시
    assert apply_event(s, "bye") == s
    assert apply_event(s, "open") == s                          #    꺼진 동안엔 모델도 안 바뀐다
    s = apply_event(s, "wake")
    assert s["on"] and s["dpu"] == "gesture"                    # 2) wake 로 켜지고 제스처 모델

    s = apply_event(s, "thumbs up")
    assert s["zoom"] > 1.0                                      # 3) 줌 인
    s = apply_event(s, "left")
    assert s["cx"] < 0.5                                        # 4) 줌 인 상태에서 팬이 먹는다
    s = apply_event(s, "up")
    assert s["cy"] < 0.5
    zoomed = (s["cx"], s["cy"], s["zoom"])

    assert not apply_event(s, "close")["on"]                    # 5) close 로 끈다
    assert apply_event(s, "bye") == s                           #    bye 는 아무 데도 안 묶인다
    s = apply_event(s, "open")
    assert s["dpu"] == "task"                                   # 6) open  -> cam1 xmodel 켠다

    # 7) 모델을 갈아 끼워도 cam1 뷰포트는 그대로 남는다
    assert (s["cx"], s["cy"], s["zoom"]) == zoomed
    s = apply_event(s, "wake")
    assert s["dpu"] == "gesture"                                # 8) 복귀는 wake 뿐
    assert (s["cx"], s["cy"], s["zoom"]) == zoomed

    for _ in range(20):
        s = apply_event(s, "thumbs down")
    assert s["zoom"] == ZOOM_MIN                                # 9) 줌 하한
    assert (s["cx"], s["cy"]) == (0.5, 0.5)                     # 10) 줌 아웃하면 중앙으로 돌아온다
    assert apply_event(s, "left")["cx"] == 0.5                  # 11) 배율 1 에서는 이동 여백이 없다
    for _ in range(20):
        s = apply_event(s, "thumbs up")
    assert s["zoom"] == ZOOM_MAX                                # 12) 줌 상한

    s = apply_event(s, "close")
    assert not s["on"]                                          # 13) close 로 꺼진다
    assert apply_event(s, "wake")["on"]
    assert apply_event(s, "hi") == s                            # 14) hi 는 무동작
    print("상태 기계 OK")

    _viewport_test()
    _display_test()


def _viewport_test():
    """뷰포트 — 워커가 여기서 자른 영역을 DPU 에 넣고 박스를 되돌린다."""
    st = apply_event(INIT_STATE, "wake")
    for _ in range(3):                     # 한 번만 당기면 팬이 가장자리로 클램프돼서
        st = apply_event(st, "thumbs up")  # 오프셋이 0 이 된다. 산술을 실제로 태우려고 3번.
    st = apply_event(apply_event(st, "left"), "up")

    x, y, cw, ch = crop_rect(WIDTH, HEIGHT, st)
    assert x > 0 and y > 0                                     # 1) 오프셋이 실제로 붙는다
    assert x + cw <= WIDTH and y + ch <= HEIGHT                #    사각형이 프레임 안에 있다
    assert cw < WIDTH and ch < HEIGHT                          # 2) 줌했으면 실제로 작아진다

    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    assert crop_view(frame, st).shape[:2] == (ch, cw)          # 3) 둘이 서로 맞는다
    assert crop_rect(WIDTH, HEIGHT, INIT_STATE) == (0, 0, WIDTH, HEIGHT)
    assert crop_view(frame, INIT_STATE).shape[:2] == (HEIGHT, WIDTH)   # 4) 배율 1 = 원본

    # 5) 크롭 좌표 -> 원본 좌표 되돌리기. 크롭 안 박스는 되돌리면 크롭 사각형 안에 있다.
    #    워커가 이걸 빼먹으면 cam1 박스가 화면과 어긋난다.
    boxes = np.array([[0.0, 0.0, cw - 1.0, ch - 1.0]], np.float32)
    boxes[:, [0, 2]] += x
    boxes[:, [1, 3]] += y
    assert (boxes[0][0], boxes[0][1]) == (x, y)
    assert boxes[0][2] <= WIDTH and boxes[0][3] <= HEIGHT

    # 6) 검출이 없을 때도 같은 산술이 안 터진다. decode() 가 (0, 4) 를 돌려주기 때문이다.
    empty = np.zeros((0, 4), np.float32).copy()
    empty[:, [0, 2]] += x
    empty[:, [1, 3]] += y
    assert empty.shape == (0, 4)

    print(f"뷰포트 OK — x{st['zoom']:.2f} 에서 {WIDTH}x{HEIGHT} -> {cw}x{ch} @({x},{y})")


def _display_test():
    """셀 12 를 다시 쓰면서 새로 생긴 코드만 본다. 카메라도 DPU 도 필요 없다.

    det=None 으로 두면 compose() 가 draw() 를 타지 않으므로 gesture_live 없이 돈다.
    """
    class Cam:
        def __init__(self, blank=False):
            self.label, self.fps, self.blank = "CAM test", 29.7, blank

        def snapshot(self):
            return None if self.blank else np.full((HEIGHT, WIDTH, 3), 40, np.uint8)

    class Worker:
        det = None                  # None 이라야 draw() 를 안 탄다
        loaded, dpu_ms, swap_ms, gesture, error = "x.xmodel", 18.4, 320.0, "left 0.91", None

    class Listener:
        n_wake, score, drops, error = 3, 0.62, 0, None

    w, h = DISP_WH
    cams = [Cam(), Cam()]
    st = apply_event(apply_event(INIT_STATE, "wake"), "thumbs up")

    assert compose(cams, Worker(), st).shape == (h, w * 2, 3)       # 1) 두 장이 가로로 붙는다
    assert compose([Cam(blank=True), Cam()], Worker(), st).shape == (h, w * 2, 3)
    #    2) 한쪽이 아직 프레임을 안 줘도 hstack 이 안 터진다

    deep = st                                                      # 3) 깊게 줌·팬해도 크기 유지
    for _ in range(8):
        deep = apply_event(deep, "thumbs up")
    deep = apply_event(deep, "left")
    assert compose(cams, Worker(), deep).shape == (h, w * 2, 3)

    assert off_canvas().shape == (h, w * 2, 3)                     # 4) OFF 도 같은 크기 (창이 안 튄다)

    lines = status_lines(Worker(), Listener(), st)                 # 5) 2줄, 전부 ASCII
    assert len(lines) == 2
    for line in lines:
        line.encode("ascii")       # Hershey 폰트에 없는 글자가 섞이면 여기서 걸린다
    err = Worker()
    err.error = "RuntimeError: boom"
    assert "ERR" in status_lines(err, Listener(), st)[1]           # 6) 에러가 상태줄에 뜬다

    # 7) 데모 키가 전부 실제 이벤트다. 오타가 있으면 상태가 안 바뀌어서 걸린다.
    #    (배율 1.0 에서는 팬 여백이 0 이라 안 움직이므로 미리 당겨 둔다)
    on = apply_event(apply_event(INIT_STATE, "wake"), "thumbs up")
    for k, ev in DEMO_KEYS.items():
        assert ev in ("wake", "hi", "bye") or apply_event(on, ev) != on, (k, ev)
    assert set(DEMO_KEYS.values()) >= set(PAN) | set(DPU_BY_EVENT) | {"wake", "bye", "close"}

    # 8) 작업 모델 클래스 목록. 개수가 틀리면 보드에서 reg_max mismatch 로 죽고,
    #    한글이 섞이면 화면 라벨이 전부 물음표로 찍힌다 (5번과 같은 이유).
    assert len(TASK_CLASSES) == 9, len(TASK_CLASSES)       # tempModel/data.yaml
    for name in TASK_CLASSES:
        name.encode("ascii")

    print(f"표시 경로 OK — 합성 {(h, w * 2, 3)} · 키 {len(DEMO_KEYS)}개 "
          f"· 작업 클래스 {len(TASK_CLASSES)}개")


# ============================================================ 셀 8 · 웨이크 워드 리스너
# 이 스레드는 DPU 를 전혀 만지지 않는다. 그래서 xmodel 교체 중에도, 제스처 추론이
# DPU 를 붙잡고 있어도 계속 듣는다.
#
# import 를 파일 맨 위로 올리지 않은 것은 셀 경계를 그대로 두기 위해서다 (NOTEBOOK_TO_PY.md §2).

if not SELF_TEST:
    import sounddevice as sd
    from openwakeword.model import Model


def resolve_mic(index):
    """마이크 장치를 확정해 (device, 이름) 을 돌려준다. 못 찾으면 기본 입력으로 물러난다.

    인덱스가 사라지는 것은 흔한 일이다 (MIC_INDEX 주석 참고). 그때 죽어 버리면 마이크
    스레드가 통째로 나가고 시연 중 웨이크 워드가 영영 안 온다 — 물러나는 편이 낫다.
    """
    try:
        return index, sd.query_devices(index, "input")["name"]
    except Exception as e:
        print(f"[MIC ] idx={index} 를 못 찾았다 ({e}). 시스템 기본 입력으로 간다")
        return None, sd.query_devices(None, "input")["name"]


class WakeListener(threading.Thread):
    """마이크를 계속 듣다가 웨이크 워드가 잡히면 event_q 에 "wake" 를 넣는다."""

    def __init__(self, event_q):
        super().__init__(daemon=True)
        self.event_q = event_q
        self.q = queue.Queue()
        self.score = 0.0
        self.n_wake = 0
        self.drops = 0
        self.error = None
        self._running = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            self.drops += 1
        self.q.put(indata[:, 0].copy())

    def run(self):
        self._running = True
        try:
            oww = Model(wakeword_models=[WAKEWORD_MODEL], inference_framework="onnx")
            key = list(oww.models.keys())[0]
            mic, mic_name = resolve_mic(MIC_INDEX)
            last_fire = -1e9
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                                blocksize=CHUNK, device=mic, callback=self._callback):
                print(f"[MIC ] 듣는 중 — \"{key}\"  @ {mic_name}")
                while self._running:
                    try:
                        frame = self.q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    self.score = float(oww.predict(frame)[key])
                    now = time.monotonic()
                    if self.score > WAKE_THRESHOLD and (now - last_fire) > WAKE_REFRACTORY:
                        last_fire = now
                        self.n_wake += 1
                        print(f"[WAKE] score={self.score:.3f}")
                        self.event_q.put("wake")
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[MIC ] 죽었다 — {self.error}")

    def stop(self):
        self._running = False
        self.join(timeout=3.0)


# ============================================================ 셀 10 · DPU 워커
# 모드 전환 = rt.load_model(). 비트스트림은 Runner() 생성 때 한 번만 올라간다.
# off 에서는 아무 모델도 올리지 않고 DPU 를 놀린다.

class DpuWorker(threading.Thread):
    """상태에 따라 xmodel 을 갈아 끼우고, 인식된 제스처를 이벤트 큐로 보낸다."""

    def __init__(self, event_q, cams):
        super().__init__(daemon=True)
        self.event_q = event_q
        self.cams = cams            # [cam0(제스처), cam1(작업 모델 · PTZ 대상)]
        self.state = INIT_STATE     # 통째로 교체만 한다 — 표시 루프가 반쪽 상태를 볼 일이 없다
        self.loaded = os.path.basename(GESTURE_XMODEL)
        self.det = None             # {"cam": i, "boxes"/"scores"/"cls_ids"/"labels"}
        self.dpu_ms = 0.0
        self.swap_ms = 0.0
        self.gesture = ""
        self.error = None
        self.ready = False
        self._running = False

    def _drain(self, st):
        while True:
            try:
                ev = self.event_q.get_nowait()
            except queue.Empty:
                return st
            new = apply_event(st, ev)
            if new != st:
                print(f"[MODE] {ev:12s} -> on={new['on']} dpu={new['dpu']} "
                      f"zoom={new['zoom']:.2f} center=({new['cx']:.2f}, {new['cy']:.2f})")
            st = new

    def run(self):
        self._running = True

        # PYNQ 는 인터럽트를 asyncio 로 다뤄서 오버레이를 만질 때 이벤트 루프를 요구한다.
        # 워커 스레드에는 루프가 없고, 3.10+ 부터는 자동 생성도 안 해준다:
        #   RuntimeError: There is no current event loop in thread 'Thread-N'
        # 교체(load_model)도 이 스레드에서 도니 생성만 메인 스레드로 옮겨서는 못 고친다.
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())

        try:
            rt = Runner(BIT_PATH, GESTURE_XMODEL)
            gru = GestureClassifier(GRU_PATH)
            print(f"[DPU ] input={rt.in_dims} layout={rt.layout} int8={rt.in_is_int8}")
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[DPU ] 초기화 실패 — {self.error}")
            return

        seqbuf = SequenceBuffer(window_s=WINDOW_S)
        st = INIT_STATE
        run_mode = "off"             # 직전 루프가 실제로 돌던 모드
        n, cooldown_until = 0, 0.0
        self.ready = True

        while self._running:
            st = self._drain(st)
            self.state = st
            mode = "off" if not st["on"] else st["dpu"]     # off | gesture | task

            if mode != run_mode:
                # off 가 아닐 때만 DPU 에 모델을 올린다.
                if mode != "off":
                    want = GESTURE_XMODEL if mode == "gesture" else TASK_XMODEL
                    t0 = time.perf_counter()
                    rt.load_model(want)
                    self.swap_ms = (time.perf_counter() - t0) * 1000
                    self.loaded = os.path.basename(want)
                    print(f"[SWAP] {mode} · {self.loaded} 로 교체  "
                          f"{self.swap_ms:.0f} ms  (입력 {rt.in_w}x{rt.in_h})")
                # 교체 전 1초 창을 버린다. 안 버리면 제스처 모드로 돌아오는 순간 직전
                # 제스처가 그대로 다시 발화한다.
                seqbuf.buf.clear()
                cooldown_until = time.time() + GESTURE_COOLDOWN
                self.det = None
                self.gesture = ""
                run_mode = mode

            if mode == "off":               # DPU 를 아예 쓰지 않는다
                time.sleep(0.05)
                continue

            idx = 0 if mode == "gesture" else 1
            frame = self.cams[idx].snapshot()
            if frame is None:
                time.sleep(0.01)
                continue

            if mode == "task":
                # 줌한 영역만 DPU 에 넣는다. 멀리 있는 손이 커져서 실제로 잡힌다.
                fh, fw = frame.shape[:2]
                ox, oy, cw, ch = crop_rect(fw, fh, st)
                frame = frame[oy:oy + ch, ox:ox + cw]
                conf, ncls, labels = TASK_CONF, len(TASK_CLASSES), TASK_CLASSES
                mdets = TASK_MAX_DETS
            else:
                ox, oy = 0, 0
                conf, ncls, labels = CONF, len(SHAPE_CLASSES), SHAPE_CLASSES
                mdets = 1            # 2단계가 boxes[0] 로 시퀀스를 만든다. 손은 하나다

            inp, ratio, pad, orig_wh = rt.preprocess(frame)
            t0 = time.perf_counter()
            outputs = rt.infer(inp)
            boxes, scores, cls_ids = decode(outputs, ratio, pad, orig_wh, conf, ncls, mdets)
            self.dpu_ms = (time.perf_counter() - t0) * 1000

            # 박스를 원본 프레임 좌표로 되돌린다. 표시 루프는 지금처럼 원본에 그린 뒤
            # 크롭하면 되고, 그리는 시점에 배율이 바뀌어 있어도 박스가 어긋나지 않는다.
            if ox or oy:
                boxes = boxes.copy()
                boxes[:, [0, 2]] += ox
                boxes[:, [1, 3]] += oy

            self.det = {"cam": idx, "boxes": boxes, "scores": scores,
                        "cls_ids": cls_ids, "labels": labels}

            if mode == "task":
                continue             # 작업 모델은 그리기만 한다. 2단계(GRU)는 제스처 전용

            # ---- 2단계: 특징 시퀀스 -> GRU (gesture_live.main 과 같은 규칙) ----
            now = time.time()
            if len(boxes):
                x1, y1, x2, y2 = boxes[0]
                ow, oh = orig_wh
                det = ((x1 + x2) / 2 / ow, (y1 + y2) / 2 / oh,
                       (x2 - x1) / ow, (y2 - y1) / oh,
                       float(scores[0]), int(cls_ids[0]))
            else:
                det = None
            if now < cooldown_until:
                seqbuf.buf.clear()   # 쿨다운 중엔 아무것도 쌓지 않는다. 끝난 뒤 window_s
                                     # 만큼 새로 채워야 ready() 가 되므로 복귀동작은
                                     # 다음 창에 들어올 수 없다
                continue
            seqbuf.push(now, det)

            n += 1
            if n % GRU_EVERY or not seqbuf.ready(now):
                continue

            seq = build_sequence(seqbuf.sample(now))
            if seq is None:
                continue

            active, disp, shape = is_active(seq)
            if not active:                       # 정지 게이트: 손이 멈춰 있으면 GRU 를 안 부른다
                self.gesture = ""
                continue

            prob = gru.predict(seq)
            k = int(np.argmax(prob))
            if prob[k] < GESTURE_THRES:
                continue

            cooldown_until = now + GESTURE_COOLDOWN
            self.gesture = f"{GESTURE_CLASSES[k]} {prob[k]:.2f}"
            print(f"[GEST] {GESTURE_CLASSES[k]:12s} p={prob[k]:.3f}")
            # 상태를 직접 건드리지 않고 wake 와 같은 큐로 보낸다. 소비자는 _drain 한 곳뿐이다.
            self.event_q.put(GESTURE_CLASSES[k])

    def stop(self):
        self._running = False
        self.join(timeout=5.0)


# ============================================================ 셀 12 · 화면 (다시 쓴 부분)
# 노트북은 ipywidgets.Image 에 JPEG 를 밀어 넣었다. .py 에는 밀어 넣을 프론트엔드가
# 없으므로 cv2.imshow 로 바꿨다. 근거는 NOTEBOOK_TO_PY.md §1.
#
# 화면에 얹는 글자는 전부 ASCII 다 — cv2.putText 에 한글 폰트가 없다 (§3).

WINDOW = "02 wake-gesture-switch"

STATE_COLOR = {"off": (120, 120, 120),
               "gesture": (0, 220, 120), "task": (0, 180, 255)}

# 위젯 버튼(Wake 수동)과, 노트북 6장의 "event_q.put("open") 을 직접 부른다"를 한꺼번에
# 대신한다. 마이크와 손 없이 시연 경로 전체를 탈 수 있다.
DEMO_KEYS = {
    "w": "wake",
    "1": "left", "2": "right", "3": "up", "4": "down",
    "5": "thumbs up", "6": "thumbs down",
    "7": "open", "8": "close", "9": "bye", "0": "hi",
}


def state_label(st):
    return "off" if not st["on"] else st["dpu"]


# crop_view()/crop_rect() 는 셀 6 구간으로 옮겼다. 워커도 같은 뷰포트를 봐야 한다.


def overlay(frame, text, active, color):
    """좌상단 라벨 + 활성 카메라 테두리."""
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (330, 34), (0, 0, 0), -1)
    out = cv2.addWeighted(out, 0.65, frame, 0.35, 0)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 120), 1, cv2.LINE_AA)
    if active:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 4)
    return out


def compose(streams, worker, st):
    """두 카메라를 한 장으로 합친다. 노트북 loop() 의 그리는 부분과 같다."""
    det = worker.det
    color = STATE_COLOR[state_label(st)]
    panes = []
    for i, s in enumerate(streams):
        f = s.snapshot()
        if f is None:
            panes.append(np.zeros((DISP_WH[1], DISP_WH[0], 3), np.uint8))
            continue
        if det is not None and det["cam"] == i:
            # 워커가 박스를 원본 좌표로 돌려주므로 크롭 전에 그리면 된다
            draw(f, det["boxes"], det["scores"], det["cls_ids"], det["labels"])
        if i == 1:
            f = crop_view(f, st)
        f = cv2.resize(f, DISP_WH, interpolation=cv2.INTER_AREA)
        tag = (f"{s.label}  {s.fps:4.1f} fps" if i == 0
               else f"{s.label}  x{st['zoom']:.2f}")
        panes.append(overlay(f, tag, det is not None and det["cam"] == i, color))
    return np.hstack(panes)


def off_canvas():
    """OFF 일 때도 창은 갱신해야 한다 — 안 그러면 waitKey 가 안 돌아 w 키를 못 받는다."""
    c = np.zeros((DISP_WH[1], DISP_WH[0] * 2, 3), np.uint8)
    cv2.putText(c, "OFF - say the wake word (or press w)",
                (24, DISP_WH[1] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (120, 120, 120), 2, cv2.LINE_AA)
    return c


def status_lines(worker, listener, st):
    """노트북 status 위젯과 같은 값.

    화면에 얹으므로 ASCII(32~126)만 쓴다. Hershey 폰트에는 그 밖의 글자가 없어서
    한글도 가운뎃점(·)도 전부 물음표로 찍힌다. 구분자가 '|' 인 이유가 그것이다.
    """
    err = worker.error or listener.error
    return [
        f"{state_label(st).upper()} | {worker.loaded} | "
        f"DPU {worker.dpu_ms:.0f}ms | swap {worker.swap_ms:.0f}ms",
        f"zoom x{st['zoom']:.2f} ({st['cx']:.2f},{st['cy']:.2f}) | "
        f"gest {worker.gesture or '-'} | "
        f"wake {listener.n_wake} (s{listener.score:.2f} d{listener.drops})"
        + (f" | ERR {err}" if err else ""),
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--no-display", action="store_true",
                   help="헤드리스 실행 (SSH 등). DISPLAY 가 없으면 자동으로 켜진다")
    p.add_argument("--duration", type=float, default=0.0,
                   help="지정 초 후 자동 종료 (0 = 무제한)")
    p.add_argument("--self-test", action="store_true",
                   help="상태 기계 자체 점검만 하고 끝낸다. 하드웨어 불필요")
    return p.parse_args()


def stop_all(streams, worker, listener):
    for obj in [listener, worker, *streams]:
        try:
            obj.stop()
        except Exception:
            pass


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    show = not args.no_display and bool(os.environ.get("DISPLAY"))
    print_config()
    if show:
        legend = "  ".join(f"{k}={v}" for k, v in DEMO_KEYS.items())
        print(f"[KEY ] q=quit  {legend}")
    else:
        print("[INFO] 헤드리스 모드. 상태는 표준출력으로만 나온다.")

    event_q = queue.Queue()
    streams = [CamStream(CAM_A_ID, "CAM_A user"), CamStream(CAM_B_ID, "CAM_B target")]
    worker = listener = None
    try:
        for s in streams:
            if not s.open():
                print(f"[FAIL] {s.label} (/dev/video{s.dev_id}) 열기 실패")
                print("       dmesg | grep -i 'Not enough bandwidth' 로 확인할 것")
                return 1
            s.start()
        time.sleep(0.5)

        listener = WakeListener(event_q)
        listener.start()
        worker = DpuWorker(event_q, streams)
        worker.start()
        print("[INFO] 시작 중… (DPU 비트스트림 로드에 몇 초 걸린다)")

        # 표시 루프는 메인 스레드다. imshow/waitKey 는 창을 만든 스레드에서 불러야 한다.
        period = 1.0 / DISPLAY_FPS
        t_start = time.monotonic()
        n = 0
        while True:
            t0 = time.monotonic()
            st = worker.state              # 딕셔너리 통째 교체라 스냅샷 한 번이면 된다
            lines = status_lines(worker, listener, st)

            if show:
                canvas = compose(streams, worker, st) if st["on"] else off_canvas()
                for j, line in enumerate(lines):
                    y = canvas.shape[0] - 26 + 14 * j
                    cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, canvas)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                ev = DEMO_KEYS.get(chr(k)) if k != 255 else None
                if ev:
                    print(f"[KEY ] {ev}")
                    event_q.put(ev)
            elif n % DISPLAY_FPS == 0:
                print("[STAT] " + " · ".join(lines))

            n += 1
            if args.duration > 0 and (time.monotonic() - t_start) >= args.duration:
                print("[INFO] duration 도달. 종료한다.")
                break
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단")
    finally:
        stop_all(streams, worker, listener)
        if show:
            cv2.destroyAllWindows()
        print("카메라·마이크·DPU 해제 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
