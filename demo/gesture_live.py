"""
KV260 2-stage gesture recognition (live)

  Stage 1  YOLOv8s (SiLU->ReLU)  ->  DPU        : 프레임 1장 -> bbox + 손모양 4클래스
  Stage 2  GRU 2층 (tflite)      ->  Cortex-A53 : 30프레임 시퀀스 -> 제스처 10클래스

basic.py(단일 단계 RPS 검출)를 기반으로, 학습 노트북
KV260_Gesture_YOLO_GRU_Finetuning 의 trial_to_sequence() 와 동일한 규칙으로
특징 시퀀스를 구성해 GRU 에 넘긴다.

학습과 반드시 일치시켜야 하는 것:
  - CONF_THRES = 0.15  (시퀀스 추출 때 쓴 conf 값)
  - 특징 11차원 구성과 순서
  - 미검출 프레임의 conf=0.0 표식 + 선형 보간
  - 30프레임이 덮는 실제 시간 (--window)
"""

import argparse
import collections
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pynq_dpu import DpuOverlay


# ---------------------------------------------------------------- 상수
MODEL_PATH = "./gesture_stage1_kv260.xmodel"
GRU_PATH = "./gesture_stage2_gru.tflite"
BIT_PATH = "dpu.bit"

SHAPE_CLASSES = ["open_palm", "fist", "thumb_up", "thumb_down"]        # Stage 1 (DPU)
GESTURE_CLASSES = ["left", "right", "up", "down", "hi", "bye",
                   "open", "close", "thumbs up", "thumbs down"]        # Stage 2 (ARM)

STRIDES = [8, 16, 32]
CAMERA_INDEX = 0            # CAM_A. 00_dual_camera_check 기준 /dev/video0
FRAME_WIDTH = 640           # 같은 노트북에서 확정된 스펙 (MJPG 640x480)
FRAME_HEIGHT = 480
FOURCC = "MJPG"

CONF_THRES = 0.15           # 시퀀스 추출 때 쓴 값과 반드시 동일해야 한다
IOU_THRES = 0.45
NUM_CLASSES = len(SHAPE_CLASSES)
REG_MAX = 16
MAX_DETS = 1
COLORS = [(30, 200, 30), (200, 30, 30), (30, 30, 200), (200, 160, 30)]

# ---- Stage 2 ----
SEQ_LEN = 30
FEAT_DIM = 11
SEQ_WINDOW_S = 1.0      # 학습 데이터의 30프레임이 덮던 실제 시간. 촬영 fps 로 확정할 것
GRU_EVERY = 5           # 매 프레임 돌릴 필요 없다. 5프레임마다 1회
GESTURE_THRES = 0.70
COOLDOWN_S = 1.0        # 제스처 1회 발화 후 이 시간 동안 예측 정지 + 시퀀스 버퍼 폐기.
                        # 복귀동작이 다음 창에 섞여 재발화하는 것을 막는다.

# ---- 정지(idle) 게이트 ----
# 학습 데이터에는 '아무 제스처도 아님' 클래스가 없다. 960개 trial 이 전부 의도적으로
# 수행된 30프레임이라, 손을 가만히 두면 GRU 는 10개 중 하나를 고를 수밖에 없고
# 실측상 항상 'bye' 를 높은 확률(중앙값 0.951)로 뱉는다. 임계값 상향으로는 못 막는다.
# 그래서 GRU 호출 전에 시퀀스의 활동량을 재서 정지면 건너뛴다.
DISP_MIN = 0.04         # bbox 중심 이동 범위
SHAPE_MIN = 0.5         # 손모양 원-핫 변화량


# ---------------------------------------------------------------- 수치 유틸
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def letterbox(img: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)

    rw = int(round(w * r))
    rh = int(round(h * r))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)

    dw = nw - rw
    dh = nh - rh
    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def scale_boxes_back(boxes: np.ndarray, ratio: float, pad: Tuple[int, int],
                     ow: int, oh: int) -> np.ndarray:
    px, py = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / ratio
    return clip_boxes(boxes, ow, oh)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    idx = cv2.dnn.NMSBoxes(
        bboxes=[[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                for x1, y1, x2, y2 in boxes.astype(np.float32)],
        scores=scores.astype(np.float32).tolist(),
        score_threshold=0.0,
        nms_threshold=iou_thres,
    )
    if len(idx) == 0:
        return np.array([], dtype=np.int32)
    return np.array(idx).reshape(-1)


# ---------------------------------------------------------------- DPU 텐서 메타
def tensor_dims(tensor) -> Tuple[int, ...]:
    return tuple(int(x) for x in tensor.dims)


def tensor_fixpos(tensor) -> int:
    try:
        if tensor.has_attr("fix_point"):
            return int(tensor.get_attr("fix_point"))
    except Exception:
        pass
    return 0


def tensor_is_int8(tensor) -> bool:
    info = ""
    if hasattr(tensor, "dtype"):
        info += str(tensor.dtype).lower()
    if hasattr(tensor, "get_data_type"):
        try:
            info += str(tensor.get_data_type()).lower()
        except Exception:
            pass
    return "int8" in info or "xint8" in info


def valid_head_channels(c: int, num_classes: int) -> bool:
    return c > num_classes and ((c - num_classes) % 4 == 0)


def to_nchw_head(out: np.ndarray, num_classes: int) -> np.ndarray:
    if out.ndim != 4 or out.shape[0] != 1:
        raise RuntimeError(f"Unexpected output shape: {list(out.shape)}")

    _, a, b, c = out.shape
    nhwc_valid = valid_head_channels(c, num_classes)
    nchw_valid = valid_head_channels(a, num_classes)

    if nhwc_valid and (c >= a) and (c >= b):
        return np.transpose(out, (0, 3, 1, 2)).astype(np.float32)
    if nchw_valid:
        return out.astype(np.float32)
    if nhwc_valid:
        return np.transpose(out, (0, 3, 1, 2)).astype(np.float32)

    raise RuntimeError(f"Cannot infer layout from {list(out.shape)}")


# ---------------------------------------------------------------- Stage 1 디코딩
def decode(outputs: List[np.ndarray], ratio: float, pad: Tuple[int, int],
           orig_wh: Tuple[int, int], conf_thres: float):
    feats = [to_nchw_head(o, NUM_CLASSES) for o in outputs]
    feats = sorted(feats, key=lambda t: t.shape[2] * t.shape[3], reverse=True)

    all_boxes = []
    all_probs = []

    for feat, stride in zip(feats, STRIDES):
        _, c, h, w = feat.shape
        reg_max = (c - NUM_CLASSES) // 4
        if reg_max != REG_MAX:
            raise RuntimeError(f"reg_max mismatch: expected={REG_MAX}, got={reg_max}")

        dist_logits = feat[:, : 4 * REG_MAX, :, :].reshape(1, 4, REG_MAX, h, w)
        cls_logits = feat[:, 4 * REG_MAX: 4 * REG_MAX + NUM_CLASSES, :, :]

        dist_prob = softmax(dist_logits, axis=2)
        bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, REG_MAX, 1, 1)
        ltrb = np.sum(dist_prob * bins, axis=2)[0]

        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5,
                             np.arange(h, dtype=np.float32) + 0.5)
        cx = gx.reshape(-1)
        cy = gy.reshape(-1)

        l = ltrb[0].reshape(-1)
        t = ltrb[1].reshape(-1)
        r = ltrb[2].reshape(-1)
        b = ltrb[3].reshape(-1)

        x1 = (cx - l) * float(stride)
        y1 = (cy - t) * float(stride)
        x2 = (cx + r) * float(stride)
        y2 = (cy + b) * float(stride)
        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))

        probs = sigmoid(cls_logits[0]).transpose(1, 2, 0).reshape(-1, NUM_CLASSES)
        all_probs.append(probs)

    boxes = np.concatenate(all_boxes, axis=0)
    probs = np.concatenate(all_probs, axis=0)

    cls_ids = np.argmax(probs, axis=1).astype(np.int32)
    scores = probs[np.arange(len(probs)), cls_ids].astype(np.float32)

    keep = scores > conf_thres
    boxes = boxes[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    if len(boxes) == 0:
        return boxes, scores, cls_ids

    keep = nms(boxes, scores, IOU_THRES)
    boxes = boxes[keep].astype(np.float32)
    scores = scores[keep].astype(np.float32)
    cls_ids = cls_ids[keep].astype(np.int32)

    ow, oh = orig_wh
    boxes = scale_boxes_back(boxes, ratio, pad, ow, oh)

    if len(scores) > MAX_DETS:
        order = np.argsort(scores)[::-1][:MAX_DETS]
        boxes = boxes[order]
        scores = scores[order]
        cls_ids = cls_ids[order]

    return boxes, scores, cls_ids


# ---------------------------------------------------------------- Stage 2 특징
def build_sequence(raw: List[Optional[tuple]]) -> Optional[np.ndarray]:
    """학습 노트북의 trial_to_sequence() 와 동일한 규칙으로 (30, 11) 특징 생성.

    raw[i] = (cx, cy, w, h, conf, cls)  또는  None (미검출)
    좌표는 전부 원본 프레임 기준 정규화 값(0~1)이어야 한다 -- 학습이 xywhn 을 썼다.
    """
    if len(raw) != SEQ_LEN:
        raise ValueError(f"raw length {len(raw)} != SEQ_LEN {SEQ_LEN}")

    seq = np.zeros((SEQ_LEN, FEAT_DIM), dtype=np.float32)
    idx = [i for i, v in enumerate(raw) if v is not None]
    if not idx:
        return None                                   # 전 구간 미검출 -> 폐기

    arr = np.array([raw[i][:4] for i in idx], dtype=np.float32)

    for i in range(SEQ_LEN):
        if raw[i] is not None:
            cx, cy, w, h, conf, cls = raw[i]
        else:
            cx, cy, w, h = (float(np.interp(i, idx, arr[:, k])) for k in range(4))
            conf = 0.0                                            # 보간 프레임 표식
            cls = raw[min(idx, key=lambda j: abs(j - i))][5]      # 최근접 검출의 클래스
        seq[i, 0:4] = (cx, cy, w, h)
        seq[i, 4] = conf
        seq[i, 5 + int(cls)] = 1.0

    seq[1:, 9] = seq[1:, 0] - seq[:-1, 0]      # dcx
    seq[1:, 10] = seq[1:, 1] - seq[:-1, 1]     # dcy
    return seq


def is_active(seq: np.ndarray, disp_min: float = DISP_MIN,
              shape_min: float = SHAPE_MIN) -> Tuple[bool, float, float]:
    """이 시퀀스가 제스처 수행 중인지 판정한다. 정지면 GRU 를 호출하지 않는다.

    두 경로를 OR 로 묶는 이유:
      - 이동 계열(left/right/up/down/hi/bye) 은 bbox 중심이 크게 움직인다.
      - 형태 계열(open/close/thumbs) 은 손이 제자리에 있어 변위가 거의 없다.
        대신 손모양 원-핫이 뒤집히며, 전이 1회당 변화량이 정확히 2.0 이 된다.
      실측: live close 세션의 변위는 최대 0.0525 로 전부 임계값 아래였다.
      변위 조건만 두면 close/open/thumbs 가 통째로 막힌다.

    임계값 근거 (live 덤프 3세션 + 학습 시퀀스 957개):
      idle  변위 최대 0.0190,  원-핫 변화량 정확히 0.00
      학습  disp_min=0.04 일 때 실제 제스처 차단률 1.5% (14/957)
            disp_min=0.06 이면 3.0% 로 오르고 특히 hi 가 12/97 차단된다.
            hi 는 얼굴 옆 손목 흔들기라 중심 이동이 작다(중앙값 0.168).
    """
    disp = float(np.hypot(seq[:, 0].max() - seq[:, 0].min(),
                          seq[:, 1].max() - seq[:, 1].min()))
    shape = float(np.abs(np.diff(seq[:, 5:9], axis=0)).sum())
    return (disp > disp_min) or (shape > shape_min), disp, shape


class SequenceBuffer:
    """타임스탬프가 붙은 원시 검출을 쌓고, 고정 시간 창을 30점으로 리샘플링한다.

    프레임 수로 세지 않는 이유: 학습 데이터의 30프레임이 덮던 시간과 보드 카메라의
    fps 가 다르면 dcx/dcy(속도)가 통째로 스케일이 어긋난다. 시간 창을 고정하면
    카메라 fps 가 흔들려도 GRU 가 보는 속도가 유지된다.
    """

    def __init__(self, window_s: float = SEQ_WINDOW_S, maxlen: int = 600):
        self.buf = collections.deque(maxlen=maxlen)
        self.window_s = window_s

    def push(self, t: float, det: Optional[tuple]):
        self.buf.append((t, det))

    def ready(self, now: float) -> bool:
        return len(self.buf) >= 2 and (now - self.buf[0][0]) >= self.window_s

    def sample(self, now: float) -> List[Optional[tuple]]:
        ts = np.array([t for t, _ in self.buf])
        targets = np.linspace(now - self.window_s, now, SEQ_LEN)
        return [self.buf[int(np.argmin(np.abs(ts - tt)))][1] for tt in targets]


class GestureClassifier:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"tflite not found: {path}")
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter        # 보드에 TF 가 있으면 이쪽

        self.itp = Interpreter(model_path=path)
        self.itp.allocate_tensors()
        i0 = self.itp.get_input_details()[0]
        self.itp.resize_tensor_input(i0["index"], [1, SEQ_LEN, FEAT_DIM])
        self.itp.allocate_tensors()
        self.i0 = self.itp.get_input_details()[0]          # resize 후 다시 조회해야 한다
        self.o0 = self.itp.get_output_details()[0]

    def predict(self, seq: np.ndarray) -> np.ndarray:
        self.itp.set_tensor(self.i0["index"], seq[None].astype(np.float32))
        self.itp.invoke()
        return self.itp.get_tensor(self.o0["index"])[0]


# ---------------------------------------------------------------- DPU 실행기
class Runner:
    def __init__(self, bit_path: str, model_path: str):
        self.overlay = DpuOverlay(bit_path)      # 비트스트림은 평생 한 번만
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """DPU 에 올라간 xmodel 을 교체한다. 비트스트림은 건드리지 않는다.

        DPU 가 하나뿐이라 두 모델을 동시에 올릴 수 없다. 전환할 때마다 여기를 부른다.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"xmodel not found: {model_path}")

        # 이전 runner 를 먼저 놓아준다. 붙잡은 채로 다시 올리면 DPU 가 사용 중이라고 거절한다
        self.runner = None
        self.overlay.load_model(model_path)
        self.runner = self.overlay.runner
        self.model_path = model_path

        self.in_tensor = self.runner.get_input_tensors()[0]
        self.in_dims = tensor_dims(self.in_tensor)
        self.in_fix = tensor_fixpos(self.in_tensor)
        self.in_is_int8 = tensor_is_int8(self.in_tensor)

        if self.in_dims[-1] == 3:
            self.layout = "NHWC"
            _, self.in_h, self.in_w, _ = self.in_dims
        elif self.in_dims[1] == 3:
            self.layout = "NCHW"
            _, _, self.in_h, self.in_w = self.in_dims
        else:
            raise RuntimeError(f"Unsupported input dims: {self.in_dims}")

        self.out_meta = []
        for t in self.runner.get_output_tensors():
            self.out_meta.append({
                "dims": tensor_dims(t),
                "fix": tensor_fixpos(t),
                "is_int8": tensor_is_int8(t),
            })

    def preprocess(self, frame: np.ndarray):
        oh, ow = frame.shape[:2]
        img, ratio, pad = letterbox(frame, (self.in_h, self.in_w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.layout == "NCHW":
            img = np.transpose(img, (2, 0, 1))

        img = np.expand_dims(img, axis=0)

        if self.in_is_int8:
            scale = float(2 ** self.in_fix)
            img = np.clip(np.round(img * scale), -128, 127).astype(np.int8)
        else:
            img = img.astype(np.float32)

        return img, ratio, pad, (ow, oh)

    def infer(self, inp: np.ndarray) -> List[np.ndarray]:
        outputs = []
        buffers = []
        for m in self.out_meta:
            dtype = np.int8 if m["is_int8"] else np.float32
            buffers.append(np.empty(m["dims"], dtype=dtype, order="C"))

        job_id = self.runner.execute_async([inp], buffers)
        self.runner.wait(job_id)

        for i, out in enumerate(buffers):
            if self.out_meta[i]["is_int8"]:
                scale = float(2 ** self.out_meta[i]["fix"])
                outputs.append(out.astype(np.float32) / scale)
            else:
                outputs.append(out.astype(np.float32))

        return outputs


# ---------------------------------------------------------------- 시퀀스 덤프
class SequenceDumper:
    """실시간 특징 시퀀스를 모아 npz 로 저장한다.

    Colab 의 sequences.npz 와 같은 형식(X, y, groups)에 실시간 메타를 덧붙인다.
    학습 분포와 보드 분포가 어긋났는지 확인하는 것이 목적이다.
    비교 예:
        tr = np.load('sequences.npz'); lv = np.load('live_seq.npz')
        for k, name in enumerate(['cx','cy','w','h','conf']):
            print(name, tr['X'][...,k].mean(), lv['X'][...,k].mean())
    """

    def __init__(self, path: str, label: str = ""):
        self.path = path
        self.label = label
        self.X, self.probs, self.times, self.fired = [], [], [], []

    def add(self, seq: np.ndarray, prob: np.ndarray, t: float, fired: bool):
        self.X.append(seq.astype(np.float32))
        self.probs.append(prob.astype(np.float32))
        self.times.append(float(t))
        self.fired.append(bool(fired))

    def __len__(self):
        return len(self.X)

    def save(self):
        if not self.X:
            print("[DUMP] 저장할 시퀀스가 없다.")
            return
        X = np.stack(self.X)
        np.savez_compressed(
            self.path,
            X=X,
            probs=np.stack(self.probs),
            times=np.array(self.times, dtype=np.float64),
            fired=np.array(self.fired, dtype=bool),
            label=np.array(self.label),
            classes=np.array(GESTURE_CLASSES),
            shape_classes=np.array(SHAPE_CLASSES),
        )
        print(f"[DUMP] {len(self.X)}개 시퀀스 저장: {self.path}  X={X.shape}")

        # 학습 분포와 바로 대조할 수 있도록 요약을 같이 찍는다
        names = ["cx", "cy", "w", "h", "conf", *[f"oh_{s}" for s in SHAPE_CLASSES],
                 "dcx", "dcy"]
        print("[DUMP] 차원별 mean/std (학습 sequences.npz 와 비교할 것)")
        for k, nm in enumerate(names):
            col = X[..., k]
            print(f"  {k:2d} {nm:12s} mean={col.mean(): .4f}  std={col.std(): .4f}"
                  f"  min={col.min(): .4f}  max={col.max(): .4f}")

        det_rate = float((X[..., 4] > 0).mean())
        print(f"[DUMP] 실검출 프레임 비율 {det_rate:.1%} "
              f"(너무 낮으면 --conf 를 더 낮추거나 카메라 거리를 조정할 것)")


# ---------------------------------------------------------------- 표시
def draw(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray):
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = box.astype(np.int32)
        name = SHAPE_CLASSES[cls_id] if 0 <= cls_id < len(SHAPE_CLASSES) else str(cls_id)
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name}:{score:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# ---------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(description="KV260 2-stage gesture recognition (live)")
    p.add_argument("--model", default=MODEL_PATH, help="Stage 1 xmodel")
    p.add_argument("--gru", default=GRU_PATH, help="Stage 2 tflite")
    p.add_argument("--bit", default=BIT_PATH, help="DPU overlay bitstream")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX, help="CAM_A index")
    p.add_argument("--conf", type=float, default=CONF_THRES,
                   help="Stage 1 confidence. 학습 시퀀스 추출값 0.15 를 기본으로 둔다")
    p.add_argument("--window", type=float, default=SEQ_WINDOW_S,
                   help="30프레임이 덮는 실제 시간(초). 학습 데이터 촬영 fps 로 확정할 것")
    p.add_argument("--gru-every", type=int, default=GRU_EVERY,
                   help="N 프레임마다 GRU 1회 실행")
    p.add_argument("--gesture-thres", type=float, default=GESTURE_THRES)
    p.add_argument("--disp-min", type=float, default=DISP_MIN,
                   help="정지 판정 임계값(bbox 중심 이동 범위). idle 실측 최대 0.019")
    p.add_argument("--shape-min", type=float, default=SHAPE_MIN,
                   help="손모양 원-핫 변화량 임계값. 전이 1회당 2.0")
    p.add_argument("--no-gate", action="store_true",
                   help="정지 게이트를 끈다. 게이트 유무 비교용")
    p.add_argument("--gate-debug", action="store_true",
                   help="차단된 창의 disp/shape 를 출력한다")
    p.add_argument("--cooldown", type=float, default=COOLDOWN_S,
                   help="제스처 검출 후 예측을 멈추는 시간(초). 복귀동작 무시용")
    p.add_argument("--no-display", action="store_true",
                   help="헤드리스 실행 (SSH 등). DISPLAY 가 없으면 자동으로 켜진다")
    p.add_argument("--dump-seq", action="store_true",
                   help="특징 시퀀스를 npz 로 저장한다. 학습 분포와 대조용")
    p.add_argument("--dump-path", default="./live_seq.npz")
    p.add_argument("--dump-label", default="",
                   help="이 세션에서 의도한 제스처 이름 (예: left). 덤프에 함께 기록된다")
    p.add_argument("--dump-max", type=int, default=2000,
                   help="이 개수를 넘으면 자동 저장 후 종료")
    p.add_argument("--duration", type=float, default=0.0,
                   help="지정 초 후 자동 종료 (0 = 무제한)")
    return p.parse_args()


def main():
    args = parse_args()

    show = not args.no_display and bool(os.environ.get("DISPLAY"))
    if not show:
        print("[INFO] 헤드리스 모드. 검출 결과는 표준출력으로만 나온다.")

    print(f"[INFO] Script     : {os.path.abspath(__file__)}")
    print(f"[INFO] Stage1     : {args.model}")
    print(f"[INFO] Stage2     : {args.gru}")
    print(f"[INFO] conf={args.conf}  iou={IOU_THRES}  max_det={MAX_DETS}")
    print(f"[INFO] window={args.window}s  seq_len={SEQ_LEN}  gru_every={args.gru_every}")
    print(f"[INFO] cooldown={args.cooldown}s (발화 후 이 시간 동안 버퍼를 비우고 예측 정지)")
    if args.no_gate:
        print("[INFO] 정지 게이트 OFF — 손을 멈춰도 GRU 가 돌고 오검출이 난다")
    else:
        print(f"[INFO] 정지 게이트 ON  disp>{args.disp_min} or shape>{args.shape_min}")
    print(f"[INFO] 유효 샘플 간격 = {args.window / SEQ_LEN * 1000:.1f} ms "
          f"(카메라가 이보다 느리면 같은 프레임이 중복 샘플링된다)")

    rt = Runner(args.bit, args.model)
    print(f"[INFO] DPU input  : {rt.in_dims} layout={rt.layout} "
          f"int8={rt.in_is_int8} fix={rt.in_fix}")
    if (rt.in_h, rt.in_w) != (416, 416):
        print(f"[WARN] DPU 입력이 {rt.in_h}x{rt.in_w} 다. 학습은 416x416 이었다.")

    gru = GestureClassifier(args.gru)
    seqbuf = SequenceBuffer(window_s=args.window)
    dumper = SequenceDumper(args.dump_path, args.dump_label) if args.dump_seq else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev = time.time()
    t_start = time.time()
    frame_no = 0
    cooldown_until = 0.0
    latched = ("", 0.0)
    n_gated = 0          # 게이트가 막은 창 수
    n_gru = 0            # 실제로 GRU 를 돌린 창 수
    gru_ms = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            inp, ratio, pad, orig_wh = rt.preprocess(frame)

            t_dpu = time.perf_counter()
            outputs = rt.infer(inp)
            boxes, scores, cls_ids = decode(outputs, ratio, pad, orig_wh, args.conf)
            dpu_ms = (time.perf_counter() - t_dpu) * 1000

            # --- Stage 2 로 넘길 원시 검출 (xywhn: 원본 프레임 기준 정규화) ---
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
                seqbuf.buf.clear()      # 쿨다운 중엔 아무것도 쌓지 않는다. 끝난 뒤
                                        # window_s 만큼 새로 채워야 ready() 가 되므로
                                        # 복귀동작은 다음 창에 들어올 수 없다
            else:
                seqbuf.push(now, det)

            frame_no += 1
            if frame_no % args.gru_every == 0 and seqbuf.ready(now):
                seq = build_sequence(seqbuf.sample(now))
                if seq is not None:
                    active, disp, shape = is_active(seq, args.disp_min, args.shape_min)
                    if not args.no_gate and not active:
                        n_gated += 1
                        latched = ("", 0.0)          # 표시 지우기
                        if args.gate_debug:
                            print(f"[GATE] idle  disp={disp:.4f} shape={shape:.2f}  "
                                  f"t={now - t_start:6.1f}s")
                        continue

                    n_gru += 1
                    t_g = time.perf_counter()
                    prob = gru.predict(seq)
                    gru_ms = (time.perf_counter() - t_g) * 1000

                    k = int(np.argmax(prob))
                    fired = prob[k] >= args.gesture_thres
                    if fired:
                        cooldown_until = now + args.cooldown
                        latched = (GESTURE_CLASSES[k], float(prob[k]))
                        print(f"[GESTURE] {latched[0]:12s} p={latched[1]:.3f}  "
                              f"t={now - t_start:6.1f}s")

                    if dumper is not None:
                        dumper.add(seq, prob, now - t_start, fired)
                        if len(dumper) >= args.dump_max:
                            print("[DUMP] dump-max 도달. 종료한다.")
                            break

            if show:
                draw(frame, boxes, scores, cls_ids)
                if latched[0] and now < cooldown_until:
                    cv2.putText(frame, f"{latched[0]} {latched[1]:.2f}", (8, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                cv2.putText(frame, f"DPU {dpu_ms:.1f}ms  GRU {gru_ms:.1f}ms", (8, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                cv2.imshow("KV260 Gesture (2-stage)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            elif frame_no % 30 == 0:
                fps = 30.0 / max(now - prev, 1e-6)   # prev 는 30프레임 전 시각
                prev = now
                print(f"[STAT] fps={fps:5.1f}  DPU={dpu_ms:5.1f}ms  GRU={gru_ms:5.2f}ms  "
                      f"det={'Y' if det else 'N'}  buf={len(seqbuf.buf)}  "
                      f"gru={n_gru} gated={n_gated}")

            if args.duration > 0 and (now - t_start) >= args.duration:
                print("[INFO] duration 도달. 종료한다.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        if dumper is not None:
            dumper.save()


if __name__ == "__main__":
    main()