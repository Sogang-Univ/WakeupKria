#!/usr/bin/env python3
"""task_ai.xmodel 단독 실행 (쓰레기 분류 YOLOv8 9클래스 · 1단계뿐)

02 의 `task` 모드만 떼어낸 것이다. 웨이크 워드도, 제스처도, 상태 기계도, PTZ 크롭도
없다 — 카메라 한 대를 열어 DPU 에 넣고 박스를 그린다. 02 에서 `open` 이 안 될 때
"모델 문제인가 파이프라인 문제인가" 를 가르는 것이 이 파일의 용도다.

디코딩·전처리·그리기는 전부 gesture_live.py 것을 그대로 쓴다. 같은 래퍼로 export 한
같은 구조라 클래스 수와 검출 개수만 다르다.

    python3 task_live.py                      # ssh -X 로 창이 뜬다
    python3 task_live.py --no-display         # 헤드리스. 검출을 stdout 으로
    python3 task_live.py --conf 0.25          # 임계값 훑기
    python3 task_live.py --self-test          # head 채널 산술만. 하드웨어 불필요

끝낼 때 클래스별 검출 횟수를 찍는다. conf 를 얼마로 둘지는 그 표를 보고 정한다.
"""

import sys
import argparse
import collections
import os
import re
import time

SELF_TEST = "--self-test" in sys.argv        # 02 와 같은 장치. 근거는 NOTEBOOK_TO_PY.md §2

import numpy as np

if not SELF_TEST:
    import cv2
    from gesture_live import Runner, decode, draw, REG_MAX


# 순서 = tempModel/data.yaml 의 인덱스. 02_wake_gesture_switch 의 TASK_CLASSES 와 같아야 한다.
# cv2.putText 에 한글 폰트가 없어 로마자로 적는다
# (유리병류 · 도기류 · 형광등 · 전자제품 · 의류 · 비닐류 · 캔류 · 플라스틱류 · 종이류)
CLASSES = ["glass", "ceramic", "fluorescent", "electronics", "clothing",
           "vinyl", "can", "plastic", "paper"]

MODEL_PATH = "./task_ai.xmodel"
BIT_PATH = "dpu.bit"
CAMERA = "/dev/cam1"         # 02 에서 작업 모델이 보는 카메라. 숫자 인덱스도 받는다
WIDTH, HEIGHT, FOURCC = 640, 480, "MJPG"
CONF = 0.35                  # 02 의 TASK_CONF 와 같은 기본값
MAX_DETS = 10


def head_channels(outputs):
    """세 head 출력이 공유하는 차원이 채널 수다. h·w 는 스케일마다 다르다.

    레이아웃(NHWC/NCHW)도 입력 해상도도 몰라도 된다는 것이 요점이다. 여기서 나온 c 로
    `클래스 수 = c - 4*REG_MAX` 를 역산해 CLASSES 와 대조한다 — 안 맞으면 decode() 가
    "reg_max mismatch" 로 죽는데, 그때는 몇을 줘야 하는지 알 수 없다. 미리 알려준다.
    """
    common = set.intersection(*[set(o.shape[1:]) for o in outputs])
    return max(common)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--bit", default=BIT_PATH)
    p.add_argument("--camera", default=CAMERA,
                   help="숫자 인덱스 또는 /dev/cam1 같은 경로 (심볼릭 링크를 푼다)")
    p.add_argument("--conf", type=float, default=CONF,
                   help="이 값을 훑는 것이 이 스크립트의 주 용도다")
    p.add_argument("--max-det", type=int, default=MAX_DETS)
    p.add_argument("--no-display", action="store_true",
                   help="헤드리스 실행 (SSH 등). DISPLAY 가 없으면 자동으로 켜진다")
    p.add_argument("--duration", type=float, default=0.0, help="지정 초 후 종료 (0 = 무제한)")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def cam_index(spec):
    """숫자면 그대로. 경로면 심볼릭 링크를 풀어 /dev/videoN 의 N 을 뽑는다."""
    if str(spec).isdigit():
        return int(spec)
    real = os.path.realpath(spec)
    m = re.search(r"(\d+)$", real)
    if not m:
        raise ValueError(f"video 인덱스를 찾을 수 없다: {spec} -> {real}")
    return int(m.group(1))


def self_test():
    """head_channels() 만 본다. 카메라도 DPU 도 필요 없다."""
    REG = 16
    nc, (h1, h2, h3) = len(CLASSES), (52, 26, 13)          # 416 입력의 세 스케일
    c = 4 * REG + nc                                        # = 73

    nchw = [np.zeros((1, c, s, s), np.float32) for s in (h1, h2, h3)]
    nhwc = [np.zeros((1, s, s, c), np.float32) for s in (h1, h2, h3)]
    assert head_channels(nchw) == c                         # 1) 두 레이아웃 다 같은 답
    assert head_channels(nhwc) == c

    big = [np.zeros((1, s, s, c), np.float32) for s in (80, 40, 20)]
    assert head_channels(big) == c                          # 2) 640 입력 — h 가 c 보다 커도 된다

    assert head_channels(nhwc) - 4 * REG == nc              # 3) 클래스 수 역산
    assert cam_index("3") == 3 and cam_index("/dev/video7") == 7   # 4) 카메라 지정 두 방식
    print(f"head 채널 OK — c={c} → 클래스 {nc}개 · CLASSES {len(CLASSES)}개")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    show = not args.no_display and bool(os.environ.get("DISPLAY"))
    print(f"[INFO] model={args.model}  conf={args.conf}  max_det={args.max_det}")
    print(f"[INFO] classes({len(CLASSES)}) = {CLASSES}")
    if not show:
        print("[INFO] 헤드리스 모드. 검출은 표준출력으로만 나온다.")

    rt = Runner(args.bit, args.model)
    print(f"[INFO] DPU input : {rt.in_dims} layout={rt.layout} "
          f"int8={rt.in_is_int8} fix={rt.in_fix}")

    dev = cam_index(args.camera)
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패: {args.camera} -> /dev/video{dev}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[INFO] camera   : {args.camera} -> /dev/video{dev}")

    hits = collections.Counter()
    ncls = len(CLASSES)
    checked = False
    t_start = prev = time.time()
    frames = 0
    dpu_ms = 0.0
    dpu_total = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            inp, ratio, pad, orig_wh = rt.preprocess(frame)
            t0 = time.perf_counter()
            outputs = rt.infer(inp)

            if not checked:      # 첫 프레임에서 모델이 말하는 클래스 수를 확인한다
                checked = True
                model_nc = head_channels(outputs) - 4 * REG_MAX
                print(f"[INFO] head 채널에서 역산한 클래스 수 = {model_nc}")
                if model_nc != ncls:
                    print(f"[WARN] CLASSES 는 {ncls}개다. 모델을 따라 {model_nc}개로 디코딩한다 "
                          f"— 이름 없는 인덱스는 숫자로 그려진다. CLASSES 를 고칠 것")
                    ncls = model_nc

            boxes, scores, cls_ids = decode(outputs, ratio, pad, orig_wh,
                                            args.conf, ncls, args.max_det)
            dpu_ms = (time.perf_counter() - t0) * 1000
            dpu_total += dpu_ms
            frames += 1

            now = time.time()
            for cid, s in zip(cls_ids, scores):
                name = CLASSES[cid] if cid < len(CLASSES) else str(cid)
                hits[name] += 1
                if not show:
                    print(f"[DET ] {name:12s} p={s:.3f}  t={now - t_start:6.1f}s")

            if show:
                draw(frame, boxes, scores, cls_ids, CLASSES)
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                cv2.putText(frame, f"FPS {fps:.1f}  DPU {dpu_ms:.1f}ms  det {len(boxes)}",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                cv2.imshow("task_ai (9-class)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            elif frames % 30 == 0:
                fps = 30.0 / max(now - prev, 1e-6)       # prev 는 30프레임 전 시각
                prev = now
                print(f"[STAT] fps={fps:5.1f}  DPU={dpu_ms:5.1f}ms  det={len(boxes)}")

            if args.duration > 0 and (now - t_start) >= args.duration:
                print("[INFO] duration 도달. 종료한다.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

        # conf 를 어디에 둘지는 이 표를 보고 정한다. 한 클래스만 쏟아지면 그 임계값이 낮다.
        elapsed = max(time.time() - t_start, 1e-6)
        print(f"\n프레임 {frames}개 · {elapsed:.1f}s · "
              f"DPU 평균 {dpu_total / max(frames, 1):.1f}ms · conf={args.conf}")
        if hits:
            print("클래스별 검출 횟수")
            for name, n in hits.most_common():
                print(f"  {name:12s} {n:5d}  ({n / frames:.2f}/프레임)")
        else:
            print("검출 0건 — --conf 를 낮추거나(0.25) 카메라 거리·조명을 본다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
