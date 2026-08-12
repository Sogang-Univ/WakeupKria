"""decode() 최적화판 — gesture_live.decode() 드롭인 대체.

원본과 **수학적으로 동일**하다. 바꾼 것은 계산 순서뿐이다.

원본은 3,549개 앵커 **전부**에 DFL softmax(4×16 bin)를 돌린 뒤 임계값을 건다.
실제로 임계값을 넘는 앵커는 보통 한 자릿수다. 227,136개 float 중 99% 이상이 버려진다.

이 판은 순서를 뒤집는다:
  1. 클래스 로짓에서 바로 임계값 판정  (sigmoid 는 단조라 sigmoid(x)>c <=> x>log(c/(1-c)))
  2. 살아남은 K개 앵커에만 DFL softmax
  3. NHWC 를 그대로 쓴다 — 원본의 transpose 는 비연속 배열을 만들어 뒤의 reshape 가
     227k 원소를 통째로 복사하게 만든다
"""
import numpy as np
import cv2

STRIDES = [8, 16, 32]
REG_MAX = 16
IOU_THRES = 0.45
_BINS = np.arange(REG_MAX, dtype=np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _clip(b, w, h):
    b[:, 0] = np.clip(b[:, 0], 0, w - 1); b[:, 1] = np.clip(b[:, 1], 0, h - 1)
    b[:, 2] = np.clip(b[:, 2], 0, w - 1); b[:, 3] = np.clip(b[:, 3], 0, h - 1)
    return b


def _nms(boxes, scores, iou):
    if len(boxes) == 0:
        return np.array([], np.int32)
    idx = cv2.dnn.NMSBoxes(
        bboxes=[[float(a), float(b), float(c - a), float(d - b)]
                for a, b, c, d in boxes.astype(np.float32)],
        scores=scores.astype(np.float32).tolist(),
        score_threshold=0.0, nms_threshold=iou)
    return np.array(idx).reshape(-1) if len(idx) else np.array([], np.int32)


def _as_hwc(out, nc):
    """(1,a,b,c) -> (h*w, C) 2D 뷰와 (h, w). 복사 없이 처리하는 것이 요점."""
    if out.ndim != 4 or out.shape[0] != 1:
        raise RuntimeError(f"Unexpected output shape: {list(out.shape)}")
    _, a, b, c = out.shape
    ok = lambda ch: ch > nc and (ch - nc) % 4 == 0
    if ok(c) and c >= a and c >= b:                 # NHWC
        return out.reshape(a * b, c), a, b, c
    if ok(a):                                       # NCHW -> (C, h*w) 를 전치 없이 다룬다
        return None, b, c, a
    if ok(c):
        return out.reshape(a * b, c), a, b, c
    raise RuntimeError(f"Cannot infer layout from {list(out.shape)}")


def decode(outputs, ratio, pad, orig_wh, conf_thres,
           num_classes=4, max_dets=1):
    nc = num_classes
    lt = (-np.inf if conf_thres <= 0 else
          (np.inf if conf_thres >= 1 else float(np.log(conf_thres / (1 - conf_thres)))))

    parsed = []
    for o in outputs:
        _, a, b, c = o.shape
        ok = lambda ch: ch > nc and (ch - nc) % 4 == 0
        if ok(c) and c >= a and c >= b:
            parsed.append(("nhwc", o.reshape(a * b, c), a, b, c))
        elif ok(a):
            parsed.append(("nchw", o.reshape(a, b * c), b, c, a))
        elif ok(c):
            parsed.append(("nhwc", o.reshape(a * b, c), a, b, c))
        else:
            raise RuntimeError(f"Cannot infer layout from {list(o.shape)}")
    parsed.sort(key=lambda p: p[2] * p[3], reverse=True)

    AB, AS, AI = [], [], []
    for (layout, flat, h, w, C), stride in zip(parsed, STRIDES):
        reg_max = (C - nc) // 4
        if reg_max != REG_MAX:
            raise RuntimeError(f"reg_max mismatch: expected={REG_MAX}, got={reg_max}")

        # --- 1) 클래스 로짓만 보고 후보를 고른다 (sigmoid/exp 없이) ---
        if layout == "nhwc":
            cls = flat[:, 4 * REG_MAX:4 * REG_MAX + nc]        # (hw, nc)
            ids = np.argmax(cls, 1)
            best = cls[np.arange(cls.shape[0]), ids]
        else:
            cls = flat[4 * REG_MAX:4 * REG_MAX + nc]           # (nc, hw)
            ids = np.argmax(cls, 0)
            best = cls[ids, np.arange(cls.shape[1])]
        sel = np.nonzero(best > lt)[0]
        if sel.size == 0:
            continue

        # --- 2) 살아남은 K개에만 DFL ---
        if layout == "nhwc":
            dist = flat[sel, :4 * REG_MAX].reshape(-1, 4, REG_MAX)      # (K,4,16)
            dp = _softmax(dist, axis=2)
            ltrb = dp @ _BINS                                            # (K,4)
        else:
            dist = flat[:4 * REG_MAX, sel].reshape(4, REG_MAX, -1)       # (4,16,K)
            dp = _softmax(dist, axis=1)
            ltrb = np.tensordot(_BINS, dp, axes=([0], [1])).T            # (K,4)

        cx = (sel % w).astype(np.float32) + 0.5
        cy = (sel // w).astype(np.float32) + 0.5
        AB.append(np.stack([(cx - ltrb[:, 0]) * stride, (cy - ltrb[:, 1]) * stride,
                            (cx + ltrb[:, 2]) * stride, (cy + ltrb[:, 3]) * stride], 1))
        AS.append(_sigmoid(best[sel]))
        AI.append(ids[sel])

    if not AB:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.int32))

    boxes = np.concatenate(AB, 0).astype(np.float32)
    scores = np.concatenate(AS, 0).astype(np.float32)
    cls_ids = np.concatenate(AI, 0).astype(np.int32)

    k = _nms(boxes, scores, IOU_THRES)
    boxes, scores, cls_ids = boxes[k], scores[k], cls_ids[k]

    px, py = pad
    ow, oh = orig_wh
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / ratio
    boxes = _clip(boxes, ow, oh)

    if len(scores) > max_dets:
        o = np.argsort(scores)[::-1][:max_dets]
        boxes, scores, cls_ids = boxes[o], scores[o], cls_ids[o]
    return boxes, scores, cls_ids


# ---------------------------------------------------------------- 자체 점검
def _synth(nc, bias, seed=0, hit=6.0):
    rng = np.random.default_rng(seed)
    o = []
    for s in (52, 26, 13):
        a = rng.standard_normal((1, s, s, 4 * REG_MAX + nc)).astype(np.float32) * 1.5
        a[..., 4 * REG_MAX:] += bias
        o.append(a)
    o[0][0, 10, 12, 4 * REG_MAX] = hit
    return o


def n_survivors(outputs, num_classes, conf_thres):
    """임계값을 넘는 앵커 수 K. 최적화 효과가 K 에 반비례하므로 실측해 둘 것."""
    lt = float(np.log(conf_thres / (1 - conf_thres)))
    n = 0
    for o in outputs:
        c = o[0].reshape(-1, o.shape[-1])[:, 4 * REG_MAX:4 * REG_MAX + num_classes]
        n += int((c.max(1) > lt).sum())
    return n


if __name__ == "__main__":
    import time
    try:
        from gesture_live import decode as decode_ref
        print("[INFO] gesture_live.decode 와 대조한다")
    except Exception as e:
        print(f"[WARN] gesture_live 를 못 읽었다 ({e}). 속도만 잰다")
        decode_ref = None

    def bench(fn, outs, conf, nc, md, n=40):
        fn(outs, 0.65, (0, 56), (640, 480), conf, nc, md)
        t0 = time.perf_counter()
        for _ in range(n):
            fn(outs, 0.65, (0, 56), (640, 480), conf, nc, md)
        return (time.perf_counter() - t0) / n * 1000

    print(f"\n앵커 {52*52+26*26+13*13}개 · K = 임계값 통과 앵커 수\n")
    print(f"{'설정':24s} {'K':>6s} {'원본':>9s} {'최적화':>9s} {'배속':>7s} {'동일':>6s}")
    for nc, md, conf, tag in [(4, 1, 0.15, "제스처 4cls conf=.15"),
                              (9, 10, 0.35, "쓰레기 9cls conf=.35")]:
        for bias in (-12, -9, -7, -4):
            outs = _synth(nc, bias)
            K = n_survivors(outs, nc, conf)
            tf = bench(decode, outs, conf, nc, md)
            if decode_ref is not None:
                a = decode_ref(outs, 0.65, (0, 56), (640, 480), conf, nc, md)
                b = decode(outs, 0.65, (0, 56), (640, 480), conf, nc, md)
                same = (a[0].shape == b[0].shape and np.allclose(a[0], b[0], atol=1e-3)
                        and np.allclose(a[1], b[1], atol=1e-5)
                        and np.array_equal(a[2], b[2]))
                to = bench(decode_ref, outs, conf, nc, md)
                print(f"{tag:24s} {K:6d} {to:7.2f}ms {tf:7.2f}ms "
                      f"{to/tf:6.1f}x {str(same):>6s}")
            else:
                print(f"{tag:24s} {K:6d} {'-':>9s} {tf:7.2f}ms {'-':>7s} {'-':>6s}")
        print()
    print("K 가 수십 이하면 10x 이상, 수백이면 2x 수준이다.")
    print("보드에서 n_survivors() 로 실제 K 를 재 볼 것 — 모델이 제대로 학습됐으면 작다.")
