#!/usr/bin/env python3
"""Runner.preprocess() 최적화판 — 원본과 **비트 단위 동일**.

원본(gesture_live.py L410-426)은 416x416x3 = 519,168 원소에 대해
float32 변환 -> /255 -> *2^fix -> np.round -> clip -> int8 변환을 돈다.
입력이 uint8 이라 가능한 값은 **256가지뿐**이므로 전체가 256엔트리 LUT 한 장으로
압축된다. LUT 를 원본과 같은 np.round(half-to-even)로 만들기 때문에 결과가 정확히 같다.

거기에 더해:
  - `copyMakeBorder` 의 프레임마다 할당을 캔버스 재사용으로 대체
  - 테두리 채우기를 **기하가 바뀔 때만** 수행 (02 의 task 모드는 줌 배율이 바뀔 때만 변한다)
  - `cvtColor`/`LUT` 를 dst 인자로 받아 중간 배열 할당과 마지막 복사를 제거

A53 은 x86 과 비용 구조가 달라(실측 배속이 예측의 절반) 어느 변형이 빠른지 보드마다
다르다. 그래서 변형 4개를 넣고 `--tune` 으로 **보드에서 직접 재서 고르게** 했다.

    python3 preprocess_fast.py            # 등가성 검증 + 기본 변형 벤치
    python3 preprocess_fast.py --tune     # 변형 A~D + 스레드 수까지 훑어 최적 선택

    from preprocess_fast import attach
    rt = Runner(BIT_PATH, GESTURE_XMODEL)
    attach(rt)                            # 이후 rt.preprocess() 가 최적화판
"""
import numpy as np
import cv2


def build_lut(fix):
    """원본의 clip(round(v/255 * 2**fix), -128, 127) 를 256엔트리로 미리 계산."""
    v = np.arange(256, dtype=np.float32)
    return np.clip(np.round(v / 255.0 * float(2 ** fix)), -128, 127).astype(np.int8)


def letterbox_geom(oh, ow, nh, nw):
    """원본 letterbox() 와 같은 반올림 규칙. -> (r, rw, rh, left, top)"""
    r = min(nh / oh, nw / ow)
    rw, rh = int(round(ow * r)), int(round(oh * r))
    dw, dh = nw - rw, nh - rh
    return r, rw, rh, int(round(dw / 2 - 0.1)), int(round(dh / 2 - 0.1))


class FastPreproc:
    """Runner 하나에 하나. in_h/in_w/in_fix/layout 이 바뀌면 새로 만든다.

    variant:
        "B" (기본) cvtColor(dst) -> LUT(dst=out).   중간 할당·최종 복사 없음
        "A"        cvtColor(dst) -> LUT -> copy
        "C"        LUT(dst) -> 채널역순 복사        (cvtColor 제거)
        "D"        np.take 로 LUT+역순 융합         (1패스지만 strided)
        "E"        테두리를 출력(int8)에 미리 굽고 **안쪽 영역만** cvtColor+LUT.
                   416x416 중 실제 영상은 416x312 뿐이라 픽셀 작업이 25% 줄어든다.
    x86 에서는 E > B 지만 A53 은 다를 수 있다. --tune 으로 확인할 것.
    """

    VARIANTS = ("A", "B", "C", "D", "E")

    def __init__(self, in_h, in_w, in_fix, layout="NHWC", is_int8=True,
                 pad_value=114, variant="B"):
        self.in_h, self.in_w = int(in_h), int(in_w)
        self.layout, self.is_int8 = layout, bool(is_int8)
        self.in_fix = int(in_fix)
        self.pad_value = pad_value
        self.variant = variant

        self.lut = build_lut(self.in_fix)
        self.lut_u8 = self.lut.view(np.uint8)

        self._canvas = np.empty((self.in_h, self.in_w, 3), np.uint8)
        self._tmp = np.empty((self.in_h, self.in_w, 3), np.uint8)
        self._out = np.empty((1, self.in_h, self.in_w, 3), np.int8)
        self._out_u8 = self._out[0].view(np.uint8)
        self._geom = None

    # -- letterbox: 캔버스 재사용 + 테두리는 기하가 바뀔 때만 --------------
    def _letterbox(self, frame):
        oh, ow = frame.shape[:2]
        g = letterbox_geom(oh, ow, self.in_h, self.in_w)
        r, rw, rh, left, top = g
        if self._geom != g:
            self._canvas[:] = self.pad_value      # 기하가 바뀔 때만. 매 프레임 아님
            self._geom = g
        cv2.resize(frame, (rw, rh), dst=self._canvas[top:top + rh, left:left + rw],
                   interpolation=cv2.INTER_LINEAR)
        return r, (left, top), (ow, oh)

    def _run_e(self, frame):
        """테두리를 출력에 미리 굽고 안쪽만 처리. 416x416 -> 416x312 로 25% 절약."""
        oh, ow = frame.shape[:2]
        g = letterbox_geom(oh, ow, self.in_h, self.in_w)
        r, rw, rh, left, top = g
        if self._geom != g:
            self._out[0] = self.lut[self.pad_value]   # int8 테두리. 기하 바뀔 때만
            self._geom = g
        src = self._canvas[:rh, :rw]
        cv2.resize(frame, (rw, rh), dst=src, interpolation=cv2.INTER_LINEAR)
        tv = self._tmp[:rh, :rw]
        cv2.cvtColor(src, cv2.COLOR_BGR2RGB, dst=tv)
        dst = self._out_u8[top:top + rh, left:left + rw]
        if dst.flags["C_CONTIGUOUS"]:                 # 4:3 입력이면 left==0 이라 연속
            cv2.LUT(tv, self.lut_u8, dst=dst)
        else:
            dst[:] = cv2.LUT(tv, self.lut_u8)
        return self._out, r, (left, top), (ow, oh)

    def __call__(self, frame):
        if self.variant == "E" and self.is_int8 and self.layout == "NHWC":
            return self._run_e(frame)
        r, pad, wh = self._letterbox(frame)
        c = self._canvas

        if not self.is_int8:                      # float 모델이면 원래 경로
            img = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if self.layout == "NCHW":
                img = np.transpose(img, (2, 0, 1))
            return np.expand_dims(img, 0), r, pad, wh

        if self.layout == "NCHW":                 # 드문 경로. 정확성만 맞춘다
            q = cv2.LUT(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), self.lut_u8).view(np.int8)
            return np.ascontiguousarray(q.transpose(2, 0, 1))[None], r, pad, wh

        v = self.variant
        if v in ("B", "E"):
            cv2.cvtColor(c, cv2.COLOR_BGR2RGB, dst=self._tmp)
            cv2.LUT(self._tmp, self.lut_u8, dst=self._out_u8)
        elif v == "A":
            cv2.cvtColor(c, cv2.COLOR_BGR2RGB, dst=self._tmp)
            self._out[0] = cv2.LUT(self._tmp, self.lut_u8).view(np.int8)
        elif v == "C":
            cv2.LUT(c, self.lut_u8, dst=self._tmp)
            self._out[0] = self._tmp.view(np.int8)[:, :, ::-1]
        else:                                     # "D"
            np.take(self.lut, c[:, :, ::-1], out=self._out[0])
        return self._out, r, pad, wh


def attach(runner, pad_value=114, variant="B"):
    """기존 Runner 인스턴스의 preprocess 만 교체한다. gesture_live.py 는 안 건드린다."""
    fp = FastPreproc(runner.in_h, runner.in_w, runner.in_fix,
                     runner.layout, runner.in_is_int8, pad_value, variant)
    runner._preproc_fast = fp
    runner.preprocess = fp
    return runner


# ---------------------------------------------------------------- 검증 · 튜닝
def _pre_orig(frame, nh, nw, fix, layout="NHWC"):
    """gesture_live.letterbox + Runner.preprocess 를 그대로 옮긴 것 (비교 기준)."""
    oh, ow = frame.shape[:2]
    r = min(nh / oh, nw / ow)
    rw, rh = int(round(ow * r)), int(round(oh * r))
    res = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    dw, dh = nw - rw, nh - rh
    l = int(round(dw / 2 - 0.1)); rr = int(round(dw / 2 + 0.1))
    t = int(round(dh / 2 - 0.1)); b = int(round(dh / 2 + 0.1))
    img = cv2.copyMakeBorder(res, t, b, l, rr, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if layout == "NCHW":
        img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return (np.clip(np.round(img * float(2 ** fix)), -128, 127).astype(np.int8),
            r, (l, t), (ow, oh))


if __name__ == "__main__":
    import sys
    import time

    TUNE = "--tune" in sys.argv
    rng = np.random.default_rng(0)

    def bench(fn, n=60):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000

    # 02 가 실제로 넣는 크기: cam0 원본 640x480, cam1 은 줌 크롭이라 가변
    SHAPES = [("cam0 480x640", (480, 640)),
              ("task 줌x2 240x320", (240, 320)),
              ("task 줌x4 120x160", (120, 160))]
    FIX = 6

    print(f"cv2 {cv2.__version__} · threads={cv2.getNumThreads()} "
          f"(코어 {cv2.getNumberOfCPUs()})\n")

    print("=== 등가성 (원본과 비트 단위 동일한가) ===")
    bad = 0
    for _ in range(60):
        H = int(rng.integers(120, 720)); W = int(rng.integers(160, 960))
        fix = int(rng.integers(3, 8))
        f = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        ref = _pre_orig(f, 416, 416, fix)
        for v in FastPreproc.VARIANTS:
            got = FastPreproc(416, 416, fix, variant=v)(f)
            if not (np.array_equal(got[0], ref[0]) and got[2] == ref[2]
                    and got[3] == ref[3]):
                bad += 1
    print(f"  랜덤 60회 x 변형 4개 : {'전부 일치' if bad == 0 else f'{bad}건 불일치'}\n")

    variants = FastPreproc.VARIANTS if TUNE else ("B",)
    thread_opts = ([cv2.getNumThreads(), 1, cv2.getNumberOfCPUs()] if TUNE
                   else [cv2.getNumThreads()])
    thread_opts = sorted(set(int(t) for t in thread_opts if t >= 1))

    best = None
    for nt in thread_opts:
        cv2.setNumThreads(nt)
        print(f"=== cv2 threads = {nt} ===")
        hdr = f"{'변형':8s}" + "".join(f"{n:>20s}" for n, _ in SHAPES)
        print(hdr)
        f0 = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        to = bench(lambda: _pre_orig(f0, 416, 416, FIX))
        print(f"{'원본':8s}{to:18.2f}ms" + " " * (20 * (len(SHAPES) - 1)))
        for v in variants:
            ts = []
            for _, (H, W) in SHAPES:
                fr = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
                fp = FastPreproc(416, 416, FIX, variant=v)
                ts.append(bench(lambda fp=fp, fr=fr: fp(fr)))
            print(f"{v:8s}" + "".join(f"{t:18.2f}ms" for t in ts)
                  + f"   (cam0 {to/ts[0]:.1f}x)")
            if best is None or ts[0] < best[0]:
                best = (ts[0], v, nt)

    print(f"\n권장: variant=\"{best[1]}\", cv2.setNumThreads({best[2]})  "
          f"-> cam0 {best[0]:.2f} ms")
    if not TUNE:
        print("--tune 을 붙이면 변형 A~D 와 스레드 수까지 훑는다")
    print("\n적용:  attach(rt, variant=\"%s\")" % best[1])
