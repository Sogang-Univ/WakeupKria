import argparse
import collections
import copy
import glob
import os
import random
import subprocess
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pytorch_nndct.apis import torch_quantizer
from ultralytics import YOLO
import ultralytics.nn.modules.block as ublock


VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")

DEFAULT_MODEL_CANDIDATES = (
    "./best_gesture_stage1.pt",
    "./content/gesture_runs/gesture_stage1/weights/best.pt",
    "./best.pt",
)

# 노트북이 만든 YOLO 포맷 데이터셋. labels/ 가 함께 있으면 클래스 균등 샘플링이 켜진다.
DEFAULT_CALIB_DIR_CANDIDATES = (
    "./Gesture_YOLO/images/train",
    "./Gesture_YOLO/images",
    "./dataset/images",
)

# data.yaml 의 names 와 순서까지 일치해야 한다
EXPECTED_CLASS_NAMES = ["유리병류", "도기류", "형광등", "전자제품", "의류", "비닐류", "캔류", "플라스틱류", "종이류"]

DEFAULT_IMG_SIZE = 416

DEFAULT_ARCH_JSON_CANDIDATES = (
    "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json",
    "/opt/vitis_ai/compiler/arch/DPUCV2DX8G/KV260/arch.json",
)


def _label_path_for(img_path: str) -> str:
    sep = os.sep
    if f"{sep}images{sep}" not in img_path:
        return ""
    lp = img_path.replace(f"{sep}images{sep}", f"{sep}labels{sep}")
    return os.path.splitext(lp)[0] + ".txt"


def _class_of(img_path: str) -> int:
    """YOLO 라벨 첫 줄의 클래스 id. 라벨이 없으면 -1."""
    lp = _label_path_for(img_path)
    if not lp or not os.path.exists(lp):
        return -1
    try:
        with open(lp) as f:
            line = f.readline().strip()
        return int(line.split()[0]) if line else -1
    except Exception:
        return -1


class CalibrationDataset(Dataset):
    def __init__(self, img_dir: str, img_size: int = DEFAULT_IMG_SIZE,
                 max_images: int = 200, seed: int = 42):
        self.img_size = img_size

        all_files = []
        for ext in VALID_EXTS:
            all_files.extend(glob.glob(os.path.join(img_dir, "**", f"*{ext}"), recursive=True))

        all_files = sorted(set(all_files))
        if not all_files:
            raise FileNotFoundError(f"No calibration images found under: {img_dir}")

        # 정렬 순서대로 자르면 Subject1 의 앞쪽 제스처만 뽑혀 특정 클래스가 통째로 빠진다.
        # INT8 활성 범위가 그 클래스에서만 엉터리가 되므로 클래스 균등 샘플링을 쓴다.
        self.img_paths = self._select(all_files, max_images, seed)

        hist = collections.Counter(_class_of(p) for p in self.img_paths)
        if set(hist) == {-1}:
            print(f"[WARN] 라벨을 찾지 못해 무작위 샘플링으로 대체했다 ({len(self.img_paths)}장)")
        else:
            named = {
                (EXPECTED_CLASS_NAMES[k] if 0 <= k < len(EXPECTED_CLASS_NAMES) else str(k)): v
                for k, v in sorted(hist.items())
            }
            print(f"[INFO] 캘리브레이션 클래스 분포: {named}")
            missing = [n for n in EXPECTED_CLASS_NAMES if n not in named]
            if missing:
                print(f"[WARN] 캘리브레이션에 빠진 클래스: {missing}")

    @staticmethod
    def _select(files: List[str], max_images: int, seed: int) -> List[str]:
        rng = random.Random(seed)
        buckets: Dict[int, List[str]] = {}
        for p in files:
            buckets.setdefault(_class_of(p), []).append(p)
        for v in buckets.values():
            rng.shuffle(v)

        if len(buckets) <= 1:                      # 라벨 없음 -> 단순 무작위
            pool = list(files)
            rng.shuffle(pool)
            return sorted(pool[:max_images])

        picked, keys, i = [], sorted(buckets), 0   # 클래스 라운드로빈
        while len(picked) < max_images and any(buckets[k] for k in keys):
            k = keys[i % len(keys)]
            if buckets[k]:
                picked.append(buckets[k].pop())
            i += 1
        return sorted(picked)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        img = letterbox(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)
        return tensor


def letterbox(img: np.ndarray, new_shape=DEFAULT_IMG_SIZE, color=(114, 114, 114)) -> np.ndarray:
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    shape = img.shape[:2]  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NndctFriendlyC2f(nn.Module):
    """
    C2f compatibility wrapper for old NNDCT versions.
    Avoids tuple-returning chunk() and avoids slice ops by using two explicit Conv branches.
    Function is equivalent to splitting cv1 output channels in half.
    """

    def __init__(self, c2f_orig: nn.Module):
        super().__init__()
        self.cv2 = c2f_orig.cv2
        self.m = c2f_orig.m
        self.c = c2f_orig.c
        self.cv1_a, self.cv1_b = self._build_split_cv1_branches(c2f_orig.cv1, self.c)
        if hasattr(c2f_orig, "f"):
            self.f = c2f_orig.f
        if hasattr(c2f_orig, "i"):
            self.i = c2f_orig.i

    @staticmethod
    def _copy_bn_slice(src_bn: nn.BatchNorm2d, dst_bn: nn.BatchNorm2d, start: int, end: int):
        with torch.no_grad():
            dst_bn.weight.copy_(src_bn.weight[start:end])
            dst_bn.bias.copy_(src_bn.bias[start:end])
            dst_bn.running_mean.copy_(src_bn.running_mean[start:end])
            dst_bn.running_var.copy_(src_bn.running_var[start:end])
            if hasattr(src_bn, "num_batches_tracked") and hasattr(dst_bn, "num_batches_tracked"):
                dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)

    @classmethod
    def _build_split_cv1_branches(cls, cv1_module: nn.Module, c: int):
        src_conv = cv1_module.conv
        src_bn = cv1_module.bn
        src_act = cv1_module.act

        k_h, k_w = src_conv.kernel_size
        s_h, s_w = src_conv.stride
        p_h, p_w = src_conv.padding
        d_h, d_w = src_conv.dilation
        groups = src_conv.groups
        use_bias = src_conv.bias is not None

        conv_a = nn.Conv2d(
            in_channels=src_conv.in_channels,
            out_channels=c,
            kernel_size=(k_h, k_w),
            stride=(s_h, s_w),
            padding=(p_h, p_w),
            dilation=(d_h, d_w),
            groups=groups,
            bias=use_bias,
        )
        conv_b = nn.Conv2d(
            in_channels=src_conv.in_channels,
            out_channels=c,
            kernel_size=(k_h, k_w),
            stride=(s_h, s_w),
            padding=(p_h, p_w),
            dilation=(d_h, d_w),
            groups=groups,
            bias=use_bias,
        )

        bn_a = nn.BatchNorm2d(c, eps=src_bn.eps, momentum=src_bn.momentum, affine=True, track_running_stats=True)
        bn_b = nn.BatchNorm2d(c, eps=src_bn.eps, momentum=src_bn.momentum, affine=True, track_running_stats=True)

        with torch.no_grad():
            conv_a.weight.copy_(src_conv.weight[:c, ...])
            conv_b.weight.copy_(src_conv.weight[c : 2 * c, ...])
            if use_bias:
                conv_a.bias.copy_(src_conv.bias[:c])
                conv_b.bias.copy_(src_conv.bias[c : 2 * c])

        cls._copy_bn_slice(src_bn, bn_a, 0, c)
        cls._copy_bn_slice(src_bn, bn_b, c, 2 * c)

        act_a = copy.deepcopy(src_act)
        act_b = copy.deepcopy(src_act)
        return nn.Sequential(conv_a, bn_a, act_a), nn.Sequential(conv_b, bn_b, act_b)

    def forward(self, x):
        y = [self.cv1_a(x), self.cv1_b(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def replace_c2f_for_nndct(module: nn.Module) -> int:
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, ublock.C2f):
            setattr(module, name, NndctFriendlyC2f(child))
            replaced += 1
        else:
            replaced += replace_c2f_for_nndct(child)
    return replaced


def replace_silu_with_relu(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_silu_with_relu(child)


def replace_silu_with_hardswish(module: nn.Module):
    # NOTE: On some Vitis-AI versions, py_nndct.nn.Hardswish raises NameError
    # (FixNeuronWithBackward missing). Use a decomposed SiLU form as a robust fallback.
    class SiLUDecomposed(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, SiLUDecomposed())
        else:
            replace_silu_with_hardswish(child)


def replace_silu_with_leakyrelu(module: nn.Module, negative_slope: float = 0.01):
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(negative_slope=float(negative_slope), inplace=False))
        else:
            replace_silu_with_leakyrelu(child, negative_slope=negative_slope)


def force_replace_conv_acts(module: nn.Module):
    """
    Some Ultralytics blocks keep activation as module.act.
    Force-replace SiLU-like acts to ReLU to avoid aten::silu_ in old toolchains.
    """
    for m in module.modules():
        if hasattr(m, "act"):
            act = getattr(m, "act")
            if isinstance(act, nn.SiLU):
                setattr(m, "act", nn.ReLU(inplace=False))
            elif act is True:
                # Defensive fallback for blocks that toggle default activation.
                setattr(m, "act", nn.ReLU(inplace=False))


def force_replace_conv_acts_hardswish(module: nn.Module):
    class SiLUDecomposed(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    for m in module.modules():
        if hasattr(m, "act"):
            act = getattr(m, "act")
            if isinstance(act, nn.SiLU):
                setattr(m, "act", SiLUDecomposed())
            elif act is True:
                setattr(m, "act", SiLUDecomposed())


def force_replace_conv_acts_leakyrelu(module: nn.Module, negative_slope: float = 0.01):
    for m in module.modules():
        if hasattr(m, "act"):
            act = getattr(m, "act")
            if isinstance(act, nn.SiLU):
                setattr(m, "act", nn.LeakyReLU(negative_slope=float(negative_slope), inplace=False))
            elif act is True:
                setattr(m, "act", nn.LeakyReLU(negative_slope=float(negative_slope), inplace=False))


def count_silu_modules(module: nn.Module) -> int:
    count = 0
    for m in module.modules():
        if isinstance(m, nn.SiLU):
            count += 1
    return count


def trace_contains_silu_op(module: nn.Module, img_size: int) -> bool:
    """Best-effort check: detect aten::silu ops in traced graph text."""
    try:
        module = module.eval()
        dummy = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            traced = torch.jit.trace(module, dummy, strict=False)
        graph_text = str(traced.inlined_graph)
        return ("aten::silu" in graph_text) or ("aten::silu_" in graph_text)
    except Exception as exc:
        print(f"[WARN] Trace silu-op check skipped: {exc}")
        return False


def fix_conv_bias_for_nndct(module: nn.Module):
    for child in module.modules():
        if isinstance(child, nn.Conv2d) and child.bias is None:
            child.bias = nn.Parameter(torch.zeros(child.out_channels, device=child.weight.device))


def _as_tuple_outputs(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "absmax": float(x.abs().max().item()),
    }


def _compare_tensor_pair(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    mae = float(torch.mean(torch.abs(a - b)).item())
    maxae = float(torch.max(torch.abs(a - b)).item())
    denom = torch.norm(a) * torch.norm(b) + 1e-12
    cos = float((torch.dot(a, b) / denom).item())
    return {"mae": mae, "maxae": maxae, "cosine": cos}


def _sort_head_outputs_by_scale(xs):
    return sorted(_as_tuple_outputs(xs), key=lambda t: int(t.shape[-2]) * int(t.shape[-1]), reverse=True)


def _find_detect_module(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "cv2") and hasattr(m, "cv3") and hasattr(m, "nl"):
            return m
    return None


def infer_head_meta(detect_module: nn.Module, example_out: torch.Tensor):
    ch = int(example_out.shape[1])
    nc = int(getattr(detect_module, "nc", 0)) if detect_module is not None else 0
    reg_max = int(getattr(detect_module, "reg_max", 0)) if detect_module is not None else 0

    if reg_max <= 0 and nc > 0 and ch > nc and (ch - nc) % 4 == 0:
        reg_max = (ch - nc) // 4
    if nc <= 0 and reg_max > 0 and ch > 4 * reg_max:
        nc = ch - 4 * reg_max
    if reg_max <= 0 or nc <= 0 or (4 * reg_max + nc) != ch:
        raise RuntimeError(
            f"Failed to infer head meta from channels={ch}, nc={nc}, reg_max={reg_max}"
        )
    return reg_max, nc


def extract_raw_head_outputs_from_pure_model(pure_model: nn.Module, x: torch.Tensor):
    layers = pure_model.model
    y = []
    for m in layers:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

        if hasattr(m, "cv2") and hasattr(m, "cv3") and hasattr(m, "nl"):
            if not isinstance(x, (list, tuple)):
                raise RuntimeError("Detect head input is expected to be a feature list/tuple")
            x = YOLOv8HeadIncludedWrapper._run_raw_detect_head(m, list(x))
        else:
            x = m(x)

        y.append(x)

    if not isinstance(x, (list, tuple)):
        raise RuntimeError("Expected multi-scale head outputs, got single tensor output")
    return tuple(x)


def _slice_cls_logits(out: torch.Tensor, reg_max: int, num_classes: int) -> torch.Tensor:
    start = 4 * reg_max
    end = start + num_classes
    return out[:, start:end, :, :]


def debug_compare_pure_wrapper_quant(
    pure_model: nn.Module,
    wrapper_model: nn.Module,
    quant_model: nn.Module,
    samples: int,
    calib_loader: DataLoader,
    reg_max: int,
    num_classes: int,
):
    print("[DBG] Pure-vs-wrapper-vs-quant comparison (same calibration input)")
    pure_model.eval()
    wrapper_model.eval()
    quant_model.eval()

    seen = 0
    with torch.no_grad():
        for batch in calib_loader:
            inp = batch
            p_out = _sort_head_outputs_by_scale(extract_raw_head_outputs_from_pure_model(pure_model, inp))
            w_out = _sort_head_outputs_by_scale(wrapper_model(inp))
            q_out = _sort_head_outputs_by_scale(quant_model(inp))

            if not (len(p_out) == len(w_out) == len(q_out)):
                print(
                    f"[DBG] output count mismatch pure={len(p_out)} wrapper={len(w_out)} quant={len(q_out)}"
                )
                return

            for i, (po, wo, qo) in enumerate(zip(p_out, w_out, q_out)):
                pc = _slice_cls_logits(po, reg_max, num_classes)
                wc = _slice_cls_logits(wo, reg_max, num_classes)
                qc = _slice_cls_logits(qo, reg_max, num_classes)

                p2w = _compare_tensor_pair(po, wo)
                p2q = _compare_tensor_pair(po, qo)
                w2q = _compare_tensor_pair(wo, qo)
                pc2wc = _compare_tensor_pair(pc, wc)
                pc2qc = _compare_tensor_pair(pc, qc)

                p_cls_max = float(torch.sigmoid(pc).max().item())
                w_cls_max = float(torch.sigmoid(wc).max().item())
                q_cls_max = float(torch.sigmoid(qc).max().item())

                print(
                    "[DBG] sample={} out={} shape={} "
                    "cls_prob_max[pure/wrap/quant]=({:.4f}/{:.4f}/{:.4f}) "
                    "full_cos[pw/pq/wq]=({:.6f}/{:.6f}/{:.6f}) "
                    "cls_cos[pw/pq]=({:.6f}/{:.6f}) "
                    "cls_mae[pw/pq]=({:.6f}/{:.6f})".format(
                        seen + 1,
                        i,
                        list(po.shape),
                        p_cls_max,
                        w_cls_max,
                        q_cls_max,
                        p2w["cosine"],
                        p2q["cosine"],
                        w2q["cosine"],
                        pc2wc["cosine"],
                        pc2qc["cosine"],
                        pc2wc["mae"],
                        pc2qc["mae"],
                    )
                )

            seen += 1
            if seen >= samples:
                break

    if seen == 0:
        print("[DBG] No sample was available for pure/wrapper/quant compare")


def debug_compare_float_vs_quant(
    float_model: nn.Module,
    quant_model: nn.Module,
    img_size: int,
    samples: int,
    calib_loader: DataLoader,
):
    print("[DBG] Float vs quantized output comparison")
    float_model.eval()
    quant_model.eval()

    seen = 0
    with torch.no_grad():
        for batch in calib_loader:
            inp = batch
            f_out = _as_tuple_outputs(float_model(inp))
            q_out = _as_tuple_outputs(quant_model(inp))

            if len(f_out) != len(q_out):
                print(f"[DBG] Output count mismatch float={len(f_out)} quant={len(q_out)}")
                return

            for i, (fo, qo) in enumerate(zip(f_out, q_out)):
                fs = _tensor_stats(fo)
                qs = _tensor_stats(qo)
                cmpm = _compare_tensor_pair(fo, qo)
                print(
                    "[DBG] sample={} out={} shape={} "
                    "float[min,max,std]=({:.4f},{:.4f},{:.4f}) "
                    "quant[min,max,std]=({:.4f},{:.4f},{:.4f}) "
                    "mae={:.6f} maxae={:.6f} cosine={:.6f}".format(
                        seen + 1,
                        i,
                        list(fo.shape),
                        fs["min"],
                        fs["max"],
                        fs["std"],
                        qs["min"],
                        qs["max"],
                        qs["std"],
                        cmpm["mae"],
                        cmpm["maxae"],
                        cmpm["cosine"],
                    )
                )

            seen += 1
            if seen >= samples:
                break

    if seen == 0:
        print("[DBG] No sample was available for debug compare")


class YOLOv8HeadIncludedWrapper(nn.Module):
    """
    Export YOLOv8 with detect head conv branches included.
    Returns raw per-scale head tensors (pre-decode, pre-NMS), which are
    generally more compiler-friendly than full detect decode ops.
    """

    def __init__(self, pure_yolo_model: nn.Module):
        super().__init__()
        self.layers = pure_yolo_model.model

    @staticmethod
    def _run_raw_detect_head(detect_module: nn.Module, feats: List[torch.Tensor]):
        if len(feats) != int(detect_module.nl):
            raise RuntimeError(
                f"Detect head expects {int(detect_module.nl)} feature maps, got {len(feats)}"
            )

        outs = []
        for i in range(int(detect_module.nl)):
            # Keep only convolutional head branches to avoid runtime-only decode ops.
            out_i = torch.cat((detect_module.cv2[i](feats[i]), detect_module.cv3[i](feats[i])), 1)
            outs.append(out_i)
        return tuple(outs)

    def forward(self, x):
        y = []
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # Detect-like module: include head conv branches and stop at raw head outputs.
            if hasattr(m, "cv2") and hasattr(m, "cv3") and hasattr(m, "nl"):
                if not isinstance(x, (list, tuple)):
                    raise RuntimeError("Detect head input is expected to be a feature list/tuple")
                x = self._run_raw_detect_head(m, list(x))
            else:
                x = m(x)

            y.append(x)

        if not isinstance(x, (list, tuple)):
            raise RuntimeError("Expected multi-scale head outputs, got single tensor output")
        return tuple(x)


@dataclass
class QuantizeConfig:
    model: str = None
    calib_dir: str = None
    output_dir: str = "./quant_results"
    compiled_dir: str = "./compiled_output"
    img_size: int = DEFAULT_IMG_SIZE
    calib_count: int = 150
    batch_size: int = 1
    seed: int = 42
    arch: str = None
    # demo/task_live.py 와 02_wake_gesture_switch.py 가 "./task_ai.xmodel" 을 연다.
    # 이 이름이 어긋나면 보드에서 모델을 못 찾거나 제스처 xmodel 을 덮어쓴다.
    net_name: str = "task_ai"
    skip_compile: bool = False
    replace_silu: bool = True
    replace_silu_hardswish: bool = False
    replace_silu_leakyrelu: bool = False   # 학습이 ReLU 로 진행됨 (parse_args 주석 참고)
    leaky_slope: float = 0.08
    deploy_check: bool = False
    debug_compare: bool = False
    debug_samples: int = 3
    debug_ablation: bool = False
    allow_multi_subgraph: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Safe YOLOv8(with head) -> NNDCT xmodel export")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLO .pt model (생략 시 DEFAULT_MODEL_CANDIDATES 자동 탐색)")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="Calibration image root (생략 시 DEFAULT_CALIB_DIR_CANDIDATES 자동 탐색)")
    parser.add_argument("--output-dir", type=str, default="./results_quant", help="Output directory")
    parser.add_argument(
        "--compiled-dir",
        type=str,
        default="./compiled_output",
        help="Output directory for compiled xmodel",
    )
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE,
                        help="Input image size (square). 학습값 416 과 반드시 일치시킬 것")
    parser.add_argument("--calib-count", type=int, default=150, help="Max calibration images")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--arch", type=str, default=None, help="Path to DPU arch.json for compilation")
    # 기본값이 "gesture_stage1_kv260" 이면 9클래스 task 모델이 제스처 xmodel 을 덮어쓴다.
    parser.add_argument("--net-name", type=str, default="task_ai", help="Compiled xmodel name")
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip vai_c_xir compile stage and only export raw xmodel",
    )
    parser.add_argument(
        "--replace-silu",
        dest="replace_silu",
        action="store_true",
        help="Replace SiLU with ReLU for compatibility (default: enabled)",
    )
    parser.add_argument(
        "--keep-silu",
        dest="replace_silu",
        action="store_false",
        help="Keep SiLU unchanged (may lead to 0 DPU subgraphs on older toolchains)",
    )
    parser.add_argument(
        "--replace-silu-hardswish",
        action="store_true",
        help="Replace SiLU with HardSwish instead of ReLU",
    )
    parser.add_argument(
        "--replace-silu-leakyrelu",
        action="store_true",
        help="Replace SiLU with LeakyReLU instead of ReLU (default: enabled)",
    )
    parser.add_argument(
        "--leaky-slope",
        type=float,
        default=0.08,
        help="Negative slope for --replace-silu-leakyrelu",
    )
    # 이 체크포인트는 replace_silu_with_relu() 를 거친 뒤 학습됐다. SiLU 가 남아 있을 리 없고,
    # 혹시 남아 있다면 LeakyReLU 가 아니라 ReLU 로 맞춰야 학습 시 활성함수와 일치한다.
    parser.set_defaults(replace_silu=True, replace_silu_leakyrelu=False)
    parser.add_argument(
        "--allow-multi-subgraph",
        action="store_true",
        help="컴파일 결과 DPU 서브그래프가 2개 이상이어도 에러 대신 경고만 낸다",
    )
    parser.add_argument(
        "--deploy-check",
        action="store_true",
        help="Enable NNDCT deploy checker during export_xmodel",
    )
    parser.add_argument(
        "--debug-compare",
        action="store_true",
        help="Print float-vs-quant output statistics on calibration samples",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=3,
        help="Number of samples used by --debug-compare",
    )
    parser.add_argument(
        "--debug-ablation",
        action="store_true",
        help="Run transform ablation: baseline wrapper vs C2f-only vs C2f+SiLU",
    )
    return parser.parse_args()


def config_from_args(args) -> QuantizeConfig:
    return QuantizeConfig(
        model=args.model,
        calib_dir=args.calib_dir,
        output_dir=args.output_dir,
        compiled_dir=args.compiled_dir,
        img_size=args.img_size,
        calib_count=args.calib_count,
        batch_size=args.batch_size,
        seed=args.seed,
        arch=args.arch,
        net_name=args.net_name,
        skip_compile=args.skip_compile,
        replace_silu=args.replace_silu,
        replace_silu_hardswish=args.replace_silu_hardswish,
        replace_silu_leakyrelu=args.replace_silu_leakyrelu,
        leaky_slope=args.leaky_slope,
        deploy_check=args.deploy_check,
        debug_compare=args.debug_compare,
        debug_samples=args.debug_samples,
        debug_ablation=args.debug_ablation,
        allow_multi_subgraph=args.allow_multi_subgraph,
    )


def _build_wrapper_variant(pure_model: nn.Module, apply_c2f: bool, apply_silu: bool):
    variant = copy.deepcopy(pure_model).eval()
    replaced_c2f = 0
    if apply_c2f:
        replaced_c2f = replace_c2f_for_nndct(variant)
    if apply_silu:
        replace_silu_with_relu(variant)
        force_replace_conv_acts(variant)
    wrapper = YOLOv8HeadIncludedWrapper(variant).eval()
    fix_conv_bias_for_nndct(wrapper)
    silu_left = count_silu_modules(variant)
    return wrapper, replaced_c2f, silu_left


def debug_ablation_transform_effects(
    pure_model: nn.Module,
    calib_loader: DataLoader,
    samples: int,
    img_size: int = DEFAULT_IMG_SIZE,
):
    print("[DBG] Transform ablation: baseline vs C2f-only vs C2f+SiLU")

    base_wrap, base_c2f, base_silu = _build_wrapper_variant(pure_model, apply_c2f=False, apply_silu=False)
    c2f_wrap, c2f_replaced, c2f_silu_left = _build_wrapper_variant(pure_model, apply_c2f=True, apply_silu=False)
    full_wrap, full_replaced, full_silu_left = _build_wrapper_variant(pure_model, apply_c2f=True, apply_silu=True)

    print(
        f"[DBG] Variant stats: base(c2f={base_c2f}, silu_left={base_silu}) "
        f"c2f_only(c2f={c2f_replaced}, silu_left={c2f_silu_left}) "
        f"c2f_silu(c2f={full_replaced}, silu_left={full_silu_left})"
    )

    detect_ref = _find_detect_module(pure_model)
    dummy = torch.randn(1, 3, img_size, img_size)   # cfg.img_size 와 어긋나면 채널 추론이 깨진다
    with torch.no_grad():
        ref_out = _sort_head_outputs_by_scale(base_wrap(dummy))[0]
    reg_max, num_classes = infer_head_meta(detect_ref, ref_out)
    print(f"[DBG] Ablation head meta: reg_max={reg_max}, num_classes={num_classes}")

    seen = 0
    with torch.no_grad():
        for batch in calib_loader:
            inp = batch
            bo = _sort_head_outputs_by_scale(base_wrap(inp))
            co = _sort_head_outputs_by_scale(c2f_wrap(inp))
            fo = _sort_head_outputs_by_scale(full_wrap(inp))

            if not (len(bo) == len(co) == len(fo)):
                print(
                    f"[DBG] Ablation output count mismatch base={len(bo)} c2f={len(co)} full={len(fo)}"
                )
                return

            for i, (b, c, f) in enumerate(zip(bo, co, fo)):
                bc = _slice_cls_logits(b, reg_max, num_classes)
                cc = _slice_cls_logits(c, reg_max, num_classes)
                fc = _slice_cls_logits(f, reg_max, num_classes)

                b_cls_max = float(torch.sigmoid(bc).max().item())
                c_cls_max = float(torch.sigmoid(cc).max().item())
                f_cls_max = float(torch.sigmoid(fc).max().item())

                b2c = _compare_tensor_pair(b, c)
                c2f = _compare_tensor_pair(c, f)
                bcls2c = _compare_tensor_pair(bc, cc)
                ccls2f = _compare_tensor_pair(cc, fc)

                print(
                    "[DBG] ablation sample={} out={} shape={} "
                    "cls_prob_max[base/c2f/full]=({:.4f}/{:.4f}/{:.4f}) "
                    "full_cos[b2c/c2f]=({:.6f}/{:.6f}) "
                    "cls_cos[b2c/c2f]=({:.6f}/{:.6f}) "
                    "cls_mae[b2c/c2f]=({:.6f}/{:.6f})".format(
                        seen + 1,
                        i,
                        list(b.shape),
                        b_cls_max,
                        c_cls_max,
                        f_cls_max,
                        b2c["cosine"],
                        c2f["cosine"],
                        bcls2c["cosine"],
                        ccls2f["cosine"],
                        bcls2c["mae"],
                        ccls2f["mae"],
                    )
                )

            seen += 1
            if seen >= samples:
                break

    if seen == 0:
        print("[DBG] No sample was available for ablation compare")


def resolve_existing_path(user_path, candidates, kind):
    if user_path:
        if os.path.exists(user_path):
            return user_path
        raise FileNotFoundError(f"{kind} not found: {user_path}")

    for path in candidates:
        if os.path.exists(path):
            print(f"[INFO] Auto-selected {kind}: {path}")
            return path

    raise FileNotFoundError(
        f"No valid {kind} found. Checked candidates: {', '.join(candidates)}"
    )


def list_xmodel_files(output_dir: str) -> List[str]:
    files = glob.glob(os.path.join(output_dir, "*.xmodel"))
    files = sorted(files, key=lambda p: os.path.getmtime(p))
    return files


def _walk_subgraphs(root) -> List:
    nodes = [root]
    if root.is_leaf:
        return nodes
    for child in root.toposort_child_subgraph():
        nodes.extend(_walk_subgraphs(child))
    return nodes


def get_device_histogram(xmodel_path: str) -> Dict[str, int]:
    import xir

    graph = xir.Graph.deserialize(xmodel_path)
    root = graph.get_root_subgraph()
    all_subgraphs = _walk_subgraphs(root)

    hist = {}
    for sg in all_subgraphs:
        device = "UNSPECIFIED"
        if sg.has_attr("device"):
            device = str(sg.get_attr("device")).upper()
        hist[device] = hist.get(device, 0) + 1
    return hist


def count_dpu_like_subgraphs(device_hist: Dict[str, int]) -> int:
    total = 0
    for dev, n in device_hist.items():
        if "DPU" in dev.upper():
            total += n
    return total


def compile_xmodel_with_vai_c_xir(xmodel_path: str, arch_json: str, output_dir: str, net_name: str) -> str:
    before = set(glob.glob(os.path.join(output_dir, "*.xmodel")))

    cmd = [
        "vai_c_xir",
        "-x",
        xmodel_path,
        "-a",
        arch_json,
        "-o",
        output_dir,
        "-n",
        net_name,
    ]
    print(f"[STEP] Running compiler: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "vai_c_xir failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if result.stdout.strip():
        print("[INFO] vai_c_xir stdout:\n" + result.stdout.strip())
    if result.stderr.strip():
        print("[INFO] vai_c_xir stderr:\n" + result.stderr.strip())

    compiled_path = os.path.join(output_dir, f"{net_name}.xmodel")
    if os.path.exists(compiled_path):
        return compiled_path

    after = set(glob.glob(os.path.join(output_dir, "*.xmodel")))
    new_files = sorted(after - before, key=lambda p: os.path.getmtime(p))
    if new_files:
        print(f"[WARN] Expected '{compiled_path}' not found. Using generated file: {new_files[-1]}")
        return new_files[-1]

    all_files = sorted(after, key=lambda p: os.path.getmtime(p))
    if all_files:
        print(f"[WARN] No new xmodel detected. Using latest existing file: {all_files[-1]}")
        return all_files[-1]

    raise FileNotFoundError(f"Compiled xmodel not found in directory: {output_dir}")


def run_quantize(cfg: QuantizeConfig) -> Dict[str, str]:
    set_seed(cfg.seed)

    cfg.model = resolve_existing_path(cfg.model, DEFAULT_MODEL_CANDIDATES, "model file")
    cfg.calib_dir = resolve_existing_path(cfg.calib_dir, DEFAULT_CALIB_DIR_CANDIDATES, "calibration directory")
    if not cfg.skip_compile:
        cfg.arch = resolve_existing_path(cfg.arch, DEFAULT_ARCH_JSON_CANDIDATES, "arch.json")

    if not os.path.isfile(cfg.model):
        raise FileNotFoundError(f"Model file is not a file: {cfg.model}")
    if not os.path.isdir(cfg.calib_dir):
        raise FileNotFoundError(f"Calibration directory is not a directory: {cfg.calib_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.compiled_dir, exist_ok=True)

    print("[STEP] Loading YOLO model:", cfg.model)
    yolo = YOLO(cfg.model)
    pure_ref = copy.deepcopy(yolo.model).eval()
    pure = yolo.model.eval()

    # --- 체크포인트가 정말 Stage 1 결과물인지 확인 ---
    names = getattr(yolo.model, "names", None) or getattr(yolo, "names", None) or {}
    names_list = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    print(f"[CHECK] 체크포인트 클래스: {names_list}")
    if names_list != EXPECTED_CLASS_NAMES:
        print(f"[WARN] 기대값과 다르다: {EXPECTED_CLASS_NAMES}")
        print("[WARN] 잘못된 .pt 를 지정했을 가능성이 높다. 계속 진행하면 무의미한 xmodel 이 나온다.")

    n_silu_ckpt = count_silu_modules(pure)
    print(f"[CHECK] 체크포인트의 SiLU 모듈 수: {n_silu_ckpt} (기대값 0)")
    if n_silu_ckpt > 0:
        print("[WARN] 학습 노트북은 replace_silu_with_relu() 후 학습했다. 0 이 아니면 사전학습")
        print("[WARN] 가중치(yolov8s.pt)를 잘못 넘겼는지 확인할 것.")

    replaced_c2f = replace_c2f_for_nndct(pure)
    print(f"[STEP] Replaced C2f blocks for NNDCT compatibility: {replaced_c2f}")

    if cfg.replace_silu:
        if cfg.replace_silu_hardswish:
            print("[STEP] Replacing SiLU -> decomposed SiLU (x * sigmoid(x)) fallback")
            replace_silu_with_hardswish(pure)
            force_replace_conv_acts_hardswish(pure)
        elif cfg.replace_silu_leakyrelu:
            print(f"[STEP] Replacing SiLU -> LeakyReLU(slope={float(cfg.leaky_slope):.4f})")
            replace_silu_with_leakyrelu(pure, negative_slope=float(cfg.leaky_slope))
            force_replace_conv_acts_leakyrelu(pure, negative_slope=float(cfg.leaky_slope))
        else:
            print("[STEP] Replacing SiLU -> ReLU")
            replace_silu_with_relu(pure)
            force_replace_conv_acts(pure)
    silu_left = count_silu_modules(pure)
    print(f"[INFO] Remaining SiLU modules: {silu_left}")
    if cfg.replace_silu and silu_left > 0:
        print("[WARN] Some SiLU modules still remain after replacement.")

    print("[STEP] Building head-included wrapper (raw head outputs)")
    dpu_model = YOLOv8HeadIncludedWrapper(pure).eval()
    fix_conv_bias_for_nndct(dpu_model)

    if cfg.replace_silu and trace_contains_silu_op(dpu_model, cfg.img_size):
        raise RuntimeError(
            "Traced graph still contains aten::silu op. "
            "This typically prevents DPU subgraph generation on older toolchains."
        )

    dummy_input = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    with torch.no_grad():
        pre_out = dpu_model(dummy_input)
    print("[CHECK] Wrapper outputs:", [list(t.shape) for t in pre_out])

    detect_ref = _find_detect_module(pure_ref)
    reg_max_head, num_classes_head = infer_head_meta(detect_ref, pre_out[0])
    print(f"[CHECK] Head meta inferred: reg_max={reg_max_head}, num_classes={num_classes_head}")
    if num_classes_head != len(EXPECTED_CLASS_NAMES):
        raise RuntimeError(
            f"헤드 클래스 수 {num_classes_head} != 기대값 {len(EXPECTED_CLASS_NAMES)}. "
            "Stage 1 체크포인트가 아니다."
        )

    print("[STEP] Building calibration loader")
    dataset = CalibrationDataset(cfg.calib_dir, img_size=cfg.img_size,
                                 max_images=cfg.calib_count, seed=cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    print(f"[INFO] Calibration samples: {len(dataset)}")

    if cfg.debug_ablation:
        debug_ablation_transform_effects(
            pure_model=pure_ref,
            calib_loader=loader,
            samples=max(1, int(cfg.debug_samples)),
            img_size=cfg.img_size,
        )

    print("[STEP] Calibration mode")
    quantizer = torch_quantizer("calib", dpu_model, (dummy_input,), output_dir=cfg.output_dir)
    quant_model = quantizer.quant_model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            _ = quant_model(batch)
            if i % 25 == 0 or i == len(loader):
                print(f"  processed {i}/{len(loader)} batches")

    quantizer.export_quant_config()
    print("[OK] Quant config exported")

    print("[STEP] Test mode + xmodel export")
    quantizer = torch_quantizer("test", dpu_model, (dummy_input,), output_dir=cfg.output_dir)
    quant_model = quantizer.quant_model.eval()

    with torch.no_grad():
        out = quant_model(dummy_input)
    print("[CHECK] Quant wrapper outputs:", [list(t.shape) for t in out])

    if cfg.debug_compare:
        debug_compare_pure_wrapper_quant(
            pure_model=pure_ref,
            wrapper_model=dpu_model,
            quant_model=quant_model,
            samples=max(1, int(cfg.debug_samples)),
            calib_loader=loader,
            reg_max=reg_max_head,
            num_classes=num_classes_head,
        )
        debug_compare_float_vs_quant(
            float_model=dpu_model,
            quant_model=quant_model,
            img_size=cfg.img_size,
            samples=max(1, int(cfg.debug_samples)),
            calib_loader=loader,
        )

    before_export = set(list_xmodel_files(cfg.output_dir))
    quantizer.export_xmodel(output_dir=cfg.output_dir, deploy_check=cfg.deploy_check)
    print("[DONE] xmodel export finished")

    xmodels = list_xmodel_files(cfg.output_dir)
    if not xmodels:
        raise RuntimeError(f"No .xmodel file found in output dir: {cfg.output_dir}")

    after_export = set(xmodels)
    new_raw = sorted(after_export - before_export, key=lambda p: os.path.getmtime(p))
    if new_raw:
        latest_xmodel = new_raw[-1]
    else:
        # Fallback: prefer freshly exported int xmodel pattern, otherwise latest by mtime.
        int_candidates = [p for p in xmodels if p.endswith("_int.xmodel")]
        if int_candidates:
            int_candidates = sorted(int_candidates, key=lambda p: os.path.getmtime(p))
            latest_xmodel = int_candidates[-1]
        else:
            latest_xmodel = xmodels[-1]
    print(f"[STEP] Verifying raw xmodel subgraph info: {latest_xmodel}")

    try:
        dev_hist = get_device_histogram(latest_xmodel)
    except Exception as exc:
        print(f"[WARN] Could not verify subgraph count via xir: {exc}")
        return {
            "raw_xmodel": latest_xmodel,
            "compiled_xmodel": "",
            "raw_dpu_subgraphs": "unknown",
            "compiled_dpu_subgraphs": "",
        }

    dpu_count = count_dpu_like_subgraphs(dev_hist)
    print(f"[CHECK] Raw device histogram = {dev_hist}")
    print(f"[CHECK] Raw DPU-like subgraph count = {dpu_count}")

    if cfg.skip_compile:
        print("[WARN] Compile stage skipped; raw xmodel often has UNSPECIFIED device.")
        return {
            "raw_xmodel": latest_xmodel,
            "compiled_xmodel": "",
            "raw_dpu_subgraphs": str(dpu_count),
            "compiled_dpu_subgraphs": "",
        }

    compiled_xmodel = compile_xmodel_with_vai_c_xir(
        xmodel_path=latest_xmodel,
        arch_json=cfg.arch,
        output_dir=cfg.compiled_dir,
        net_name=cfg.net_name,
    )
    print(f"[STEP] Verifying compiled xmodel subgraph count: {compiled_xmodel}")

    compiled_hist = get_device_histogram(compiled_xmodel)
    compiled_dpu = count_dpu_like_subgraphs(compiled_hist)
    print(f"[CHECK] Compiled device histogram = {compiled_hist}")
    print(f"[CHECK] Compiled DPU-like subgraph count = {compiled_dpu}")
    if compiled_dpu != 1:
        msg = (
            f"DPU 서브그래프가 {compiled_dpu}개다 (기대값 1). 중간에 CPU 로 떨어지는 레이어가 "
            "있다는 뜻이고, 그 지점마다 DPU<->CPU 전송이 발생해 fps 가 크게 떨어진다.\n"
            f"  전체 device 분포: {compiled_hist}\n"
            "  대응: --debug-ablation 으로 어느 변환이 원인인지 확인하거나, 학습 노트북의 "
            "replace_c2f_with_dpu_c3() 주석을 풀고 Stage 1 을 재학습한다."
        )
        if not cfg.allow_multi_subgraph:
            raise RuntimeError(msg)
        print("[WARN] " + msg)
    else:
        print("[OK] Single DPU subgraph verified on compiled xmodel")

    return {
        "raw_xmodel": latest_xmodel,
        "compiled_xmodel": compiled_xmodel,
        "raw_dpu_subgraphs": str(dpu_count),
        "compiled_dpu_subgraphs": str(compiled_dpu),
    }


def main():
    # CLI-only execution path for Docker/runtime usage.
    args = parse_args()
    cfg = config_from_args(args)
    run_quantize(cfg)


if __name__ == "__main__":
    main()
