#!/bin/bash
# =============================================================================
# Phase A — 부팅 시 자동 적용 설치 (커널 교체 없음)
#
# 설치되는 것:
#   /lib/modules/<커널>/extra/uvcvideo_bwcap.ko   패치 모듈
#   /usr/local/sbin/uvc-swap.sh                   전환 스크립트
#   /etc/systemd/system/uvc-bwcap.service         부팅 시 스왑
#   /etc/udev/rules.d/99-uvc-bwcap.rules          핫플러그 + 노출 + 고정 심볼릭
#
# 사용법:
#   sudo ./install-phase-a.sh [cap]        (cap 기본값 1024)
#   sudo ./install-phase-a.sh --uninstall
# =============================================================================
set -euo pipefail

MODNAME="uvcvideo_bwcap"
KVER="$(uname -r)"
MODDIR="/lib/modules/${KVER}/extra"
SRC_KO="${SRC_KO:-$HOME/uvc-bwcap/build/drivers/media/usb/uvc/${MODNAME}.ko}"
SRC_SWAP="${SRC_SWAP:-./uvc-swap.sh}"
UNIT="/etc/systemd/system/uvc-bwcap.service"
RULES="/etc/udev/rules.d/99-uvc-bwcap.rules"
SBIN="/usr/local/sbin/uvc-swap.sh"

[[ $EUID -eq 0 ]] || { echo "ERROR: sudo 로 실행하세요"; exit 1; }

# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--uninstall" ]]; then
    echo "==> 제거"
    systemctl disable --now uvc-bwcap.service 2>/dev/null || true
    [[ -x "$SBIN" ]] && "$SBIN" off || true
    rm -f "$UNIT" "$RULES" "$SBIN" "$MODDIR/${MODNAME}.ko"
    systemctl daemon-reload
    udevadm control --reload
    depmod -a "$KVER"
    echo "==> 완료. builtin uvcvideo 로 복귀했습니다."
    exit 0
fi

CAP="${1:-1024}"
[[ -f "$SRC_KO" ]]   || { echo "ERROR: 모듈 없음: $SRC_KO"; exit 1; }
[[ -f "$SRC_SWAP" ]] || { echo "ERROR: uvc-swap.sh 없음: $SRC_SWAP"; exit 1; }

echo "==> 커널 $KVER / bandwidth_cap=$CAP"

# ---------------------------------------------------------------------------
# 1. 모듈과 스크립트 설치
# ---------------------------------------------------------------------------
echo "==> [1/4] 모듈 설치"
install -D -m 644 "$SRC_KO" "$MODDIR/${MODNAME}.ko"
depmod -a "$KVER"

install -m 755 "$SRC_SWAP" "$SBIN"
# 설치된 모듈 경로를 기본값으로 고정한다. 홈 디렉터리 빌드 트리에
# 의존하면 root 로 도는 systemd/udev 컨텍스트에서 경로가 어긋난다.
sed -i "s|^KO=.*|KO=\"\${KO:-$MODDIR/${MODNAME}.ko}\"|" "$SBIN"
grep -q "$MODDIR" "$SBIN" || { echo "ERROR: KO 경로 치환 실패"; exit 1; }

# ---------------------------------------------------------------------------
# 2. systemd 유닛
#
# RemainAfterExit 를 쓰지 않는다. 그래야 udev 가 나중에 다시 start 를 걸었을 때
# oneshot 이 재실행된다 (핫플러그 대응).
# ---------------------------------------------------------------------------
echo "==> [2/4] systemd 유닛"
cat > "$UNIT" <<EOF
[Unit]
Description=Swap UVC cameras onto patched ${MODNAME} (bandwidth_cap=${CAP})
After=systemd-udev-settle.service
Wants=systemd-udev-settle.service

[Service]
Type=oneshot
ExecStart=${SBIN} on ${CAP}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ---------------------------------------------------------------------------
# 3. udev 규칙
# ---------------------------------------------------------------------------
echo "==> [3/4] udev 규칙"

# 현재 붙어 있는 UVC 카메라의 포트 경로를 읽어 고정 심볼릭 링크를 만든다.
# C270 2대는 VID/PID 가 같고 iSerial 도 없어서 포트 위치로만 구분된다.
SYMLINKS=""
idx=0
for i in /sys/bus/usb/devices/*-*:*; do
    [[ -f "$i/bInterfaceClass" ]] || continue
    [[ "$(cat "$i/bInterfaceClass")"    == "0e" ]] || continue
    [[ "$(cat "$i/bInterfaceSubClass")" == "01" ]] || continue
    port="$(basename "$i")"; port="${port%%:*}"
    SYMLINKS+="SUBSYSTEM==\"video4linux\", KERNELS==\"${port}\", ATTR{index}==\"0\", SYMLINK+=\"cam${idx}\"
"
    echo "    cam${idx} -> 포트 ${port}"
    idx=$((idx+1))
done

cat > "$RULES" <<EOF
# uvcvideo bandwidth_cap 자동 적용 (Phase A)

# --- 핫플러그: 카메라가 꽂히면 builtin 에서 패치 모듈로 다시 스왑 ---
# udev RUN 은 오래 걸리는 작업에 부적합하므로 systemd 로 넘긴다.
ACTION=="add", SUBSYSTEM=="usb", ENV{INTERFACE}=="14/1/*", \\
  RUN+="/bin/systemctl --no-block start uvc-bwcap.service"

# --- 노출: 저조도에서 프레임레이트를 절반으로 떨구는 동작을 끈다 ---
# 이걸 켜두면 어두운 곳에서 30fps 가 15fps 로 내려간다.
ACTION=="add", SUBSYSTEM=="video4linux", ATTR{index}=="0", \\
  ATTRS{idVendor}=="046d", ATTRS{idProduct}=="0825", \\
  RUN+="/usr/bin/v4l2-ctl -d /dev/%k -c exposure_dynamic_framerate=0"

# --- 고정 이름: 두 카메라가 VID/PID 동일, iSerial 없음 → 포트로 구분 ---
${SYMLINKS}
EOF

# ---------------------------------------------------------------------------
# 4. 활성화
# ---------------------------------------------------------------------------
echo "==> [4/4] 활성화"
systemctl daemon-reload
systemctl enable uvc-bwcap.service
udevadm control --reload
udevadm trigger --subsystem-match=video4linux --action=add

systemctl start uvc-bwcap.service || true
sleep 1

echo
echo "============================================================"
"$SBIN" status
echo "============================================================"
cat <<EOF

확인
  systemctl status uvc-bwcap.service -> 1-1.1:1.0 -> uvcvideo_bwcap (패치됨)
  journalctl -u uvc-bwcap.service -b
  ls -l /dev/cam*
  스트림을 한 번 열어서 로그 유발
  v4l2-ctl -d /dev/cam0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG \
  --set-parm=30 --stream-mmap --stream-count=30 --stream-to=/dev/null
  dmesg | grep -iE "Capping|alternate setting" | tail
  
중요 — 커널 업그레이드 시 모듈이 깨집니다
  .ko 는 ${KVER} 전용입니다. apt 가 커널을 올리면 로드에 실패하고
  builtin 으로 되돌아가 두 번째 카메라가 다시 -ENOSPC 를 냅니다.
  고정을 권합니다:

    sudo apt-mark hold linux-image-xilinx-zynqmp \\
                       linux-headers-xilinx-zynqmp linux-xilinx-zynqmp

  업그레이드가 필요하면 새 커널로 부팅한 뒤 phase1-build.sh 를 다시 돌리고
  이 스크립트를 재실행하세요.

제거
  sudo ./install-phase-a.sh --uninstall
EOF
