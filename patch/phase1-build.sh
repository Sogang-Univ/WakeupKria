#!/bin/bash
# =============================================================================
# Phase 1: uvcvideo bandwidth_cap 검증용 out-of-tree 모듈 빌드
#
# 실행 위치: KV260 본체 (aarch64). x86 PC가 아닙니다.
#            Ubuntu 헤더 패키지의 scripts/ 바이너리가 arm64라서
#            x86에서 크로스 빌드하면 실패합니다. 온보드 네이티브가 정답입니다.
#
# 소요: 소스 다운로드 5~10분 + 컴파일 1~2분
# 위험: 커널을 교체하지 않습니다. rmmod 한 번이면 원상복구됩니다.
#
# 사용법:
#   sudo ./phase1-build.sh /path/to/uvc-bandwidth-cap-5.15.patch
# =============================================================================
set -euo pipefail

PATCHFILE="${1:-./uvc-bandwidth-cap-5.15.patch}"
KVER="$(uname -r)"
WORK="${WORK:-$HOME/uvc-bwcap}"
MODNAME="uvcvideo_bwcap"

[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: KV260 본체에서 실행하세요 (현재: $(uname -m))"; exit 1; }
[[ -f "$PATCHFILE" ]] || { echo "ERROR: 패치 파일 없음: $PATCHFILE"; exit 1; }
PATCHFILE="$(readlink -f "$PATCHFILE")"

echo "==> 커널: $KVER / 작업 디렉터리: $WORK"
mkdir -p "$WORK"; cd "$WORK"

# ---------------------------------------------------------------------------
# 1. 빌드 도구와 커널 헤더
# ---------------------------------------------------------------------------
echo "==> [1/5] 헤더와 빌드 도구 설치"
apt-get update -qq
apt-get install -y build-essential bc kmod cpio flex bison libssl-dev \
                   libelf-dev dpkg-dev "linux-headers-${KVER}"

[[ -d "/lib/modules/${KVER}/build" ]] \
  || { echo "ERROR: /lib/modules/${KVER}/build 없음. 헤더 설치 실패"; exit 1; }

# ---------------------------------------------------------------------------
# 2. 실제 Ubuntu/Xilinx 커널 소스 확보
#
#    바닐라 v5.15를 쓰면 안 됩니다. 이 커널의 uvcvideo는 수정되어 있습니다.
#    (dmesg의 "Allocated 50 URB buffers of 48x2048" — 바닐라는 5 x 32)
#    그 수정을 유지해야 지금 검증한 동작이 재현됩니다.
# ---------------------------------------------------------------------------
echo "==> [2/5] 커널 소스 다운로드 (deb-src 활성화 필요)"
if ! grep -rqs '^deb-src' /etc/apt/sources.list /etc/apt/sources.list.d/; then
    echo "    deb-src가 꺼져 있어 활성화합니다"
    sed -i 's/^# *deb-src/deb-src/' /etc/apt/sources.list
    apt-get update -qq
fi

if [[ ! -d "$WORK/src" ]]; then
    mkdir -p "$WORK/dl"; pushd "$WORK/dl" >/dev/null

    # --only-source 가 반드시 필요하다.
    # 없이 부르면 apt가 동명의 '바이너리' 패키지 linux-xilinx-zynqmp 를 보고
    # 그 소스인 linux-meta-xilinx-zynqmp (의존성만 든 8KB 껍데기) 를 받아온다.
    rm -rf linux-meta-* 2>/dev/null || true
    apt-get source --only-source linux-xilinx-zynqmp \
      || { echo "ERROR: 소스 받기 실패. README의 수동 대안을 참고하세요"; exit 1; }

    # 이름이 아니라 '내용물'로 고른다 — uvc 디렉터리를 실제로 가진 것만 채택
    SRCDIR=""
    for d in linux-*/; do
        [[ -d "$d/drivers/media/usb/uvc" ]] && { SRCDIR="${d%/}"; break; }
    done
    [[ -n "$SRCDIR" ]] || {
        echo "ERROR: drivers/media/usb/uvc 를 포함한 소스 디렉터리가 없습니다."
        echo "       받아진 것: $(ls -d linux-*/ 2>/dev/null | tr '\n' ' ')"
        exit 1; }
    echo "    소스 디렉터리: $SRCDIR"

    # 아카이브는 최신 ABI 하나만 유지하므로 실행 중인 커널과 다를 수 있다.
    SRCVER="${SRCDIR#linux-xilinx-zynqmp-}"
    RUNVER="${KVER%-xilinx-zynqmp}"
    if [[ "$SRCVER" != "$SRCDIR" && "$SRCVER" != "$RUNVER"* ]]; then
        echo
        echo "    ┌─ 버전 불일치 ────────────────────────────────────────────"
        echo "    │ 소스      : $SRCVER"
        echo "    │ 실행 커널 : $RUNVER"
        echo "    │"
        echo "    │ 5.15.x 내부 API는 안정적이라 대개 그대로 빌드됩니다."
        echo "    │ 컴파일 에러가 나면 커널을 소스에 맞추세요:"
        echo "    │   sudo apt full-upgrade && sudo reboot"
        echo "    └──────────────────────────────────────────────────────────"
        echo
    fi

    popd >/dev/null
    mkdir -p "$WORK/src"
    cp -a "$WORK/dl/$SRCDIR/drivers/media/usb/uvc/." "$WORK/src/"
    echo "    uvc 소스 $(ls "$WORK/src"/*.c | wc -l)개 파일 확보"
else
    echo "    $WORK/src 가 이미 있어 건너뜁니다 (다시 받으려면 이 디렉터리를 지우세요)"
fi

# ---------------------------------------------------------------------------
# 3. 패치 적용
# ---------------------------------------------------------------------------
echo "==> [3/5] bandwidth_cap 패치 적용"
mkdir -p "$WORK/build/drivers/media/usb/uvc"
cp -a "$WORK/src/." "$WORK/build/drivers/media/usb/uvc/"
pushd "$WORK/build" >/dev/null
if patch -p1 --dry-run --silent < "$PATCHFILE"; then
    patch -p1 < "$PATCHFILE"
elif patch -p1 --dry-run --silent --fuzz=3 < "$PATCHFILE"; then
    # 소스 버전이 실행 커널과 달라 문맥이 조금 밀린 경우. 내용은 동일하다.
    echo "    문맥이 약간 다릅니다 (--fuzz=3 으로 적용)"
    patch -p1 --fuzz=3 < "$PATCHFILE"
else
    echo "ERROR: 패치를 적용할 수 없습니다."
    echo "       이 커널의 uvc_video.c 가 5.15 기준과 크게 다릅니다."
    echo "       drivers/media/usb/uvc/ 에서 수동 적용하거나,"
    echo "       sudo apt full-upgrade 로 커널/소스 버전을 맞춘 뒤 재시도하세요."
    exit 1
fi

# 세 군데가 모두 들어갔는지 확인. 하나라도 빠지면 컴파일이 엉뚱하게 실패한다.
for chk in "uvcvideo.h:uvc_bandwidth_cap_param" \
           "uvc_driver.c:bandwidth_cap" \
           "uvc_video.c:uvc_bandwidth_cap_param"; do
    f="drivers/media/usb/uvc/${chk%%:*}"; s="${chk##*:}"
    grep -q "$s" "$f" || { echo "ERROR: $f 에 패치가 반영되지 않았습니다"; exit 1; }
done
echo "    패치 반영 확인 완료"
popd >/dev/null

# ---------------------------------------------------------------------------
# 4. 리네임
#
#    builtin uvcvideo가 이미 /sys/module/uvcvideo 를 점유하고 있어서
#    같은 이름의 모듈은 -EEXIST로 로드가 거부됩니다. 세 곳을 바꿉니다:
#      - 모듈 이름 (Makefile)
#      - usb_driver.name  → /sys/bus/usb/drivers/ 아래 이름
#      - debugfs 루트 디렉터리 → stats를 읽으려면 필요
# ---------------------------------------------------------------------------
echo "==> [4/5] ${MODNAME} 으로 리네임"
cd "$WORK/build/drivers/media/usb/uvc"
sed -i "s/\.name\s*=\s*\"uvcvideo\"/.name\t\t= \"${MODNAME}\"/" uvc_driver.c
sed -i "s/debugfs_create_dir(\"uvcvideo\"/debugfs_create_dir(\"${MODNAME}\"/" uvc_debugfs.c
grep -q "\"${MODNAME}\"" uvc_driver.c || { echo "ERROR: driver name 치환 실패"; exit 1; }

cat > Makefile <<EOF
${MODNAME}-objs := uvc_driver.o uvc_queue.o uvc_v4l2.o uvc_video.o uvc_ctrl.o \\
                   uvc_status.o uvc_isight.o uvc_debugfs.o uvc_metadata.o uvc_entity.o
obj-m += ${MODNAME}.o
EOF

# ---------------------------------------------------------------------------
# 5. 빌드
# ---------------------------------------------------------------------------
echo "==> [5/5] 컴파일"
make -C "/lib/modules/${KVER}/build" M="$PWD" modules -j"$(nproc)"

echo
echo "============================================================"
echo " 빌드 완료: $PWD/${MODNAME}.ko"
echo
echo " 다음: sudo ./uvc-swap.sh on 1024"
echo "============================================================"
