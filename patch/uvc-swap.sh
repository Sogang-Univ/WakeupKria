#!/bin/bash
# =============================================================================
# builtin uvcvideo <-> 패치된 uvcvideo_bwcap 전환
#
#   sudo ./uvc-swap.sh on [cap]    패치 모듈로 전환 (cap 기본값 1024)
#   sudo ./uvc-swap.sh off         builtin으로 복귀
#   sudo ./uvc-swap.sh cap <값>    로드된 상태에서 cap만 변경
#   sudo ./uvc-swap.sh status      현재 상태 확인
#
# builtin은 언로드할 수 없으므로, 카메라 인터페이스를 builtin에서 unbind한 뒤
# 패치 모듈로 bind하는 방식입니다. 부팅 때마다 builtin이 먼저 잡으므로
# 재부팅 후에는 다시 'on'을 실행해야 합니다.
# =============================================================================
set -euo pipefail

MODNAME="uvcvideo_bwcap"
KO="${KO:-$HOME/uvc-bwcap/build/drivers/media/usb/uvc/${MODNAME}.ko}"
BUILTIN_DRV="/sys/bus/usb/drivers/uvcvideo"
PATCHED_DRV="/sys/bus/usb/drivers/${MODNAME}"

# ---------------------------------------------------------------------------
# 바인딩 상태와 무관하게 sysfs에서 UVC VideoControl 인터페이스를 찾는다.
#
# 드라이버 디렉터리를 뒤지면 안 된다 — unbind 직후에는 목록이 비어 있어서
# 복구 경로가 동작하지 않는다.
#
# bind/unbind 대상은 VideoControl(class 0e / subclass 01)뿐이다.
# VideoStreaming(subclass 02)은 uvcvideo가 usb_driver_claim_interface()로
# 스스로 가져가므로 직접 건드리면 안 된다.
# ---------------------------------------------------------------------------
uvc_ctrl_interfaces() {
    local i
    for i in /sys/bus/usb/devices/*-*:*; do
        [[ -f "$i/bInterfaceClass" ]] || continue
        [[ "$(cat "$i/bInterfaceClass")"    == "0e" ]] || continue
        [[ "$(cat "$i/bInterfaceSubClass")" == "01" ]] || continue
        basename "$i"
    done
}

case "${1:-status}" in
on)
    CAP="${2:-1024}"
    [[ -f "$KO" ]] || { echo "ERROR: 모듈 없음: $KO  (phase1-build.sh 먼저 실행)"; exit 1; }

    IFACES=$(uvc_ctrl_interfaces)
    [[ -n "$IFACES" ]] || { echo "ERROR: UVC 카메라를 찾을 수 없습니다"; exit 1; }
    echo "==> 대상 인터페이스: $(echo $IFACES | tr '\n' ' ')"

    echo "==> builtin에서 분리"
    for i in $IFACES; do
        [[ -e "$BUILTIN_DRV/$i" ]] || continue
        echo "$i" > "$BUILTIN_DRV/unbind" && echo "    unbind $i"
    done

    if ! lsmod | grep -q "^${MODNAME}"; then
        echo "==> 모듈 로드 (bandwidth_cap=${CAP})"
        insmod "$KO" bandwidth_cap="$CAP" quirks=128 trace=3072
    else
        echo "$CAP" > "/sys/module/${MODNAME}/parameters/bandwidth_cap"
        echo "==> 이미 로드됨. bandwidth_cap=${CAP} 로 갱신"
    fi

    echo "==> 패치 모듈로 연결"
    sleep 1
    for i in $IFACES; do
        [[ -e "$PATCHED_DRV/$i" ]] && continue
        if echo "$i" > "$PATCHED_DRV/bind" 2>/dev/null; then
            echo "    bind $i"
        else
            echo "    WARN: $i bind 실패"
        fi
    done

    echo
    echo "==> 결과"
    v4l2-ctl --list-devices 2>/dev/null | head -20 || ls /dev/video* 2>/dev/null
    ;;

off)
    IFACES=$(uvc_ctrl_interfaces)
    echo "==> 패치 모듈에서 분리"
    for i in $IFACES; do
        [[ -e "$PATCHED_DRV/$i" ]] || continue
        echo "$i" > "$PATCHED_DRV/unbind" && echo "    unbind $i" || true
    done

    if lsmod | grep -q "^${MODNAME}"; then
        rmmod "$MODNAME" && echo "==> ${MODNAME} 언로드"
    fi

    # rmmod 후 sysfs를 다시 읽는다. 목록이 드라이버 상태에 의존하지 않으므로
    # 여기서 비어버리는 문제가 없다.
    echo "==> builtin으로 복귀"
    sleep 1
    for i in $(uvc_ctrl_interfaces); do
        [[ -e "$BUILTIN_DRV/$i" ]] && continue
        if echo "$i" > "$BUILTIN_DRV/bind" 2>/dev/null; then
            echo "    bind $i"
        else
            echo "    WARN: $i bind 실패 — 카메라를 뽑았다 꽂으면 복구됩니다"
        fi
    done
    ;;

cap)
    CAP="${2:?사용법: uvc-swap.sh cap <바이트값>}"
    [[ -f "/sys/module/${MODNAME}/parameters/bandwidth_cap" ]] \
      || { echo "ERROR: ${MODNAME} 미로드"; exit 1; }
    echo "$CAP" > "/sys/module/${MODNAME}/parameters/bandwidth_cap"
    echo "==> bandwidth_cap=${CAP}. 다음 STREAMON부터 적용됩니다."
    ;;

status)
    echo "== UVC VideoControl 인터페이스 =="
    for i in $(uvc_ctrl_interfaces); do
        if   [[ -e "$PATCHED_DRV/$i" ]]; then drv="${MODNAME}  (패치됨)"
        elif [[ -e "$BUILTIN_DRV/$i" ]]; then drv="uvcvideo  (builtin)"
        else                                  drv="(바인딩 안 됨)"
        fi
        printf "  %-14s -> %s\n" "$i" "$drv"
    done
    echo
    if [[ -d "/sys/module/${MODNAME}/parameters" ]]; then
        echo "== ${MODNAME} 파라미터 =="
        for p in bandwidth_cap quirks nodrop trace; do
            [[ -f "/sys/module/${MODNAME}/parameters/$p" ]] \
              && printf "  %-14s = %s\n" "$p" "$(cat "/sys/module/${MODNAME}/parameters/$p")"
        done
    else
        echo "== ${MODNAME} 미로드 (builtin 사용 중) =="
        echo "  quirks         = $(cat /sys/module/uvcvideo/parameters/quirks 2>/dev/null)"
    fi
    echo
    echo "== 비디오 노드 =="
    v4l2-ctl --list-devices 2>/dev/null | head -20 || ls /dev/video* 2>/dev/null
    ;;

*)
    sed -n '2,15p' "$0"; exit 1 ;;
esac
