#!/bin/bash
# =============================================================================
# bandwidth_cap 값을 바꿔가며 fps와 uvc errors를 자동 측정
#
#   sudo ./sweep-cap.sh                       기본: MJPG 1280x720@30
#   sudo ./sweep-cap.sh 640 480 MJPG          해상도/포맷 지정
#   CAPS="512 1024 2048" sudo -E ./sweep-cap.sh
#   FPS=30 COUNT=300 sudo -E ./sweep-cap.sh
#
# 카메라 1대 / 2대 동시 두 경우를 모두 돌립니다.
# 판정: fps가 목표의 95% 이상 + errors 0에 가까움 + frames가 기대치만큼.
# =============================================================================
set -uo pipefail

W="${1:-1280}"; H="${2:-720}"; FMT="${3:-MJPG}"; FPS="${FPS:-30}"
COUNT="${COUNT:-300}"
CAPS="${CAPS:-0 512 1024 1536 2048}"
MODNAME="uvcvideo_bwcap"
CAPFILE="/sys/module/${MODNAME}/parameters/bandwidth_cap"
DBG="/sys/kernel/debug/usb/${MODNAME}"

command -v bc >/dev/null || { echo "ERROR: bc 없음.  apt install bc"; exit 1; }
command -v v4l2-ctl >/dev/null || { echo "ERROR: v4l2-ctl 없음.  apt install v4l-utils"; exit 1; }
[[ -f "$CAPFILE" ]] || { echo "ERROR: 패치 모듈 미로드. sudo ./uvc-swap.sh on 먼저"; exit 1; }
mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug 2>/dev/null

# 해당 포맷을 실제로 지원하는 캡처 노드만 추린다.
# C270은 메타데이터 노드(video1/video3)도 만들지만 --list-formats에 안 나와 걸러진다.
DEVS=()
for d in /dev/video*; do
    v4l2-ctl -d "$d" --list-formats 2>/dev/null | grep -q "$FMT" && DEVS+=("$d")
done
[[ ${#DEVS[@]} -ge 1 ]] || { echo "ERROR: $FMT 지원 장치를 못 찾음"; exit 1; }

echo "대상 장치 : ${DEVS[*]}"
echo "설정      : ${W}x${H} ${FMT} @${FPS}fps, ${COUNT}프레임"
echo "cap 목록  : ${CAPS}   (0 = 패치 비활성 = 현재 커널 기본 동작)"
echo

# 한 대를 끝까지 돌리고 "종료코드 소요초" 반환
run_one() {
    local dev="$1" t0 t1 rc
    t0=$(date +%s.%N)
    v4l2-ctl -d "$dev" --set-fmt-video=width=$W,height=$H,pixelformat=$FMT \
        --set-parm=$FPS --stream-mmap --stream-count=$COUNT \
        --stream-to=/dev/null >/dev/null 2>&1
    rc=$?
    t1=$(date +%s.%N)
    echo "$rc $(echo "$t1 - $t0" | bc)"
}

# 모든 스트림 블록을 합산한다.
# head -20 으로 자르면 두 번째 카메라 블록이 잘려 나가, 정작 실패한 쪽을 놓친다.
read_stats() {
    local s
    s=$(cat "$DBG"/*/stats 2>/dev/null)
    echo "$(awk '/^packets:/{n+=$2}END{print n+0}' <<<"$s")" \
         "$(awk '/^errors:/ {n+=$2}END{print n+0}' <<<"$s")" \
         "$(awk '/^empty:/  {n+=$2}END{print n+0}' <<<"$s")" \
         "$(awk '/^frames:/ {n+=$2}END{print n+0}' <<<"$s")"
}

hdr() { printf "%-6s %-7s %-13s %-13s %-9s %-9s %s\n" \
        "cap" "대수" "소요(s)" "fps" "frames" "errors" "판정"; }
sep() { printf '%s\n' "--------------------------------------------------------------------------------"; }

hdr; sep

for cap in $CAPS; do
    echo "$cap" > "$CAPFILE"

    # ---------- 1대 ----------
    r=($(run_one "${DEVS[0]}")); rc=${r[0]}; el=${r[1]}
    st=($(read_stats)); err=${st[1]}; frm=${st[3]}
    fps=$(echo "scale=2; $COUNT / $el" | bc)
    if [[ $rc -ne 0 ]]; then v="FAIL(STREAMON)"
    else v=$(echo "$fps $FPS" | awk '{print ($1>=$2*0.95)?"OK":"LOW"}'); fi
    printf "%-6s %-7s %-13s %-13s %-9s %-9s %s\n" \
           "$cap" "1대" "$el" "$fps" "$frm" "$err" "$v"

    # ---------- 2대 동시 ----------
    if [[ ${#DEVS[@]} -ge 2 ]]; then
        A=$(mktemp); B=$(mktemp)
        run_one "${DEVS[0]}" > "$A" &  pa=$!
        sleep 0.3
        run_one "${DEVS[1]}" > "$B" &  pb=$!
        wait $pa; wait $pb
        ra=($(cat "$A")); rb=($(cat "$B")); rm -f "$A" "$B"
        st=($(read_stats)); err=${st[1]}; frm=${st[3]}
        fa=$(echo "scale=1; $COUNT / ${ra[1]}" | bc)
        fb=$(echo "scale=1; $COUNT / ${rb[1]}" | bc)
        if [[ ${ra[0]} -ne 0 || ${rb[0]} -ne 0 ]]; then
            v="FAIL(-ENOSPC 가능)"
        else
            v=$(echo "$fa $fb $FPS" | awk '{print ($1>=$3*0.95 && $2>=$3*0.95)?"OK":"LOW"}')
        fi
        printf "%-6s %-7s %-13s %-13s %-9s %-9s %s\n" "$cap" "2대" \
               "$(printf '%.0f/%.0f' ${ra[1]} ${rb[1]})" "$fa/$fb" "$frm" "$err" "$v"
    fi
    sep
done

cat <<'EOF'

읽는 법
  cap=0       패치 비활성. 현재 커널 기본 동작이므로 비교 기준선.
  frames      두 스트림 합산. 2대인데 한쪽이 죽으면 기대치의 절반만 찍힌다.
  errors      카메라가 페이로드 헤더에 ERR 비트를 세운 횟수. 0이어야 정상.
  채택        2대 모두 OK인 가장 큰 cap. 클수록 MJPEG 비트레이트 스파이크에 여유.

실패 원인 확인
  dmesg | grep -iE "Capping|alternate|Not enough bandwidth|URB" | tail -20
EOF
