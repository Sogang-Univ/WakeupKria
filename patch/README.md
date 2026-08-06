# uvcvideo-bandwidth-cap

USB 2.0 UVC 카메라 여러 대를 하나의 호스트 버스에서 동시에 구동하기 위한 Linux 커널 패치.

`uvcvideo`에 `bandwidth_cap` 모듈 파라미터를 추가해 `dwMaxPayloadTransferSize`의 상한을
지정한다. 실제 사용량보다 훨씬 큰 대역폭을 요구하는 카메라 때문에 두 번째 장치가
`-ENOSPC`로 실패하는 문제를 해결한다.

**검증 환경** · AMD Kria KV260 / Ubuntu 22.04 (`5.15.0-1027-xilinx-zynqmp`) / Logitech C270 ×2

| | 패치 전 | 패치 후 |
|---|---|---|
| 동시 구동 | 1대 | **2대** |
| 해상도 | 1280×720@30 | 1280×720@30 |
| 대역폭 예약 | 3060 B/µframe | 1024 B/µframe |
| 버스 점유 (2대) | 6420 / 6000 ❌ | **2148 / 6000 (36%)** ✅ |

---

## 배경

USB 2.0 High-Speed 버스는 마이크로프레임당 약 6000 byte-time을 주기적 전송에 할당한다.
UVC 드라이버는 스트리밍 협상 시 카메라가 신고한 `dwMaxPayloadTransferSize`를 그대로 믿고
alternate setting을 고르는데, 상당수 카메라가 **실제 비트레이트와 무관하게 최대값을 신고한다.**

Logitech C270은 MJPEG에서 해상도에 관계없이 항상 3060 B/frame(≈24.5 MB/s)을 요구한다.
실측 사용량은 약 2 MB/s로 요구량의 8%에 불과하다. 두 대를 붙이면 예산을 초과해
두 번째 카메라가 `usb_set_interface()` 단계에서 거부된다.

```
usb 1-1.3: Not enough bandwidth for new device state.
usb 1-1.3: Not enough bandwidth for altsetting 7
```

기존 `UVC_QUIRK_FIX_BANDWIDTH`(quirks=128)는 커널이 압축 프레임 크기를 추정할 수 없어
**압축 포맷에는 적용되지 않는다.** 그래서 별도의 수동 상한이 필요하다.

---

## 요구사항

- USB 2.0 UVC 카메라 2대 이상
- Linux 5.15 계열 (다른 버전은 패치 문맥 조정 필요)
- 타깃 보드에 `linux-headers-$(uname -r)` 설치 가능
- 디스크 여유 5GB 이상 (커널 소스 다운로드용)
- `v4l-utils`, `bc`

> **크로스 컴파일 불가**
> Ubuntu `linux-headers` 패키지의 `scripts/` 바이너리가 타깃 아키텍처로 컴파일되어 있어
> 다른 아키텍처 호스트에서는 모듈 빌드가 실패한다. **타깃 보드에서 네이티브로 빌드할 것.**

---

## 파일 구성

| 파일 | 실행 위치 | 역할 |
|---|---|---|
| `uvc-bandwidth-cap-5.15.patch` | — | 커널 패치 (3개 파일 수정) |
| `phase1-build.sh` | 타깃 | 소스 취득 → 패치 → 리네임 → 모듈 빌드 |
| `uvc-swap.sh` | 타깃 | builtin ↔ 패치 모듈 전환 |
| `sweep-cap.sh` | 타깃 | cap 값별 fps/errors 자동 측정 |
| `install-phase-a.sh` | 타깃 | systemd + udev 영구 설치 |

---

## 빠른 시작

```bash
git clone <repo> && cd uvcvideo-bandwidth-cap
chmod +x *.sh

sudo ./phase1-build.sh ./uvc-bandwidth-cap-5.15.patch   # 빌드 (소스 5~10분 + 컴파일 1~2분)
sudo ./uvc-swap.sh on 1024                              # 적용
sudo ./sweep-cap.sh                                     # 검증
sudo ./install-phase-a.sh 1024                          # 영구화
```

---

## 단계별 설치

### 1. 모듈 빌드

```bash
sudo ./phase1-build.sh ./uvc-bandwidth-cap-5.15.patch
```

수행 내용:

1. `linux-headers-$(uname -r)` 및 빌드 도구 설치
2. `apt-get source --only-source`로 배포판 커널 소스 취득
3. `drivers/media/usb/uvc/`만 추출해 패치 적용 (문맥이 밀리면 `--fuzz=3` 재시도)
4. `uvcvideo_bwcap`으로 리네임 후 out-of-tree 빌드

**`--only-source`가 필수다.** 없으면 apt가 동명의 바이너리 패키지를 보고
`linux-meta-*`(의존성만 든 껍데기)를 받아온다.

**리네임이 필요한 이유** — 대부분의 배포판은 `uvcvideo`를 builtin으로 컴파일한다.
`/sys/module/uvcvideo`가 이미 존재하므로 동명의 모듈은 `-EEXIST`로 로드가 거부된다.
스크립트가 세 군데를 자동 치환한다: 모듈명 / `usb_driver.name` / debugfs 루트.

**바닐라 커널 소스를 쓰지 말 것.** 배포판이 uvcvideo를 수정했을 수 있다.
(검증 환경의 경우 URB 버퍼가 `50×48`, 바닐라는 `5×32`)

### 2. 적용 및 검증

```bash
sudo ./uvc-swap.sh on 1024
sudo ./uvc-swap.sh status
```

```
  1-1.1:1.0      -> uvcvideo_bwcap  (패치됨)
  1-1.3:1.0      -> uvcvideo_bwcap  (패치됨)
```

builtin은 언로드할 수 없으므로, 카메라의 VideoControl 인터페이스를 builtin에서
unbind한 뒤 패치 모듈로 bind한다. VideoStreaming 인터페이스는 드라이버가
`usb_driver_claim_interface()`로 알아서 가져가므로 건드리지 않는다.

```bash
sudo ./sweep-cap.sh                       # MJPG 1280x720@30 기본
sudo ./sweep-cap.sh 640 480 MJPG          # 해상도/포맷 지정
CAPS="512 1024 2048" sudo -E ./sweep-cap.sh
```

`cap=0`은 패치 비활성이므로 비교 기준선이 된다.
**2대 모두 OK인 가장 큰 값**을 채택하면 비트레이트 스파이크에 여유가 크다.

되돌리기: `sudo ./uvc-swap.sh off`

### 3. 영구 설치

```bash
sudo ./install-phase-a.sh 1024
sudo reboot
```

설치되는 것:

| 경로 | 역할 |
|---|---|
| `/lib/modules/<커널>/extra/uvcvideo_bwcap.ko` | 패치 모듈 |
| `/usr/local/sbin/uvc-swap.sh` | 전환 스크립트 (모듈 경로 고정됨) |
| `/etc/systemd/system/uvc-bwcap.service` | 부팅 시 스왑 |
| `/etc/udev/rules.d/99-uvc-bwcap.rules` | 핫플러그 + 노출 + 고정 심볼릭 링크 |

udev 규칙 세 가지:

- **핫플러그** — 카메라가 꽂히면 builtin이 먼저 잡으므로 systemd 유닛을 다시 트리거한다.
  udev `RUN`은 타임아웃이 있어 `systemctl --no-block`으로 넘긴다.
- **노출** — `exposure_dynamic_framerate=0`. 켜져 있으면 저조도에서 프레임레이트가 절반으로 떨어진다.
- **고정 이름** — 동일 모델 카메라는 VID/PID가 같고 iSerial도 없어 `/dev/videoN` 번호가
  부팅마다 바뀐다. 포트 경로 기준으로 `/dev/cam0`, `/dev/cam1` 심볼릭 링크를 만든다.
  (원본 `/dev/videoN`도 그대로 유지된다)

제거: `sudo ./install-phase-a.sh --uninstall`

---

## 사용

```bash
# GStreamer
gst-launch-1.0 v4l2src device=/dev/cam0 ! \
  "image/jpeg,width=1280,height=720,framerate=30/1" ! jpegdec ! fakesink

# OpenCV — 정수 인덱스 대신 경로를 쓸 것
cap = cv2.VideoCapture("/dev/cam0")
```

런타임 조정 (파라미터가 `0644`라 재로드 불필요, 다음 `STREAMON`부터 적용):

```bash
echo 1536 | sudo tee /sys/module/uvcvideo_bwcap/parameters/bandwidth_cap
```

---

## 문제 해결

### 두 번째 카메라가 여전히 실패

```bash
dmesg | grep -iE "Capping|alternate|Not enough bandwidth"
```

`Capping bandwidth 3060 -> 1024`가 없다면 패치 모듈이 아니라 builtin이 처리 중이다.
`uvc-swap.sh status`로 바인딩을 확인한다.

### fps가 목표의 절반

`exposure_dynamic_framerate`가 켜져 있다. 저조도에서 카메라가 노출 시간을 늘리려고
프레임 주기를 스스로 두 배로 늘린다.

```bash
v4l2-ctl -d /dev/cam0 -L | grep -i exposure
v4l2-ctl -d /dev/cam0 -c exposure_dynamic_framerate=0
```

### 프레임이 깨지거나 대량 폐기

```bash
cat /sys/kernel/debug/usb/uvcvideo_bwcap/*/stats
```

`errors`는 USB 전송 오류가 아니라 **카메라가 페이로드 헤더에 ERR 비트를 세운 횟수**다.
값이 크면 카메라가 그 데이터율을 감당하지 못하는 것이므로, cap을 올려도 해결되지 않는다.
포맷(비압축 → MJPEG)이나 해상도를 조정해야 한다.

### bind 실패

builtin이 인터페이스를 완전히 놓기 전에 bind를 시도한 경우다.
카메라를 뽑았다 다시 꽂거나 `uvc-swap.sh on`을 재실행한다.

### 커널 업그레이드 후 동작 중단

`.ko`는 특정 커널 버전 전용이다. 커널이 올라가면 로드에 실패하고 builtin으로 되돌아간다.

```bash
sudo apt-mark hold linux-image-xilinx-zynqmp \
                   linux-headers-xilinx-zynqmp linux-xilinx-zynqmp
```

업그레이드가 필요하면 새 커널로 부팅 후 `phase1-build.sh` → `install-phase-a.sh`를 재실행한다.

---

## 패치 상세

```c
/* uvc_fixup_video_ctrl() 말미 */
if (uvc_bandwidth_cap_param && stream->intf->num_altsetting > 1 &&
    ctrl->dwMaxPayloadTransferSize > uvc_bandwidth_cap_param) {
        uvc_dbg(stream->dev, VIDEO, "Capping bandwidth %u -> %u B/frame\n",
                ctrl->dwMaxPayloadTransferSize, uvc_bandwidth_cap_param);
        ctrl->dwMaxPayloadTransferSize = uvc_bandwidth_cap_param;
}
```

수정 파일: `uvcvideo.h`(extern 선언) / `uvc_driver.c`(변수 + `module_param_named`) /
`uvc_video.c`(적용 로직). 기본값 0은 비활성이므로 기존 동작에 영향이 없다.

### cap 값 선택 기준

| cap | alt | 예약 대역폭 | 2대 버스 점유 |
|---|---|---|---|
| 512 | 2 | 4.10 MB/s | 1148 / 6000 |
| **1024** | **3** | **8.19 MB/s** | **2148 / 6000** |
| 1536 | 4 | 12.3 MB/s | 3272 / 6000 |
| 2048 | 5 | 16.4 MB/s | 4296 / 6000 |

MJPEG는 프레임 크기가 가변이라 복잡한 장면에서 순간 비트레이트가 튄다.
실측 평균의 3~4배 여유를 두는 것이 안전하다.

---

## 진단 도구

```bash
# alt 선택 과정 로깅 (UVC_DBG_VIDEO | UVC_DBG_STATS)
echo 3072 | sudo tee /sys/module/uvcvideo_bwcap/parameters/trace
dmesg | grep -iE "requested|alternate|Capping"

# 스트림 품질
cat /sys/kernel/debug/usb/uvcvideo_bwcap/*/stats

# fps 실측
time v4l2-ctl -d /dev/cam0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG \
  --set-parm=30 --stream-mmap --stream-count=300 --stream-to=/dev/null

# 버스 토폴로지 — 포트가 실제로 분리되어 있는지 확인
lsusb -t
```

---

## 라이선스

패치 부분은 Linux 커널과 동일하게 GPL-2.0. 스크립트는 자유롭게 사용 가능.
