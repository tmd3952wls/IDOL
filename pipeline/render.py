"""구성안(plan.json) -> 완성 쇼츠 mp4.

사용: python pipeline/render.py work/<이름>/plan.json [--prepare]
  --prepare: 클립을 받고 대사를 인식해 work/<이름>/captions.json 만 만든다(자막 교정용).
결과: output/<이름>/<이름>.mp4, meta.txt, contact.jpg (검수용 장면 모음)

화면(1080x1920): 검정 배경 / 상단 노란 키워드 제목 / 가운데 원본 클립(1080x888) / 하단 댓글 한 줄
- 구간 안에서는 말이 나오는 부분만 남겨 빠르게 이어 붙인다(컷컷컷).
- 자막 한 줄마다 컷을 나누고, 컷마다 얼굴을 찾아 크게 자른다(확대 정도를 컷마다 바꾼다).
- 자른 화면에 방송 문구가 남으면 글자를 찾아 블러로 가린다.
- 하단 댓글은 원본 영상의 인기 댓글을 1.5초마다 바꾼다.
- 방송 자막이 잘려 나가므로 대사를 자막으로 다시 넣는다(상황별 색·애니메이션, 배민 주아체).
"""
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import speech

ROOT = Path(__file__).resolve().parent.parent
W, H = 1080, 1920
CLIP_Y, CLIP_H = 547, 888
TITLE_CY = 420
STRIP_H = 170
FONT = ROOT / "assets/fonts/Jua-Regular.ttf"          # 배민 주아
FACE_MODEL = ROOT / "assets/models/yunet.onnx"
TEXT_MODEL = ROOT / "assets/models/text_det.onnx"
COMMENT_EVERY = 1.5                 # 하단 댓글 교체 간격(초)
STYLE_VERSION = "v14"  # 레이아웃을 바꾸면 올린다 (기존 구간을 모두 다시 만들게 됨)
TITLE_COLOR = "#ffe600"
AVATAR_COLORS = ["#6fbf73", "#f2a03d", "#5b7fd1", "#d96a8a", "#8e6fd1", "#49b3b3"]
PUSH_PER_FRAME = 0.0006             # 조각 안에서 천천히 밀고 들어가는 속도
FACE_FILLS = [0.52, 0.74, 0.60, 0.84]          # 얼굴 높이가 화면 높이에서 차지할 비율(컷마다 번갈아)
SAFE_TOP, SAFE_BOTTOM = 0.21, 0.69  # 방송 자막·로고가 주로 놓이는 위아래 띠는 화면에 넣지 않는다
CAPTION_Y = CLIP_Y + CLIP_H - 46    # 자막 아랫변 위치

# 자막 스타일: 글자색(ASS는 BBGGRR 순서), 크기, 애니메이션 태그
CAPTION_STYLES = {
    # 일반 대사: 애니메이션 없이 바로 뜬다(색만 다름)
    "normal":   ("FFFFFF", 80, ""),
    "question": ("FFD66F", 80, ""),
    "aside":    ("D8D8D8", 68, ""),
    # 포인트 자막: 상황에 맞는 애니메이션
    "shout":    ("00E6FF", 104, r"\fscx190\fscy190\frz-5\t(0,90,\fscx100\fscy100\frz4)"
                               r"\t(90,160,\frz-3)\t(160,230,\frz0)"),                             # 큰 소리·펀치라인: 쾅 찍히며 흔들림
    "laugh":    ("B87AFF", 90, r"\t(0,120,\fscx118\fscy118)\t(120,240,\fscx100\fscy100)"
                               r"\t(240,360,\fscx114\fscy114)\t(360,480,\fscx100\fscy100)"),       # 웃음·애교: 통통 튐
    "point":    ("4DFFA6", 92, r"\fscx40\fscy140\t(0,130,\fscx100\fscy100)"),                      # 핵심 단어: 옆으로 쫙 펼쳐짐
}


def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"명령 실패: {' '.join(map(str, cmd[:6]))} ...\n{r.stderr[-900:]}")
    return r


def fit_font(text, size, max_w):
    while size > 40:
        f = ImageFont.truetype(str(FONT), size)
        if f.getbbox(text)[2] <= max_w:
            return f
        size -= 6
    return ImageFont.truetype(str(FONT), size)


_cmap = None
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def safe_text(text):
    """주아체에 없는 글자(녴, 냨 같은 드문 받침 글자)는 받침을 떼어 풀어 쓴다: 녴 -> 녜ㅋ"""
    global _cmap
    if _cmap is None:
        from fontTools.ttLib import TTFont
        _cmap = TTFont(str(FONT)).getBestCmap()
    out = []
    for ch in text:
        code = ord(ch)
        if code in _cmap or ch.isspace():
            out.append(ch)
        elif 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28:
            jong = (code - 0xAC00) % 28
            out.append(chr(code - jong) + _JONG[jong])
    return "".join(out)


def tidy_comment(text):
    text = safe_text(text)
    text = re.sub(r"([ㅋㅎㅠㅜ!?.])\1{5,}", lambda m: m.group(1) * 5, text)
    return re.sub(r"\s+", " ", text).strip()


def wrap(draw, text, font, max_w, max_lines=2):
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > max_w:
            lines.append(cur)
            cur = ch.lstrip()
        else:
            cur += ch
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    return lines


def make_frame(seg, path):
    img = Image.new("RGB", (W, H), "black")
    d = ImageDraw.Draw(img)
    title = seg["keyword"].rstrip(";") + ";"
    d.text((W // 2, TITLE_CY), title, font=fit_font(title, 200, W - 160), fill=TITLE_COLOR, anchor="mm")
    img.save(path)


def make_comment(comment, path, rng):
    """하단 댓글 한 장(1080xSTRIP_H). 프로필·닉네임은 가린 형태(흐린 원 + 흐린 막대)로만 표시"""
    img = Image.new("RGB", (W, STRIP_H), "black")
    bd = ImageDraw.Draw(img)
    cy = STRIP_H // 2
    bd.ellipse([120, cy - 46, 212, cy + 46], fill=rng.choice(AVATAR_COLORS))
    bd.rounded_rectangle([250, cy - 52, 250 + rng.randint(150, 240), cy - 24], radius=8, fill="#8a8a8a")
    img = img.filter(ImageFilter.GaussianBlur(9))
    d = ImageDraw.Draw(img)
    for size in (42, 38, 34):  # 한 줄에 들어가면 한 줄로, 안 되면 두 줄
        cf = ImageFont.truetype(str(FONT), size)
        lines = wrap(d, comment, cf, W - 250 - 90)
        if len(lines) == 1:
            break
    else:
        cf = ImageFont.truetype(str(FONT), 42)
        lines = wrap(d, comment, cf, W - 250 - 90)
    ty = cy - 8 if len(lines) == 1 else cy - 14
    for i, ln in enumerate(lines):
        d.text((250, ty + i * 50), ln, font=cf, fill="#f2f2f2")
    img.save(path)


def fetch_clip(seg, cache):
    out = cache / f"{seg['video']}_{seg['start']}_{seg['end']}.mp4"
    if out.exists() and out.stat().st_size > 10_000:
        return out
    cmd = ["yt-dlp", "--compat-options", "no-certifi", "--no-warnings", "-q",
           "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
           "--download-sections", f"*{seg['start']}-{seg['end']}", "--force-keyframes-at-cuts",
           "--merge-output-format", "mp4", "-o", str(out),
           f"https://www.youtube.com/watch?v={seg['video']}"]
    try:
        run(cmd)
    except RuntimeError:  # 일시적 다운로드 실패는 한 번 더 시도
        run(cmd)
    return out


def probe(clip):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height:format=duration", "-of", "json", str(clip)])
    j = json.loads(r.stdout)
    return j["streams"][0]["width"], j["streams"][0]["height"], float(j["format"]["duration"])


def scene_cuts(clip):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(clip), "-vf", "select='gt(scene,0.32)',showinfo",
                        "-an", "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8",
                       errors="ignore")
    return [round(float(x), 2) for x in re.findall(r"pts_time:([\d.]+)", r.stderr)]


_detector = None
_textdet = None
_frames = {}
FW = 1280


def frame_at(clip, t, sw, sh):
    """t초 프레임(BGR, 가로 1280)."""
    key = (str(clip), round(t, 2))
    if key not in _frames:
        fh = int(round(FW * sh / sw / 2)) * 2
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{max(t, 0):.2f}", "-i", str(clip),
                              "-frames:v", "1", "-vf", f"scale={FW}:{fh}", "-f", "rawvideo",
                              "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
        _frames[key] = (np.frombuffer(raw, np.uint8).reshape(fh, FW, 3)
                        if len(raw) == FW * fh * 3 else None)
    return _frames[key]


def faces_in(img, sw):
    """얼굴들 [(cx, cy, h)] — 원본 픽셀 기준."""
    global _detector
    small = cv2.resize(img, (640, 360))
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (640, 360), 0.75)
    _, found = _detector.detect(small)
    if found is None:
        return []
    kx, ky = sw / 640, sw * img.shape[0] / img.shape[1] / 360
    return [((f[0] + f[2] / 2) * kx, (f[1] + f[3] / 2) * ky, f[3] * ky) for f in found]


_textcache = {}


def frame_text(img, sw):
    """프레임 전체에서 방송 문구(자막·태그·로고) 위치 [(x0, y0, x1, y1)] — 원본 픽셀 기준.
    (자른 뒤에 찾으면 가장자리에 반쯤 걸린 글자를 놓치므로, 자르기 전 전체 화면에서 찾는다)"""
    global _textdet
    key = id(img)
    if key in _textcache:
        return _textcache[key]
    if _textdet is None:
        _textdet = cv2.dnn.TextDetectionModel_DB(str(TEXT_MODEL))
        _textdet.setBinaryThreshold(0.25)
        _textdet.setPolygonThreshold(0.5)
        _textdet.setMaxCandidates(100)
        _textdet.setUnclipRatio(1.8)
        _textdet.setInputParams(1 / 255.0, (1280, 736), (122.68, 116.67, 104.0))
    polys, _ = _textdet.detect(cv2.resize(img, (1280, 736)))
    kx, ky = sw / 1280, sw * img.shape[0] / img.shape[1] / 736
    out = []
    for poly in polys:
        pts = np.array(poly, dtype=float)
        out.append((pts[:, 0].min() * kx, pts[:, 1].min() * ky, pts[:, 0].max() * kx, pts[:, 1].max() * ky))
    _textcache[key] = out
    return out


def text_boxes(img, crop, sw, face):
    """자른 영역에 걸치는 방송 문구의 위치 [(x0, y0, x1, y1)] — 자른 영역 기준 원본 픽셀."""
    cw, ch, cx, cy = crop
    out = []
    for x0, y0, x1, y1 in frame_text(img, sw):
        px, py = (x1 - x0) * 0.06 + 8, (y1 - y0) * 0.3 + 8
        x0, y0, x1, y1 = x0 - px - cx, y0 - py - cy, x1 + px - cx, y1 + py - cy
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(cw, x1), min(ch, y1)
        if x1 - x0 < 16 or y1 - y0 < 10:
            continue                                    # 자른 화면 밖
        if face:   # 얼굴 위는 건드리지 않는다
            fx, fy, fh = face[0] - cx, face[1] - cy, face[2]
            if fx - fh * 0.3 < (x0 + x1) / 2 < fx + fh * 0.3 and fy - fh * 0.3 < (y0 + y1) / 2 < fy + fh * 0.5:
                continue
        out.append((x0, y0, x1, y1))
    return out


def merge_boxes(boxes, cw, ch, limit=6):
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    out = []
    for x0, y0, x1, y1 in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:limit]:
        x, y = int(x0) // 2 * 2, int(y0) // 2 * 2
        w, h = int(min(x1, cw) - x) // 2 * 2, int(min(y1, ch) - y) // 2 * 2
        if w >= 16 and h >= 16:
            out.append((x, y, w, h))
    return out


def face_crop(clip, a, b, sw, sh, focus, fill):
    """조각(a~b초)에서 가장 큰 얼굴만 화면에 꽉 차도록 자를 영역 (w, h, x, y)과, 그 안에 남은 방송 문구 위치."""
    ratio = W / CLIP_H
    times = sorted({round(a + 0.08 + (b - a - 0.16) * i / 4, 2) for i in range(5)})
    imgs = [im for im in (frame_at(clip, t, sw, sh) for t in times) if im is not None]
    picks = []
    for im in imgs:
        fs = faces_in(im, sw)
        if fs:
            picks.append(max(fs, key=lambda f: f[2]))
    face = None
    if not picks:   # 얼굴이 없으면 자막 띠를 피한 가운데 부분만
        ch = sh * (SAFE_BOTTOM - SAFE_TOP)
        cw = ch * ratio
        crop = (int(cw) // 2 * 2, int(ch) // 2 * 2, int((sw - cw) * focus), int(sh * SAFE_TOP))
    else:
        cx, cy, fh = (float(np.median([p[i] for p in picks])) for i in range(3))
        face = (cx, cy, fh)
        # 위쪽 프로그램 태그와 아래쪽 방송 자막 띠 사이에서만 자른다(얼굴 자체는 잘리지 않게)
        top = min(sh * SAFE_TOP, cy - fh * 0.55)
        bottom = max(sh * SAFE_BOTTOM, cy + fh * 0.58)
        ch = min(fh / fill, bottom - top, sh * 0.6)
        ch = max(ch, fh * 1.15, sh * 0.22)
        cw = ch * ratio
        y = min(max(cy - ch * 0.46, top), max(top, bottom - ch))
        x = cx - cw / 2
        if y < sh * SAFE_TOP:   # 얼굴이 화면 위쪽이라 태그 높이를 못 피하면, 좌상단 태그·우상단 로고를 옆으로 피한다
            lo = min(sw * 0.41, cx - fh * 0.5)
            hi = max(sw * 0.80, cx + fh * 0.5) - cw
            x = max(lo, min(x, hi)) if hi >= lo else lo
        x = min(max(x, 0), sw - cw)
        y = min(max(y, 0), sh - ch)
        crop = (int(cw) // 2 * 2, int(ch) // 2 * 2, int(x), int(y))
    # 그래도 남은 방송 문구는 블러로 가린다
    found = []
    for im in imgs:
        found += text_boxes(im, crop, sw, face)
    return crop, merge_boxes(found, crop[0], crop[1])


def write_ass(lines, parts, path):
    """클립 시각 기준 자막을 점프컷 이후 시각으로 옮겨 ASS 파일로 쓴다."""
    def out_time(t):
        acc = 0.0
        for a, b in parts:
            if t < a:
                return acc
            if t <= b:
                return acc + (t - a)
            acc += b - a
        return acc

    def ts(x):
        return f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"

    total = sum(b - a for a, b in parts)
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 2\n"
            "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: D,Jua,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,1,7,3,2,20,20,20,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    ev = []
    timed = [(out_time(c["t0"]), out_time(c["t1"]), c) for c in lines if c.get("text")]
    for i, (s, e, c) in enumerate(timed):
        e = max(e + 0.1, s + 0.32)                      # 말이 끝나면 바로 사라진다
        if i + 1 < len(timed):
            e = min(e, timed[i + 1][0])                  # 다음 자막과 겹치지 않게
        e = min(e, total)
        if e - s < 0.15:
            continue
        color, size, anim = CAPTION_STYLES.get(c.get("style"), CAPTION_STYLES["normal"])
        anim = anim.replace("{y2}", str(CAPTION_Y + 50)).replace("{y}", str(CAPTION_Y))
        pos = "" if r"\move" in anim else rf"\pos(540,{CAPTION_Y})"
        text = safe_text(c["text"]).replace("{", "(").replace("}", ")")
        ev.append(f"Dialogue: 0,{ts(s)},{ts(e)},D,,0,0,0,,{{\\an2{pos}\\fs{size}\\c&H{color}&{anim}}}{text}")
    Path(path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8-sig")


def compose(seg, clip, frame, lines, ws, comments, out, rng):
    sw, sh, src_dur = probe(clip)
    # 자막이 남아 있는 부분만 영상에 남긴다(자막을 지우면 그 부분도 잘린다)
    spoken = [(c["t0"], c["t1"], c["text"]) for c in lines] or ws
    ranges = speech.keep_ranges(spoken, src_dur) if seg.get("jumpcut", True) else [(0.0, src_dur)]
    # 자막 한 줄이 시작될 때마다, 그리고 장면이 바뀔 때마다 컷을 나눈다(한 마디 = 한 컷)
    edges = sorted(set(scene_cuts(clip)) | {round(c["t0"] - 0.04, 2) for c in lines})
    parts = speech.chunks(ranges, edges)
    n = len(parts)
    total = sum(b - a for a, b in parts)
    focus = float(seg.get("focus", 0.5))
    ass = out.with_suffix(".ass")
    write_ass(lines, parts, ass)

    blurred = 0
    g = "[1:v]fps=30,split=" + str(n) + "".join(f"[b{k}]" for k in range(n)) + ";"
    g += f"[1:a]asplit={n}" + "".join(f"[s{k}]" for k in range(n)) + ";"
    for k, (a, b) in enumerate(parts):
        (cw, ch, cx, cy), boxes = face_crop(clip, a, b, sw, sh, focus, FACE_FILLS[k % len(FACE_FILLS)])
        blurred += bool(boxes)
        g += f"[b{k}]trim={a}:{b},setpts=PTS-STARTPTS,crop={cw}:{ch}:{cx}:{cy}"
        for j, (bx, by, bw, bh) in enumerate(boxes):
            r = max(2, min(bw, bh) // 5)
            g += (f",split[p{k}_{j}][q{k}_{j}];[q{k}_{j}]crop={bw}:{bh}:{bx}:{by},"
                  f"boxblur={r}:3:{max(1, r // 2)}:3[r{k}_{j}];[p{k}_{j}][r{k}_{j}]overlay={bx}:{by}")
        g += (f",scale={W * 2}:{CLIP_H * 2},setsar=1,"
              f"zoompan=z='1+{PUSH_PER_FRAME}*on':x='iw/2-(iw/zoom/2)':y='max(0,ih*0.44-(ih/zoom/2))':"
              f"d=1:s={W}x{CLIP_H}:fps=30[v{k}];")
        d = b - a
        g += (f"[s{k}]atrim={a}:{b},asetpts=PTS-STARTPTS,afade=t=in:d=0.02,"
              f"afade=t=out:st={max(0, d - 0.03):.3f}:d=0.03[a{k}];")
    g += "".join(f"[v{k}][a{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=1[cv][ca];"
    g += f"[0:v][cv]overlay=0:{CLIP_Y}[c0];"

    # 하단 댓글: COMMENT_EVERY초마다 다음 댓글로
    slots = max(1, int(total / COMMENT_EVERY + 0.999))
    inputs = []
    for j in range(slots):
        cpath = out.with_name(f"{out.stem}_c{j}.png")
        make_comment(comments[j % len(comments)], cpath, rng)
        inputs += ["-loop", "1", "-framerate", "30", "-i", str(cpath)]
        t0 = j * COMMENT_EVERY
        t1 = (j + 1) * COMMENT_EVERY if j + 1 < slots else total + 1
        g += (f"[c{j}][{2 + j}:v]overlay=0:{CLIP_Y + CLIP_H}:enable='between(t,{t0:.2f},{t1:.2f})'"
              f"[c{j + 1}];")
    rel = ass.relative_to(ROOT).as_posix()
    g += (f"[c{slots}]subtitles=f='{rel}':fontsdir='assets/fonts',format=yuv420p[vo];"
          "[ca]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[ao]")
    script = out.with_suffix(".filter")
    script.write_text(g, encoding="utf-8")
    run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", str(frame),
         "-i", str(clip), *inputs, "-/filter_complex", str(script),
         "-map", "[vo]", "-map", "[ao]", "-t", f"{total:.3f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-c:a", "aac", "-b:a", "160k", "-ac", "2",
         "-video_track_timescale", "30000", str(out)], cwd=ROOT)
    return total, n, src_dur, blurred, slots


def comment_pool(work):
    """영상별 인기 댓글 목록(좋아요 많은 순). collect.py가 만든 candidates.json에서 가져온다."""
    path = work / "candidates.json"
    pool = {}
    if not path.exists():
        return pool
    for v in json.loads(path.read_text(encoding="utf-8")):
        seen, items = set(), []
        raw = list(v.get("top_comments", [])) + [c for m in v.get("moments", []) for c in m.get("comments", [])]
        for c in sorted(raw, key=lambda c: -c.get("likes", 0)):
            text = tidy_comment(c["text"])
            if 4 <= len(text) <= 40 and text not in seen:
                seen.add(text)
                items.append(text)
        pool[v["id"]] = items
    return pool


def seg_id(seg):
    return f"{seg['video']}_{seg['start']}_{seg['end']}"


def save_caps(path, caps):
    """자막 한 줄을 파일 한 줄로 써서 사람이 고치기 쉽게 한다."""
    blocks = []
    for sid, lines in caps.items():
        rows = ",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in lines)
        blocks.append(f' "{sid}": [\n{rows}\n ]')
    Path(path).write_text("{\n" + ",\n".join(blocks) + "\n}\n", encoding="utf-8")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prepare_only = "--prepare" in sys.argv
    plan_path = Path(args[0]).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    name = plan["name"]
    work = plan_path.parent
    cache, parts = work / "clips", work / "parts"
    out_dir = ROOT / "output" / name
    for p in (cache, parts, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    segs = plan["segments"]
    hint = plan.get("hint") or f"{name}이(가) 나오는 예능 프로그램."

    # 1) 클립 받기 + 대사 인식(단어 타이밍은 words.json, 자막은 captions.json에 저장·재사용)
    words_path, caps_path = work / "words.json", work / "captions.json"
    all_words = json.loads(words_path.read_text(encoding="utf-8")) if words_path.exists() else {}
    all_caps = json.loads(caps_path.read_text(encoding="utf-8")) if caps_path.exists() else {}
    clips, snapped = {}, {}
    for seg in segs:
        sid = seg_id(seg)
        try:
            clips[sid] = fetch_clip(seg, cache)
        except Exception as e:  # noqa: BLE001
            print(f"  받기 실패 {seg['keyword']}: {str(e)[:300]}", file=sys.stderr)
            continue
        if sid not in all_words:
            all_words[sid] = speech.words(clips[sid], hint)
            words_path.write_text(json.dumps(all_words, ensure_ascii=False), encoding="utf-8")
        audio = speech.load_audio(clips[sid])
        snapped[sid] = speech.snap_onsets([tuple(w) for w in all_words[sid]], audio)
        if sid not in all_caps:
            all_caps[sid] = speech.caption_lines(snapped[sid], audio)
    ordered = {seg_id(s): all_caps[seg_id(s)] for s in segs if seg_id(s) in all_caps}
    save_caps(caps_path, ordered)
    if prepare_only:
        print(f"자막 초안: {caps_path}")
        return

    # 2) 구간별 렌더
    pool, used = comment_pool(work), {}
    part_files, starts, t = [], [], 0.0
    for i, seg in enumerate(segs):
        sid = seg_id(seg)
        if sid not in clips:
            continue
        lines = all_caps.get(sid, [])
        # 댓글: 구간에 지정한 댓글을 먼저, 이어서 같은 영상의 인기 댓글을 순서대로(다른 구간과 겹치지 않게)
        first = tidy_comment(seg["comment"])
        rest = [c for c in pool.get(seg["video"], []) if c != first]
        off = used.get(seg["video"], 0) % max(1, len(rest))
        rest = rest[off:] + rest[:off]
        used[seg["video"]] = used.get(seg["video"], 0) + 9
        comments = [first] + rest
        sig = STYLE_VERSION + json.dumps([seg, lines, comments[:14]], sort_keys=True, ensure_ascii=False)
        key = hashlib.md5(sig.encode()).hexdigest()[:8]
        part = parts / f"{i + 1:02d}_{key}.mp4"
        if not part.exists():
            try:
                frame = parts / f"{i + 1:02d}_{key}.png"
                make_frame(seg, frame)
                total, nr, src, nb, nc = compose(seg, clips[sid], frame, lines, snapped.get(sid, []),
                                                 comments, part, random.Random(key))
                print(f"[{i + 1}/{len(segs)}] {seg['keyword']}: 원본 {src:.1f}초 -> {total:.1f}초 "
                      f"(컷 {nr}개, 문구 블러 {nb}컷, 자막 {len(lines)}줄, 댓글 {nc}개)")
            except Exception as e:  # noqa: BLE001
                print(f"  건너뜀 {seg['keyword']}: {str(e)[:500]}", file=sys.stderr)
                continue
        d = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                       str(part)]).stdout.strip())
        part_files.append(part)
        starts.append(t)
        t += d

    if not part_files:
        sys.exit("렌더된 구간이 없습니다.")
    lst = parts / "concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in part_files), encoding="utf-8")
    final = out_dir / f"{name}.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", "-movflags", "+faststart", str(final)])

    # 검수용: 구간마다 두 장면(시작 1.2초, 중간)
    shots = []
    for k, st in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else t
        shots += [st + 1.2, (st + end) / 2]
    cols = 6
    sheet = Image.new("RGB", (cols * 270, -(-len(shots) // cols) * 480), "black")
    for k, st in enumerate(shots):
        shot = parts / "_shot.png"
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{st:.2f}", "-i", str(final),
             "-frames:v", "1", "-vf", "scale=270:480", str(shot)])
        sheet.paste(Image.open(shot).copy(), ((k % cols) * 270, (k // cols) * 480))
    sheet.save(out_dir / "contact.jpg", quality=85)

    sources = sorted({s["video"] for s in segs})
    meta = [plan["title"], "", f"#{name} #예능 #웃긴영상 #shorts", "", "원본 출처:"]
    meta += [f"https://www.youtube.com/watch?v={v}" for v in sources]
    (out_dir / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    print(f"완성: {final} ({t:.0f}초, {len(part_files)}구간)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
