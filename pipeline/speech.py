"""음성 인식(단어 단위 타이밍), 말하는 부분만 남기는 컷 계산, 자막 줄 만들기."""
import re
import subprocess

import numpy as np

_model = None
SR = 16000
HOP = 320  # 20ms


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def words(path, hint=""):
    """[(시작초, 끝초, 단어)] — 클립 기준 시각. hint: 출연자 이름 등 인식에 도움 되는 말."""
    global _model
    audio = load_audio(path)
    if audio.size < 1600:
        return []
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("turbo", device="cpu", compute_type="int8")
    segs, _ = _model.transcribe(audio, language="ko", word_timestamps=True, vad_filter=True,
                                vad_parameters={"min_silence_duration_ms": 300},
                                condition_on_previous_text=False, initial_prompt=hint or None)
    return [(round(w.start, 2), round(w.end, 2), w.word.strip()) for s in segs for w in s.words
            if w.word.strip()]


def snap_onsets(ws, audio):
    """쉬었다 나오는 단어의 시작을 실제 소리가 올라오는 지점으로 당긴다.
    (인식기는 쉬는 구간 뒤 첫 단어의 시작을 실제보다 이르게 잡는 경향이 있다)"""
    n = audio.size // HOP
    if not n:
        return ws
    rms = np.sqrt(np.mean(audio[:n * HOP].reshape(n, HOP) ** 2, axis=1) + 1e-12)
    out, prev_end = [], -9.0
    for a, b, w in ws:
        if a - prev_end > 0.15 and b - a > 0.12:
            i0, i1 = int(a * SR / HOP), max(int(b * SR / HOP), int(a * SR / HOP) + 1)
            seg = rms[i0:i1]
            if seg.size:
                thr = 0.3 * float(seg.max())
                lim = min(seg.size, int(0.45 * SR / HOP))
                hit = next((k for k in range(lim) if seg[k] >= thr), 0)
                a = min(round(a + hit * HOP / SR, 2), b - 0.08)
        out.append((a, b, w))
        prev_end = b
    return out


def keep_ranges(ws, dur, lead=0.06, tail=0.14, join=0.22):
    """말이 나오는 부분만 남긴 [(a, b)] — 숨 쉴 틈 없이 붙도록 앞뒤 여유를 거의 두지 않는다."""
    if not ws:
        return [(0.0, dur)]
    out = []
    for a, b, _ in ws:
        a, b = max(0.0, a - lead), min(dur, b + tail)
        if out and a - out[-1][1] <= join:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    out = [p for p in out if p[1] - p[0] >= 0.3]
    if not out:
        return [(0.0, dur)]
    out[-1][1] = min(dur, out[-1][1] + 0.25)  # 마지막 한 박자
    return [(round(a, 2), round(b, 2)) for a, b in out]


def chunks(ranges, scene_cuts=(), max_len=2.4):
    """남긴 구간을 장면 전환 지점에서 나누고, 긴 조각은 더 나눈다(조각마다 화면 크기를 바꾼다). [(a, b)]"""
    out = []
    for a, b in ranges:
        edges = [a] + [c for c in scene_cuts if a + 0.35 < c < b - 0.35] + [b]
        for x, y in zip(edges, edges[1:]):
            n = max(1, round((y - x) / max_len + 0.3))
            step = (y - x) / n
            out += [(round(x + i * step, 3), round(x + (i + 1) * step, 3)) for i in range(n)]
    return out


def caption_lines(ws, audio, max_chars=8):
    """단어들을 짧은 자막 줄로 묶고(말하는 순간에 맞춰 뜨도록) 기본 스타일을 붙인다. [{t0, t1, text, style}]
    스타일: normal(기본) / shout(큰 소리) / question(질문) / laugh(웃음) / aside(혼잣말) / point(강조)"""
    if not ws:
        return []
    mean = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
    lines, cur = [], []

    def flush():
        if not cur:
            return
        text = " ".join(w for _, _, w in cur)
        text = re.sub(r"[.,]+$", "", re.sub(r"[.,](?=\s)", "", text)).strip()
        a, b = cur[0][0], cur[-1][1]
        seg = audio[int(a * SR):int(b * SR)]
        loud = 20 * np.log10((float(np.sqrt(np.mean(seg ** 2))) if seg.size else mean) / mean + 1e-9)
        if re.fullmatch(r"[ㅋㅎ하헤히흐아으 !~]+", text) and len(text) >= 2:
            style = "laugh"
        elif text.endswith("?"):
            style = "question"
        elif "!" in text or loud >= 4.0:
            style = "shout"
        else:
            style = "normal"
        if text:
            lines.append({"t0": round(a, 2), "t1": round(b, 2), "text": text, "style": style})
        cur.clear()

    for a, b, w in ws:
        if cur:
            length = sum(len(x[2]) + 1 for x in cur)
            if a - cur[-1][1] > 0.25 or length + len(w) > max_chars or cur[-1][2][-1] in ".?!":
                flush()
        cur.append((a, b, w))
    flush()
    return lines
