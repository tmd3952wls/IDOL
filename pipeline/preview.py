"""후보 순간 미리보기: 점수 상위 순간마다 앞뒤 구간을 저화질로 받아 2초 간격 장면표를 만든다.

사용: python pipeline/preview.py "장도연" [--top 18]
결과: work/<이름>/preview/NN_<id>_<t>.jpg (16장면, 각 칸에 원본 영상의 초 표시)
      work/<이름>/preview/index.json (순간별 댓글, 장면 전환 시각)
구간 선정 시 이 장면표를 직접 보고 시작·끝·키워드를 정한다.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
BEFORE, AFTER, STEP = 8, 24, 2
FONT = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 30)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")


def build(job):
    i, v, m, out_dir = job
    start = max(0, m["t"] - BEFORE)
    end = min(v["duration"], m["t"] + AFTER)
    clip = out_dir / f"{v['id']}_{m['t']}.mp4"
    if not clip.exists():
        r = run(["yt-dlp", "--compat-options", "no-certifi", "--no-warnings", "-q",
                 "-f", "bv*[height<=480]+ba/b[height<=480]/b", "--download-sections", f"*{start}-{end}",
                 "--force-keyframes-at-cuts", "--merge-output-format", "mp4", "-o", str(clip),
                 f"https://www.youtube.com/watch?v={v['id']}"])
        if r.returncode != 0 or not clip.exists():
            return None
    # 장면 전환 시각 (원본 영상 기준 초)
    r = run(["ffmpeg", "-hide_banner", "-i", str(clip), "-vf", "select='gt(scene,0.32)',showinfo",
             "-an", "-f", "null", "-"])
    cuts = [round(start + float(x), 1) for x in re.findall(r"pts_time:([\d.]+)", r.stderr)]

    with tempfile.TemporaryDirectory() as td:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-vf", f"fps=1/{STEP},scale=480:270",
             f"{td}/f%02d.jpg"])
        frames = sorted(Path(td).glob("f*.jpg"))[:16]
        sheet = Image.new("RGB", (1920, 270 * 4), "black")
        d = ImageDraw.Draw(sheet)
        for k, f in enumerate(frames):
            x, y = (k % 4) * 480, (k // 4) * 270
            sheet.paste(Image.open(f), (x, y))
            t = start + k * STEP + STEP / 2
            label = f"{t:.0f}s" + (" *" if abs(t - m["t"]) <= STEP / 2 else "")
            d.rectangle([x, y, x + 110, y + 40], fill="black")
            d.text((x + 6, y + 2), label, font=FONT, fill="yellow")
        name = f"{i:02d}_{v['id']}_{m['t']}.jpg"
        sheet.save(out_dir / name, quality=82)
    return {"n": i, "sheet": name, "video": v["id"], "title": v["title"], "t": m["t"], "score": m["score"],
            "window": [start, end], "cuts": cuts, "comments": m["comments"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--per-video", type=int, default=4)
    a = ap.parse_args()
    work = ROOT / "work" / a.name
    out_dir = work / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = json.loads((work / "candidates.json").read_text(encoding="utf-8"))

    pool = sorted(((m, v) for v in videos for m in v["moments"] if m["comments"]), key=lambda x: -x[0]["score"])
    picked, used = [], {}
    for m, v in pool:
        if used.get(v["id"], 0) >= a.per_video:
            continue
        if any(pv["id"] == v["id"] and abs(pm["t"] - m["t"]) < 25 for pm, pv in picked):
            continue
        picked.append((m, v))
        used[v["id"]] = used.get(v["id"], 0) + 1
        if len(picked) >= a.top:
            break

    jobs = [(i, v, m, out_dir) for i, (m, v) in enumerate(picked, 1)]
    with ThreadPoolExecutor(4) as ex:
        results = [r for r in ex.map(build, jobs) if r]
    # 대사 인식 (원본 영상 기준 초). 유튜브 자동 자막보다 훨씬 정확하다.
    import speech
    for r in results:
        ws = speech.words(out_dir / f"{r['video']}_{r['t']}.mp4")
        r["speech"] = " ".join(f"[{r['window'][0] + a:.0f}]{w}" if k % 4 == 0 else w
                               for k, (a, _, w) in enumerate(ws))
    (out_dir / "index.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in results:
        print(f"{r['sheet']}  cuts={r['cuts']}")
        print(f"    대사: {r['speech']}")
        for c in r["comments"][:3]:
            print(f"    ({c['likes']}) {c['text']}")
    print(f"장면표 {len(results)}개: {out_dir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
