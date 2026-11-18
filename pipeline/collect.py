"""소재 수집: 연예인 이름 -> 유튜브 검색 -> 후보 영상의 댓글/타임스탬프/자막/다시보기 히트맵.

사용: python pipeline/collect.py "장도연" [--videos 8] [--comments 300]
결과: work/<이름>/candidates.json, work/<이름>/digest.md (구간 선정용 요약)
"""
import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parent.parent
BASE = {"quiet": True, "no_warnings": True, "compat_opts": {"no-certifi"}, "skip_download": True}
QUERIES = ["{n} 레전드", "{n} 웃긴 순간", "{n} 드립 모음", "{n} 예능 레전드", "{n} 웃참"]
TS = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")


def search(name, per_query=15):
    seen = {}
    with yt_dlp.YoutubeDL({**BASE, "extract_flat": True}) as ydl:
        for q in QUERIES:
            query = q.format(n=name)
            try:
                res = ydl.extract_info(f"ytsearch{per_query}:{query}", download=False)
            except Exception as e:  # noqa: BLE001
                print(f"  검색 실패 {query}: {e}", file=sys.stderr)
                continue
            for e in res.get("entries") or []:
                if not e or not e.get("id"):
                    continue
                dur = e.get("duration") or 0
                title = e.get("title") or ""
                # 쇼츠/너무 긴 영상 제외, 제목에 이름이 있는 것만
                if not (90 <= dur <= 45 * 60) or name not in title.replace(" ", ""):
                    continue
                seen.setdefault(e["id"], {
                    "id": e["id"], "title": title, "duration": dur,
                    "views": e.get("view_count") or 0, "channel": e.get("channel") or e.get("uploader") or "",
                })
    return sorted(seen.values(), key=lambda v: -v["views"])


def clean_comment(text):
    text = TS.sub("", text)
    text = re.sub(r"[\r\n]+", " ", text)
    # 한글/영문/숫자/기본 문장부호만 남김 (이모지는 폰트에 없어 깨짐)
    text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?~'\"()%\-…:;^]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_vtt(raw):
    lines, last, t = [], None, None
    for ln in raw.splitlines():
        if "-->" in ln:
            h, m, s = ln.split("-->")[0].strip().split(":")
            t = int(h) * 3600 + int(m) * 60 + float(s)
            continue
        if not ln.strip() or ln.startswith(("WEBVTT", "Kind:", "Language:")) or t is None:
            continue
        txt = re.sub(r"<[^>]*>", "", ln).strip()
        if txt and txt != last:
            lines.append([round(t, 1), txt])
            last = txt
    return lines


def fetch_subs(info):
    for src in ("subtitles", "automatic_captions"):
        tracks = info.get(src) or {}
        for lang in ("ko", "ko-orig", "ko-KR"):
            for f in tracks.get(lang) or []:
                if f.get("ext") == "vtt":
                    try:
                        raw = urllib.request.urlopen(f["url"], timeout=20).read().decode("utf-8", "ignore")
                        subs = parse_vtt(raw)
                        if subs:
                            return subs
                    except Exception:  # noqa: BLE001
                        pass
    return []


def detail(video, max_comments):
    opts = {**BASE, "getcomments": True,
            "extractor_args": {"youtube": {"max_comments": [str(max_comments), "all", "0", "0"],
                                           "comment_sort": ["top"]}}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video['id']}", download=False)
    dur = info.get("duration") or video["duration"]
    video.update(duration=dur, width=info.get("width"), height=info.get("height"),
                 views=info.get("view_count") or video["views"], upload_date=info.get("upload_date"))

    stamped, plain = [], []
    for c in info.get("comments") or []:
        raw = c.get("text") or ""
        likes = c.get("like_count") or 0
        text = clean_comment(raw)
        times = [int(h or 0) * 3600 + int(m) * 60 + int(s) for h, m, s in TS.findall(raw)]
        times = [t for t in times if t < dur]
        if times:
            for t in times:
                stamped.append({"t": t, "likes": likes, "text": text if len(times) == 1 else ""})
        elif 4 <= len(text) <= 45:
            plain.append({"likes": likes, "text": text})

    # 타임스탬프 군집화 (8초 이내는 같은 순간)
    moments = []
    for s in sorted(stamped, key=lambda x: x["t"]):
        if moments and s["t"] - moments[-1]["end"] <= 8:
            m = moments[-1]
            m["end"] = s["t"]
        else:
            m = {"t": s["t"], "end": s["t"], "score": 0.0, "n": 0, "comments": []}
            moments.append(m)
        m["score"] += 1 + math.sqrt(s["likes"])
        m["n"] += 1
        if 4 <= len(s["text"]) <= 45:
            m["comments"].append({"likes": s["likes"], "text": s["text"]})

    heat = info.get("heatmap") or []
    for m in moments:
        hv = [h["value"] for h in heat if h["start_time"] <= m["t"] + 5 and h["end_time"] >= m["t"] - 2]
        m["heat"] = round(max(hv), 2) if hv else None
        m["score"] = round(m["score"] * (1 + (m["heat"] or 0)), 1)
        m["comments"] = sorted(m["comments"], key=lambda c: -c["likes"])[:4]
        del m["end"]
    peaks = sorted(heat, key=lambda h: -h["value"])[:6]
    video["moments"] = sorted(moments, key=lambda m: -m["score"])[:12]
    video["heat_peaks"] = [{"t": round(h["start_time"]), "value": round(h["value"], 2)} for h in peaks]
    video["top_comments"] = sorted(plain, key=lambda c: -c["likes"])[:12]
    video["subs"] = fetch_subs(info)
    return video


def fmt(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def digest(name, videos):
    out = [f"# {name} 소재 후보\n"]
    for v in videos:
        out.append(f"\n## [{v['id']}] {v['title']}\n{v['channel']} · {fmt(v['duration'])} · 조회 {v['views']:,} · "
                   f"{v.get('width')}x{v.get('height')} · 자막 {'있음' if v['subs'] else '없음'}")
        out.append("일반 인기 댓글: " + " / ".join(f"{c['text']}({c['likes']})" for c in v["top_comments"][:8]))
        out.append("다시보기 피크: " + ", ".join(f"{fmt(p['t'])}({p['value']})" for p in v["heat_peaks"]))
        for m in sorted(v["moments"], key=lambda m: m["t"]):
            out.append(f"\n### {fmt(m['t'])} (t={m['t']}s, 점수 {m['score']}, 언급 {m['n']}회, heat {m['heat']})")
            for c in m["comments"]:
                out.append(f"- 댓글({c['likes']}): {c['text']}")
            near = [f"{s[0]:.0f}s {s[1]}" for s in v["subs"] if m["t"] - 6 <= s[0] <= m["t"] + 28]
            if near:
                out.append("```\n" + "\n".join(near) + "\n```")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--comments", type=int, default=300)
    a = ap.parse_args()
    work = ROOT / "work" / a.name
    work.mkdir(parents=True, exist_ok=True)

    found = search(a.name)
    print(f"검색 결과 {len(found)}개, 상위 {a.videos}개 상세 수집")
    videos = []
    for v in found[: a.videos * 2]:
        if len(videos) >= a.videos:
            break
        try:
            d = detail(v, a.comments)
        except Exception as e:  # noqa: BLE001
            print(f"  건너뜀 {v['id']}: {str(e)[:120]}", file=sys.stderr)
            continue
        if (d.get("width") or 0) < (d.get("height") or 0):
            continue  # 세로 영상 제외
        print(f"  {d['id']} 순간 {len(d['moments'])}개, 자막 {len(d['subs'])}줄 · {d['title'][:40]}")
        videos.append(d)

    (work / "candidates.json").write_text(json.dumps(videos, ensure_ascii=False, indent=1), encoding="utf-8")
    (work / "digest.md").write_text(digest(a.name, videos), encoding="utf-8")
    print(f"저장: {work / 'digest.md'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
