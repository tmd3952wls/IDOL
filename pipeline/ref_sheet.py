"""레퍼런스 분석용: 영상의 일정 구간을 촘촘한 장면표로 만든다.
사용: python pipeline/ref_sheet.py <영상> <시작초> <간격초> <장수> <출력.jpg> [열수]"""
import subprocess, sys, tempfile, os
from PIL import Image, ImageDraw, ImageFont
src, t0, step, n, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cols = int(sys.argv[6]) if len(sys.argv) > 6 else 8
w, h = 225, 400
font = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 18)
sheet = Image.new("RGB", (cols * w, -(-n // cols) * h), "black")
d = ImageDraw.Draw(sheet)
with tempfile.TemporaryDirectory() as td:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-i", src, "-vf",
                    f"fps={1/step},scale={w}:{h}", "-frames:v", str(n), os.path.join(td, "f%03d.png")])
    for i in range(n):
        f = os.path.join(td, f"f{i+1:03d}.png")
        if not os.path.exists(f): break
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(Image.open(f).copy(), (x, y))
        d.rectangle([x, y, x + 62, y + 22], fill="black")
        d.text((x + 3, y), f"{t0 + i * step:.1f}", font=font, fill="yellow")
sheet.save(out, quality=88)
