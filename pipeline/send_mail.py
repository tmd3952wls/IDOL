"""완성된 쇼츠를 메일로 보낸다 (네이버 SMTP).

사용: python pipeline/send_mail.py "<이름>" [--check]
  --check: 메일을 보내지 않고 설정과 SMTP 로그인만 확인한다.
보내는 것: 제목·태그·원본 출처(meta.txt 내용), 장면 모음(contact.jpg),
          영상(<이름>.mp4. MAIL_MAX_MB를 넘으면 그 크기에 맞게 줄인 사본 <이름>_mail.mp4를 첨부한다)
네이버 SMTP는 메일 한 통에 약 38MB까지 받는다(첨부는 인코딩으로 37% 커지므로 MAIL_MAX_MB는 25 이하로).
설정은 프로젝트 루트의 .env 에서 읽는다 (.env.example 참고).
"""
import mimetypes
import smtplib
import ssl
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    env = {}
    path = ROOT / ".env"
    if not path.exists():
        sys.exit(".env 파일이 없습니다. .env.example 을 .env 로 복사해 값을 채워 주세요.")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "MAIL_TO") if not env.get(k)]
    if missing:
        sys.exit(f".env 에 값이 비어 있습니다: {', '.join(missing)}")
    return env


def connect(env):
    host, port = env["SMTP_HOST"], int(env["SMTP_PORT"])
    ctx = ssl.create_default_context()
    if port == 465:
        server = smtplib.SMTP_SSL(host, port, context=ctx, timeout=30)
    else:
        server = smtplib.SMTP(host, port, timeout=30)
        server.starttls(context=ctx)
    server.login(env["SMTP_USER"], env["SMTP_PASSWORD"])
    return server


def shrink(video, limit_mb):
    """메일 한도에 맞게 영상을 줄인 사본(<이름>_mail.mp4)을 만든다. 720x1280, 한도에 맞춘 비트레이트."""
    out = video.with_name(video.stem + "_mail.mp4")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(video)], capture_output=True, text=True).stdout.strip())
    total_kbps = limit_mb * 8000 * 0.93 / dur
    for attempt in range(3):
        v = max(200, int(total_kbps - 96))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", "scale=720:-2",
                        "-c:v", "libx264", "-preset", "slow", "-b:v", f"{v}k", "-maxrate", f"{int(v * 1.3)}k",
                        "-bufsize", f"{v * 2}k", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
                        str(out)], check=True)
        if out.stat().st_size / 1_000_000 <= limit_mb:
            return out
        total_kbps *= 0.85
    return None


def attach(msg, path):
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    main, sub = ctype.split("/", 1)
    msg.add_attachment(path.read_bytes(), maintype=main, subtype=sub, filename=path.name)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    env = load_env()
    if "--check" in sys.argv:
        try:
            connect(env).quit()
        except smtplib.SMTPAuthenticationError:
            sys.exit("로그인 실패: 비밀번호를 확인하고, 네이버 메일 환경설정에서 POP3/SMTP 사용이 켜져 있는지 확인해 주세요.")
        print(f"로그인 성공: {env['SMTP_USER']} -> 받는 사람 {env['MAIL_TO']}")
        return
    if not args:
        sys.exit('사용: python pipeline/send_mail.py "<이름>" [--check]')

    name = args[0]
    out = ROOT / "output" / name
    video, meta, sheet = out / f"{name}.mp4", out / "meta.txt", out / "contact.jpg"
    if not video.exists():
        sys.exit(f"완성 영상이 없습니다: {video}")
    body = meta.read_text(encoding="utf-8") if meta.exists() else name
    title = body.splitlines()[0] if body else name
    size_mb = video.stat().st_size / 1_000_000
    limit = float(env.get("MAIL_MAX_MB") or 20)

    msg = EmailMessage()
    msg["Subject"] = f"[쇼츠 완성] {title}"
    msg["From"] = env["SMTP_USER"]
    recipients = [x.strip() for x in env["MAIL_TO"].split(",") if x.strip()]
    msg["To"] = ", ".join(recipients)
    to_attach = video
    if size_mb <= limit:
        note = f"영상 파일({size_mb:.0f}MB)을 첨부했습니다."
    else:
        to_attach = shrink(video, limit)   # 한도를 넘으면 메일용으로 줄여서 첨부
        if to_attach:
            note = (f"영상을 메일용으로 줄여 첨부했습니다 (원본 {size_mb:.0f}MB -> "
                    f"{to_attach.stat().st_size / 1_000_000:.0f}MB, 720x1280).\n원본 위치: {video}")
        else:
            note = (f"영상 파일은 {size_mb:.0f}MB로, 줄여도 첨부 한도({limit:.0f}MB)에 맞지 않아 첨부하지 않았습니다.\n"
                    f"PC의 위치: {video}")
    msg.set_content(f"{body}\n\n----\n{note}\n")
    if sheet.exists():
        attach(msg, sheet)
    if to_attach:
        attach(msg, to_attach)

    try:
        server = connect(env)
    except smtplib.SMTPAuthenticationError:
        sys.exit("로그인 실패: 비밀번호를 확인하고, 네이버 메일 환경설정에서 POP3/SMTP 사용이 켜져 있는지 확인해 주세요.")
    with server:
        server.send_message(msg, from_addr=env["SMTP_USER"], to_addrs=recipients)
    print(f"보냈습니다: {msg['To']} / {msg['Subject']} / {note.splitlines()[0]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
