import os, re, smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone

import requests
import feedparser
from dateutil import parser as dtparser

ARXIV_BASE = "http://export.arxiv.org/api/query"
CATEGORIES = ["cs.RO", "eess.IV", "cs.CV", "cs.LG", "cs.AI"]

KEYWORDS = [
    "robot", "robotics", "autonomous", "driving", "vehicle", "planning", "control",
    "slam", "localization", "mapping", "trajectory", "motion", "navigation",
    "perception", "sensor fusion", "lidar", "radar", "multi-modal",
    "imitation", "reinforcement", "policy", "closed-loop", "sim2real", "world model"
]

def fetch_arxiv(days=7, max_results=250):
    cat_query = " OR ".join([f"cat:{c}" for c in CATEGORIES])
    params = dict(
        search_query=f"({cat_query})",
        start=0,
        max_results=max_results,
        sortBy="submittedDate",
        sortOrder="descending",
    )
    r = requests.get(ARXIV_BASE, params=params, timeout=30)
    r.raise_for_status()
    feed = feedparser.parse(r.text)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    papers = []
    for e in feed.entries:
        published = dtparser.parse(e.published).astimezone(timezone.utc)
        if published < cutoff:
            continue
        abstract = re.sub(r"\s+", " ", e.summary).strip()
        title = re.sub(r"\s+", " ", e.title).strip()
        arxiv_id = e.id.split("/abs/")[-1]
        papers.append({
            "title": title,
            "authors": [a.name for a in e.authors],
            "published": published,
            "link": e.link,
            "abstract": abstract,
            "arxiv_id": arxiv_id,
        })
    return papers

def score(p):
    text = (p["title"] + " " + p["abstract"]).lower()
    s = sum(1.0 for kw in KEYWORDS if kw in text)
    if "closed-loop" in text or "real-world" in text:
        s += 1.0
    if "autonomous" in text or "driving" in text:
        s += 1.0
    return s

def pick_top(papers, k=10):
    uniq = {p["arxiv_id"]: p for p in papers}
    papers = list(uniq.values())
    for p in papers:
        p["score"] = score(p)
    papers.sort(key=lambda x: (x["score"], x["published"]), reverse=True)
    return papers[:k]

def render_email(top):
    now = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"Weekly Robotics/AV/AI Papers (Top {len(top)})")
    lines.append(f"Window: last 7 days | Generated: {now} | Source: arXiv")
    lines.append("")
    for i, p in enumerate(top, 1):
        authors = ", ".join(p["authors"][:6]) + ("" if len(p["authors"]) <= 6 else ", et al.")
        pub = p["published"].strftime("%Y-%m-%d")
        tldr = p["abstract"].split(". ")[0].strip()
        if len(tldr) > 300:
            tldr = tldr[:297] + "..."
        lines.append(f"{i}. {p['title']}")
        lines.append(f"   {authors} | {pub} | score={p['score']:.1f}")
        lines.append(f"   Link: {p['link']}")
        lines.append(f"   TL;DR: {tldr}")
        lines.append("")
    return "\n".join(lines)

def send_email(subject, body):
    api_key = os.environ["SENDGRID_API_KEY"]
    email_from = os.environ["EMAIL_FROM"]
    email_to = os.environ["EMAIL_TO"]

    payload = {
        "personalizations": [{"to": [{"email": email_to}]}],
        "from": {"email": email_from},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }

    r = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    r.raise_for_status()

def main():
    papers = fetch_arxiv(days=7, max_results=250)
    top = pick_top(papers, k=10)
    if not top:
        print("No papers found; no email sent.")
        return
    subject = f"Weekly Robotics/AV/AI Paper Digest ({datetime.now().strftime('%Y-%m-%d')})"
    body = render_email(top)
    send_email(subject, body)
    print(f"Sent email with {len(top)} papers.")

if __name__ == "__main__":
    main()