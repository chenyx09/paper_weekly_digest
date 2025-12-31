import os, re, smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone
import tempfile

import requests
import feedparser
from dateutil import parser as dtparser
from openai import OpenAI
import pymupdf  # PyMuPDF for PDF extraction

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

def download_pdf(arxiv_id):
    """Download PDF from arXiv and return the file path."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"  Error downloading PDF {arxiv_id}: {e}")
        return None

def extract_pdf_text(pdf_path, max_chars=30000):
    """Extract text from PDF, focusing on key sections."""
    try:
        doc = pymupdf.open(pdf_path)
        full_text = ""
        
        # Extract text from all pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text += page.get_text()
        
        doc.close()
        
        # Clean up the text
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        # Limit to max_chars to avoid token limits
        if len(full_text) > max_chars:
            # Try to find key sections
            sections_text = extract_key_sections(full_text, max_chars)
            if sections_text:
                return sections_text
            # Otherwise just truncate
            return full_text[:max_chars] + "..."
        
        return full_text
    except Exception as e:
        print(f"  Error extracting text from PDF: {e}")
        return None

def extract_key_sections(text, max_chars):
    """Try to extract introduction, methodology, and conclusion sections."""
    text_lower = text.lower()
    sections = []
    
    # Common section headers
    intro_patterns = [r'\babstract\b', r'\bintroduction\b', r'\b1\.?\s+introduction\b']
    method_patterns = [r'\bmethod\b', r'\bmethodology\b', r'\bapproach\b', r'\b3\.?\s+method']
    conclusion_patterns = [r'\bconclusion\b', r'\bdiscussion\b', r'\bresults\b']
    
    # Try to find sections
    for pattern in intro_patterns + method_patterns + conclusion_patterns:
        match = re.search(pattern, text_lower)
        if match:
            start = match.start()
            # Get text around the section (1000 chars after header)
            section_text = text[start:start+1500]
            if section_text not in sections:
                sections.append(section_text)
    
    combined = " [...] ".join(sections)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "..."
    
    return combined if combined else None

def generate_gpt_summary(paper, client, use_pdf=True):
    """Generate an in-depth summary using GPT."""
    
    # Try to get full paper text
    paper_text = None
    if use_pdf:
        print(f"    Downloading PDF...")
        pdf_path = download_pdf(paper['arxiv_id'])
        if pdf_path:
            print(f"    Extracting text from PDF...")
            paper_text = extract_pdf_text(pdf_path)
            # Clean up temp file
            try:
                os.unlink(pdf_path)
            except:
                pass
    
    # Prepare prompt based on available information
    if paper_text:
        prompt = f"""You are a research assistant helping to digest recent academic papers. 
Please provide a comprehensive yet detailed summary of the following paper in 4-6 sentences.
Focus on:
1. The main problem being addressed and why it matters
2. The key contribution, methodology, or technical approach
3. Main results or findings
4. Potential impact, applications, or future directions

Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'][:10])}

Full Paper Text (extracted):
{paper_text}

Provide a clear, technical summary suitable for researchers in robotics, autonomous vehicles, and AI. Be specific about the methods and results mentioned in the paper."""
    else:
        # Fallback to abstract-only
        print(f"    Falling back to abstract-only summary...")
        prompt = f"""You are a research assistant helping to digest recent academic papers. 
Please provide a comprehensive summary of the following paper in 3-4 sentences based on the abstract.
Focus on:
1. The main problem being addressed
2. The key contribution or methodology
3. Potential impact or applications

Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'][:10])}
Abstract: {paper['abstract']}

Provide a clear, technical summary suitable for researchers in robotics, autonomous vehicles, and AI."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # More cost-effective than gpt-4
            messages=[
                {"role": "system", "content": "You are a helpful research assistant specializing in robotics, computer vision, and AI."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Increased for more detailed summaries
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Error generating summary: {e}")
        # Fallback to original TL;DR
        tldr = paper["abstract"].split(". ")[0].strip()
        return tldr[:300] + "..." if len(tldr) > 300 else tldr

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
    lines.append(f"Enhanced with GPT-4 summaries from full paper PDFs")
    lines.append("")
    for i, p in enumerate(top, 1):
        authors = ", ".join(p["authors"][:6]) + ("" if len(p["authors"]) <= 6 else ", et al.")
        pub = p["published"].strftime("%Y-%m-%d")
        summary = p.get("gpt_summary", "")
        
        lines.append(f"{i}. {p['title']}")
        lines.append(f"   {authors} | {pub} | score={p['score']:.1f}")
        lines.append(f"   Link: {p['link']}")
        lines.append("")
        lines.append(f"   Summary: {summary}")
        lines.append("")
        lines.append("   " + "-" * 80)
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
    # Debug mode: set via environment variable
    debug_mode = os.environ.get("DEBUG_MODE", "").lower() in ["1", "true", "yes"]
    num_papers = int(os.environ.get("NUM_PAPERS", "10"))
    
    if debug_mode:
        print("=" * 80)
        print("DEBUG MODE ENABLED - Email will NOT be sent")
        print(f"Processing only {num_papers} papers")
        print("=" * 80)
    
    papers = fetch_arxiv(days=7, max_results=250)
    top = pick_top(papers, k=num_papers)
    if not top:
        print("No papers found; no email sent.")
        return
    
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"\nGenerating GPT summaries from full PDFs for top {len(top)} papers...")
        print("This may take a few minutes...\n")
        client = OpenAI(api_key=api_key)
        for i, paper in enumerate(top, 1):
            print(f"  [{i}/{len(top)}] {paper['title'][:70]}...")
            paper["gpt_summary"] = generate_gpt_summary(paper, client, use_pdf=True)
        print("\nAll summaries generated successfully!")
    else:
        print("Warning: OPENAI_API_KEY not found. Falling back to basic TL;DR.")
        for paper in top:
            tldr = paper["abstract"].split(". ")[0].strip()
            paper["gpt_summary"] = tldr[:300] + "..." if len(tldr) > 300 else tldr
    
    subject = f"Weekly Robotics/AV/AI Paper Digest ({datetime.now().strftime('%Y-%m-%d')})"
    body = render_email(top)
    
    if debug_mode:
        print("\n" + "=" * 80)
        print("EMAIL PREVIEW (not sent):")
        print("=" * 80)
        print(f"Subject: {subject}\n")
        print(body)
        print("\n" + "=" * 80)
        print("DEBUG MODE: Email was NOT sent. Set DEBUG_MODE=0 to actually send.")
        print("=" * 80)
    else:
        send_email(subject, body)
        print(f"Sent email with {len(top)} papers.")

if __name__ == "__main__":
    main()