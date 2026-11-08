"""XFIX review pipeline.

PR link -> diff from GitHub -> lessons from MongoDB -> Meta Muse review
-> ASI:One cross-check -> SubQ Snap merge verdict -> run saved to MongoDB.

Run it on its own to test without the chat agent:
    python pipeline.py https://github.com/<owner>/<repo>/pull/<number>
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient

load_dotenv()

PR_URL = re.compile(r"https?://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)")
MAX_REVIEW_CHARS = 60_000   # diff sent to the reviewer models
MAX_JUDGE_CHARS = 400_000   # SubQ Snap has a 384k-token window
LESSON_LIMIT = 5
MAX_FIX_ATTEMPTS = 2
FIX_PREVIEW_LINES = 40      # lines of the fix diff shown in chat

META_MODEL = "muse-spark-1.3"
ASI1_MODEL = "asi1"
SUBQ_MODEL = "subq-snap-1.1"

meta = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["META_API_KEY"])
asi1 = OpenAI(base_url="https://api.asi1.ai/v1", api_key=os.environ["ASI1_API_KEY"])
db = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=10000)["xfix"]

REVIEW_PROMPT = """You are XFIX, a strict pre-merge code reviewer.
Review the unified diff. Report only real problems this change introduces:
bugs, security risks, broken behavior, missing error handling, or risky logic with no test.
Do not report style nits or personal preferences.

Return JSON only, in this shape:
{"findings": [{"file": "path", "line": 12, "severity": "high|medium|low",
  "title": "short name of the problem", "reason": "one sentence on why it is a problem"}]}
Return {"findings": []} if the change is safe to merge."""

CHECK_PROMPT = """You are verifying another reviewer's findings on a pull request.
For each numbered finding, decide whether it is a real problem in this diff.
Reject findings that are wrong, speculative, or not caused by this change.

Return JSON only, in this shape:
{"checks": [{"id": 1, "confirmed": true, "note": "one short sentence"}]}"""

# Meta Model API structured output: decoding is constrained to this schema,
# so the review reply is always parseable JSON with all fields present.
FINDINGS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "review_findings",
        "schema": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string"},
                            "line": {"type": "integer"},
                            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                            "title": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["file", "line", "severity", "title", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["findings"],
            "additionalProperties": False,
        },
    },
}

FIX_PROMPT = """You are XFIX's fixer. Rewrite the pull request so every confirmed problem is fixed
while keeping what the author was trying to do.
Return only the corrected change as a unified diff against the same base, starting with "diff --git".
No commentary."""


def parse_json(text):
    """Pull the first JSON object out of a model reply, with or without code fences."""
    text = text or ""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start:end + 1])


def ask(client, model, system, user, max_tokens, response_format=None):
    kwargs = {"response_format": response_format} if response_format is not None else {}
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        **kwargs,
    )
    return r.choices[0].message.content


def fetch_diff(owner, repo, number):
    # github.com/.../pull/N.diff is not counted against the REST API rate limit
    r = requests.get(f"https://github.com/{owner}/{repo}/pull/{number}.diff", timeout=30)
    r.raise_for_status()
    if not r.text.startswith("diff --git"):
        raise ValueError(f"{owner}/{repo}#{number} is not a pull request with a diff")
    return r.text


def load_lessons(repo):
    """Findings confirmed in past runs: this repo's first, then any repo."""
    lessons = list(db.lessons.find({"repo": repo}).sort("created_at", -1).limit(LESSON_LIMIT))
    if len(lessons) < LESSON_LIMIT:
        lessons += list(db.lessons.find({"repo": {"$ne": repo}})
                        .sort("created_at", -1).limit(LESSON_LIMIT - len(lessons)))
    return lessons


def review(diff, lessons):
    """Meta Muse writes the findings via structured output. Falls back to ASI:One if the Meta call fails."""
    user = f"Diff:\n{diff[:MAX_REVIEW_CHARS]}"
    if lessons:
        past = "\n".join(f"- {l['title']} ({l['file']}): {l['reason']}" for l in lessons)
        user = f"Problems confirmed in earlier reviews. Check whether this change repeats any:\n{past}\n\n{user}"
    attempts = (("Meta Muse", meta, META_MODEL, FINDINGS_SCHEMA),
                ("ASI:One", asi1, ASI1_MODEL, None))
    for name, client, model, fmt in attempts:
        try:
            reply = ask(client, model, REVIEW_PROMPT, user, 8000, fmt)
            try:
                findings = json.loads(reply or "").get("findings", [])
            except (ValueError, AttributeError):
                findings = parse_json(reply).get("findings", [])
            return name, [f for f in findings if isinstance(f, dict) and f.get("title")]
        except Exception as e:
            print(f"[xfix] {name} review failed: {e}", file=sys.stderr)
    raise RuntimeError("both reviewer models failed")


def cross_check(diff, findings):
    """ASI:One confirms or rejects each finding. confirmed=None means it could not be checked."""
    if not findings:
        return findings
    numbered = "\n".join(f"{i}. [{f.get('severity', '?')}] {f.get('file')}:{f.get('line')} "
                         f"{f['title']}: {f.get('reason', '')}" for i, f in enumerate(findings, 1))
    try:
        reply = ask(asi1, ASI1_MODEL, CHECK_PROMPT,
                    f"Findings:\n{numbered}\n\nDiff:\n{diff[:MAX_REVIEW_CHARS]}", 3000)
        checks = {c.get("id"): c for c in parse_json(reply).get("checks", [])}
    except Exception as e:
        print(f"[xfix] ASI:One cross-check failed: {e}", file=sys.stderr)
        checks = {}
    for i, f in enumerate(findings, 1):
        c = checks.get(i)
        f["confirmed"] = bool(c.get("confirmed")) if c else None
        f["check_note"] = c.get("note", "") if c else ""
    return findings


def judge(label, diff, kept):
    """SubQ Snap returns the probability that the PR should be blocked from merging."""
    listed = "\n".join(f"- [{f.get('severity', '?')}] {f.get('file')}: {f['title']}: {f.get('reason', '')}"
                       for f in kept) or "- none"
    body = {
        "model": SUBQ_MODEL,
        "state": f"Pull request {label}.\n\nConfirmed review findings:\n{listed}\n\nDiff:\n{diff[:MAX_JUDGE_CHARS]}",
        "questions": {"block_merge": {
            "type": "noul",
            "instructions": "Should this pull request be blocked from merging until it is changed?",
            "criteria": {"true": "The change introduces a bug, security risk, or broken behavior.",
                         "false": "The change is safe to merge as it is."},
        }},
    }
    try:
        r = requests.post("https://api.subq.ai/v1/systemone", json=body, timeout=60,
                          headers={"Authorization": f"Bearer {os.environ['SUBQ_API_KEY']}"})
        r.raise_for_status()
        return "SubQ Snap", float(r.json()["answers"]["block_merge"]["noul"])
    except Exception as e:
        print(f"[xfix] SubQ judge failed: {e}", file=sys.stderr)
        severities = {f.get("severity") for f in kept}
        return "fallback rule", 0.9 if "high" in severities else 0.6 if kept else 0.1


def finding_lines(findings):
    return "\n".join(f"- [{f.get('severity', '?')}] `{f.get('file')}:{f.get('line')}` {f['title']}" for f in findings)


def evaluate(label, diff, lessons, say=lambda _m: None, heading=lambda name, n: f"### {name}"):
    """One review pass: Meta findings, ASI:One cross-check, SubQ verdict. `say` posts progress to chat."""
    reviewer, findings = review(diff, lessons)
    found = f"Found {len(findings)} possible problem(s):\n{finding_lines(findings)}" if findings else "No problems found."
    by = "" if reviewer == "Meta Muse" else f"\n(Meta was unavailable, so {reviewer} reviewed.)"
    say(f"{heading('Meta Muse review', 2)}\n{found}{by}\n\n_Now: ASI:One is checking each finding._")

    findings = cross_check(diff, findings)
    kept = [f for f in findings if f.get("confirmed") is not False]
    say(f"{heading('ASI:One cross-check', 3)}\n{len(kept)} confirmed, {len(findings) - len(kept)} dropped as false alarms."
        "\n\n_Now: SubQ Snap is scoring the merge risk._")

    judge_name, block_probability = judge(label, diff, kept)
    verdict = "REQUEST CHANGES" if block_probability >= 0.5 else "APPROVE"
    say(f"{heading('SubQ Snap merge verdict', 4)}\nBlock score {block_probability:.2f}: **{verdict}**.")
    return {"reviewer": reviewer, "judge": judge_name, "findings": findings,
            "kept": kept, "block_probability": block_probability}


def propose_fix(diff, kept):
    """Meta Muse rewrites the change so the confirmed problems are gone."""
    problems = "\n".join(f"- {f.get('file')}:{f.get('line')} {f['title']}: {f.get('reason', '')}"
                         for f in kept) or "- the merge judge blocked this change"
    try:
        reply = ask(meta, META_MODEL, FIX_PROMPT,
                    f"Confirmed problems:\n{problems}\n\nOriginal diff:\n{diff[:MAX_REVIEW_CHARS]}", 8000) or ""
    except Exception as e:
        print(f"[xfix] Meta fix failed: {e}", file=sys.stderr)
        return ""
    fenced = re.search(r"```(?:diff|patch)?\s*(diff --git.*?)```", reply, re.S)
    if fenced:
        return fenced.group(1).strip()
    return reply[reply.find("diff --git"):].strip() if "diff --git" in reply else ""


def save_run(run, kept):
    db.runs.insert_one(run)
    now = datetime.now(timezone.utc)
    if kept:
        db.lessons.insert_many([{
            "repo": run["repo"], "file": f.get("file"), "title": f["title"],
            "reason": f.get("reason", ""), "severity": f.get("severity"),
            "source_pr": run["label"], "created_at": now,
        } for f in kept])
    return db.runs.count_documents({})


def format_reply(run, run_number):
    verdict = "REQUEST CHANGES" if run["block_probability"] >= 0.5 else "APPROVE"
    kept = [f for f in run["findings"] if f.get("confirmed") is not False]
    dropped = [f for f in run["findings"] if f.get("confirmed") is False]
    lines = [
        f"**XFIX verdict: {verdict}** (block score {run['block_probability']:.2f}, judged by {run['judge']})",
        f"PR: {run['label']}",
        "",
    ]
    if kept:
        lines.append(f"**Confirmed findings** (reviewed by {run['reviewer']}, checked by ASI:One):")
        for i, f in enumerate(kept, 1):
            where = f"{f.get('file')}:{f.get('line')}" if f.get("line") else f.get("file")
            flag = "" if f.get("confirmed") else " _(not cross-checked)_"
            lines.append(f"{i}. [{f.get('severity', '?')}] `{where}`: {f['title']}. {f.get('reason', '')}{flag}")
    else:
        lines.append("No confirmed problems found.")
    if dropped:
        lines += ["", f"Dropped {len(dropped)} finding(s) that ASI:One could not confirm:"]
        lines += [f"- {f['title']}" for f in dropped]
    fix = run.get("fix")
    if fix:
        fixed_ok = fix["block_probability"] < 0.5
        still = [f for f in fix["findings"] if f.get("confirmed") is not False]
        preview = fix["diff"].splitlines()
        if len(preview) > FIX_PREVIEW_LINES:
            preview = preview[:FIX_PREVIEW_LINES] + [f"... ({len(fix['diff'].splitlines()) - FIX_PREVIEW_LINES} more lines)"]
        lines += ["", f"**XFIX fix** (written by Meta Muse, attempt {fix['attempts']}):",
                  "```diff", *preview, "```",
                  f"**Re-review of the fix: {'APPROVE' if fixed_ok else 'STILL BLOCKED'}** "
                  f"(block score {run['block_probability']:.2f} → {fix['block_probability']:.2f}, "
                  f"{len(still)} confirmed finding(s) left, judged by {fix['judge']})"]
    lines += ["", f"Run #{run_number} saved to MongoDB. Lessons used from past runs: {run['lessons_used']}. "
                  f"Took {run['seconds']:.0f}s."]
    return "\n".join(lines)


HELP = ("Send me a GitHub pull request link, like https://github.com/owner/repo/pull/123, "
        "or paste a unified diff. I'll review it and tell you whether it's safe to merge.")


def run_review(text, progress=None):
    """Full pipeline for one chat message. Returns the reply text; `progress` gets one message per stage."""
    say = progress or (lambda _m: None)
    started = time.time()
    m = PR_URL.search(text)
    if m:
        owner, repo, number = m.groups()
        repo_full, label = f"{owner}/{repo}", f"{owner}/{repo}#{number}"
        say(f"## XFIX review started\nPR: {label}\n\n_Now: downloading the diff from GitHub._")
        diff = fetch_diff(owner, repo, number)
    elif "diff --git" in text:
        repo_full, label = "pasted", "pasted diff"
        say("## XFIX review started\nReviewing the pasted diff.")
        diff = text[text.index("diff --git"):]
    else:
        return HELP

    lessons = load_lessons(repo_full)
    say(f"### Stage 1 of 5: Fetch\nDownloaded the diff ({len(diff):,} characters) and loaded {len(lessons)} "
        "lesson(s) from past runs in MongoDB.\n\n_Now: Meta Muse is reviewing the diff._")
    first = evaluate(label, diff, lessons, say, lambda name, n: f"### Stage {n} of 5: {name}")

    # Fix loop: while the merge judge blocks, Meta writes a fix and XFIX re-reviews it.
    fix, current, current_diff = None, first, diff
    if first["block_probability"] < 0.5:
        say("### Stage 5 of 5: Fix loop\nNo fix needed. The change is safe to merge.")
    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        if current["block_probability"] < 0.5:
            break
        say(f"### Stage 5 of 5: Fix loop (attempt {attempt})\nThe merge is blocked, so Meta Muse is writing a fix "
            f"for {len(current['kept'])} confirmed problem(s).")
        fixed_diff = propose_fix(current_diff, current["kept"])
        if not fixed_diff:
            say("#### Fix failed\nMeta Muse could not produce a fix. Sending the review as it is.")
            break
        say(f"#### Fix written\n{len(fixed_diff.splitlines())} lines. Re-checking it with Meta Muse, ASI:One and SubQ Snap.")
        current, current_diff = evaluate(f"{label} (XFIX fix {attempt})", fixed_diff, lessons, say,
                                         lambda name, n: f"#### Re-check: {name}"), fixed_diff
        fix = {"attempts": attempt, "diff": fixed_diff, "reviewer": current["reviewer"],
               "judge": current["judge"], "findings": current["findings"],
               "block_probability": current["block_probability"]}

    run = {
        "label": label, "repo": repo_full, "url": m.group(0) if m else None,
        "diff_chars": len(diff), "reviewer": first["reviewer"], "judge": first["judge"],
        "findings": first["findings"], "block_probability": first["block_probability"],
        "fix": fix, "lessons_used": len(lessons), "seconds": time.time() - started,
        "created_at": datetime.now(timezone.utc),
    }
    return format_reply(run, save_run(run, first["kept"]))


if __name__ == "__main__":
    print(run_review(" ".join(sys.argv[1:]) or "https://github.com/fetchai/uAgents/pull/610",
                     progress=lambda msg: print(msg + "\n---", file=sys.stderr)))
