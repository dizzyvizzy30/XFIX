# XFIX

**A pre-merge verifier agent.** Paste a GitHub pull request link into ASI:One chat, and XFIX reviews the change with Meta's Muse model, has ASI:One double-check every finding, asks SubQ Snap for a merge verdict, writes and re-checks a fix when the PR is blocked, and remembers each run in MongoDB so later reviews learn from earlier ones.

Built at the AIE Code Summit 2026 Hackathon for the theme *verify what other agents ship before it merges*.

## How a review works

1. **ASI:One chat (Fetch.ai).** A user sends a PR link, or pastes a unified diff, to the XFIX agent.
2. **Agentverse mailbox (Fetch.ai).** Delivers the message to the XFIX uAgent running locally (`agent.py`).
3. **GitHub + MongoDB.** XFIX downloads the PR diff and loads *lessons*: findings confirmed in past runs.
4. **Meta Model API (`muse-spark-1.3`).** The main reviewer. It returns structured findings (file, line, severity, reason) and is told to check whether the change repeats any past lesson.
5. **ASI:One (`asi1`).** An independent cross-checker. Findings it can't confirm are dropped, which cuts false alarms.
6. **Subquadratic SubQ Snap (`subq-snap-1.1`).** A decision model with a 384k-token window. It reads the diff and the confirmed findings and returns the probability that the PR should be blocked.
7. **Fix loop (Meta).** If SubQ blocks the PR, Meta Muse writes a corrected diff, and XFIX re-reviews it (steps 4–6) up to 2 times until it passes.
8. **MongoDB Atlas.** The run is saved to `xfix.runs`, and confirmed findings are added to `xfix.lessons` for the next review.

Demo, from a real run on the planted bug in branch `demo/fast-path` (a "fast path" that silently auto-approves almost every PR):

```
XFIX verdict: REQUEST CHANGES (block score 1.00, judged by SubQ Snap)
1. [high] pipeline.py:210: Review bypass auto-approves PRs without checks.

XFIX fix (written by Meta Muse): large diffs now escalate to human review instead of auto-approving.
Re-review of the fix: APPROVE (block score 1.00 → 0.02, 0 confirmed finding(s) left)
Run #3 saved to MongoDB. Lessons used from past runs: 3. Took 55s.
```

Example reply, from a real run on `fetchai/uAgents#610`:

```
XFIX verdict: REQUEST CHANGES (block score 0.96, judged by SubQ Snap)
PR: fetchai/uAgents#610

Confirmed findings (reviewed by Meta Muse, checked by ASI:One):
1. [high] python/src/uagents/agent.py:332: Agent constructor mutates global logging state. ...
2. [high] python/src/uagents/utils.py:38: set_global_log_level affects all loggers unsafely. ...

Dropped 2 finding(s) that ASI:One could not confirm.
Run #1 saved to MongoDB. Lessons used from past runs: 0. Took 70s.
```

If a model call fails, XFIX keeps going: the review falls back to ASI:One, and the verdict falls back to a severity rule.

## Sponsor tech

| Sponsor | What XFIX uses | Job |
|---|---|---|
| Meta | Meta Model API, `muse-spark-1.3` | Writes the review findings |
| Fetch.ai | uAgents, Agentverse mailbox, ASI:One chat and model | Chat entry point, agent runtime, cross-checker |
| Subquadratic | SubQ Snap 1.1 decision API | Merge verdict and block score |
| MongoDB | Atlas (`xfix.runs`, `xfix.lessons`) | Run history and lessons that improve later reviews |

## Run it

Requires Python 3.10–3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

Test the pipeline without the chat agent:

```bash
python pipeline.py https://github.com/fetchai/uAgents/pull/610
```

Run the chat agent:

```bash
python agent.py
```

Open the Agent inspector link it prints, click **Connect → Mailbox**, then use **Agent Profile → Chat with Agent** to talk to it from ASI:One.

## Files

- `agent.py`: the Fetch.ai uAgent. Handles chat messages and replies.
- `pipeline.py`: the review pipeline (GitHub → MongoDB lessons → Meta → ASI:One → SubQ → fix loop → MongoDB).
- `.env.example`: the keys XFIX needs. Your real `.env` is git-ignored.
