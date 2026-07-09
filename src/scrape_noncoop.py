"""
Targeted scrape of LIKELY NON-COOPERATIVE comment pairs from the high-purity
registers (askscience / AmItheAsshole / MaliciousCompliance) to break the
register->Cooperative confound: we already have plenty of Cooperative examples
from these subs, we need the violations.

Two problems this solves:
  1. Reddit's unauthenticated .json is 403'd — this uses OAuth (userless
     client_credentials grant). Create a "script" app at
     https://www.reddit.com/prefs/apps and export:
        REDDIT_CLIENT_ID=...   REDDIT_CLIENT_SECRET=...
  2. Using the classifier alone to find non-Cooperative comments is circular
     (its Cooperative bias is exactly what we're fixing), so it under-recalls.
     We flag a pair if EITHER the classifier says non-Cooperative OR a
     register-agnostic keyword/heuristic pass fires. Union = higher recall on
     the violations we actually want annotated.

Modes:
  --validate   offline: run the keyword filter over the existing labeled raw
               CSVs (joined to corpus.csv for gold) and report precision/recall
               vs the classifier flag. No network. Run this first.
  (default)    live scrape: needs the two env vars above.
"""
import argparse
import base64
import csv
import os
import re
import sys
import time
import json
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
CORPUS = ROOT / "data" / "annotated" / "corpus.csv"
TARGET_SUBS = ["askscience", "AmItheAsshole", "MaliciousCompliance"]
UA = "grice-maxim-classifier/1.0 (research; non-cooperative pragmatics sampling)"

WH = ("what", "why", "how", "when", "where", "who", "which", "did", "do ",
      "does", "are ", "is ", "can ", "could ", "would ", "should ", "will ")
STOP = set("the a an and or but if then of to in on at for with as is are was were "
           "be been being it this that these those i you he she they we me my your "
           "his her their our so do does did not no yes im ive youre dont cant".split())


def is_question(context: str) -> bool:
    c = (context or "").strip().lower()
    return c.endswith("?") or c.startswith(WH)


def _content_words(s):
    return {w for w in re.findall(r"[a-z']+", (s or "").lower()) if w not in STOP and len(w) > 2}


def noncoop_signals(utterance: str, context: str) -> list:
    """register-agnostic pragmatic-violation cues. returns list of reason tags."""
    u = (utterance or "").strip()
    ul = u.lower()
    reasons = []

    # Quality flouting: irony / sarcasm / hyperbole markers
    irony = (r"\b(oh (great|sure|wonderful|joy|good)|yeah,? right|what a "
             r"(surprise|shock|tragedy|coincidence)|i'?m sure|because that "
             r"(worked|went)|real (einstein|genius)|breaking news|sure,? and|"
             r"my heart (bleeds|literally)|just what i needed|so helpful|"
             r"a million times|and i'?m the (queen|king|pope))\b")
    if re.search(irony, ul) or "/s" in ul:
        reasons.append("irony")
    if re.search(r'".{1,30}"', u) or sum(1 for w in u.split() if len(w) > 3 and w.isupper()) >= 2:
        reasons.append("scare_quotes_or_caps")
    if u.count("!") >= 3:
        reasons.append("exclaim")

    # Opting out
    if re.search(r"\b(no comment|i'?d rather not|rather not say|none of your "
                 r"business|not at liberty|why do you (care|ask)|not going to "
                 r"(say|tell|answer)|mind your own)\b", ul):
        reasons.append("opting_out")

    # Quantity: terse non-answer to a question, or a wall of text
    words = u.split()
    if is_question(context) and len(words) <= 3 and u.endswith("."):
        reasons.append("terse")
    if len(u) > 400:
        reasons.append("verbose")

    # Manner: hedge stacking / double negatives / vague "thing"
    hedges = sum(ul.count(h) for h in ("kind of", "sort of", "i guess", "maybe",
                 "probably", "might be", "possibly", "or something"))
    if hedges >= 2 or re.search(r"\bnot un|not not\b", ul) or ul.count("thing") >= 3:
        reasons.append("manner_vague")

    # Relation: non-sequitur proxy — reply barely overlaps a question's content
    if is_question(context) and len(words) >= 4:
        cw_u, cw_c = _content_words(u), _content_words(context)
        if cw_c:
            jac = len(cw_u & cw_c) / max(1, len(cw_u | cw_c))
            if jac < 0.06:
                reasons.append("deflection")

    # Hostility (rich in AITA / MaliciousCompliance) — judgments, insults
    if re.search(r"\b(yta|esh|you'?re the (asshole|ah)|asshole|entitled|selfish|"
                 r"idiot|stupid|shut up|screw you|grow up|delusional)\b", ul):
        reasons.append("hostility")

    return reasons


# ----------------------------- offline validation -----------------------------
def validate():
    import pandas as pd

    def norm(s):
        return "" if pd.isna(s) else str(s).strip().strip('"').strip()

    corpus = pd.read_csv(CORPUS)
    gold = {}
    for _, r in corpus.iterrows():
        gold[norm(r["utterance"]) + "\x00" + norm(r["context"])] = r["maxim"]

    rows = []
    for f in sorted(RAW.glob("reddit_*.csv")):
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            k = norm(r["utterance"]) + "\x00" + norm(r["context"])
            if k not in gold:
                continue
            truth = gold[k]
            rows.append({
                "sub": r.get("subreddit", "?"),
                "truth_noncoop": truth != "Cooperative",
                "truth": truth,
                "model_flag": str(r.get("predicted_maxim", "")).strip() not in ("", "Cooperative", "nan"),
                "kw": noncoop_signals(str(r["utterance"]), str(r.get("context", ""))),
            })
    df = pd.DataFrame(rows)
    df["kw_flag"] = df["kw"].map(lambda x: len(x) > 0)
    df["union_flag"] = df["model_flag"] | df["kw_flag"]

    def prf(flag_col, sub=None):
        d = df if sub is None else df[df["sub"] == sub]
        tp = int((d[flag_col] & d["truth_noncoop"]).sum())
        fp = int((d[flag_col] & ~d["truth_noncoop"]).sum())
        fn = int((~d[flag_col] & d["truth_noncoop"]).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else float("nan")
        return tp, fp, fn, prec, rec, f1

    print(f"Validating filter on {len(df)} labeled raw rows "
          f"({df['truth_noncoop'].sum()} non-Cooperative, "
          f"{(~df['truth_noncoop']).sum()} Cooperative)\n")
    print(f"{'flag':<12}{'scope':<20}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>7}{'rec':>7}{'F1':>7}")
    for flag in ["model_flag", "kw_flag", "union_flag"]:
        tp, fp, fn, p, r, f = prf(flag)
        print(f"{flag:<12}{'ALL':<20}{tp:>4}{fp:>4}{fn:>4}{p:>7.2f}{r:>7.2f}{f:>7.2f}")
    print()
    for sub in TARGET_SUBS:
        for flag in ["model_flag", "kw_flag", "union_flag"]:
            tp, fp, fn, p, r, f = prf(flag, sub)
            print(f"{flag:<12}{sub:<20}{tp:>4}{fp:>4}{fn:>4}{p:>7.2f}{r:>7.2f}{f:>7.2f}")
        print()
    # which keyword signals fire on true non-Cooperative in target subs
    tgt = df[df["sub"].isin(TARGET_SUBS) & df["truth_noncoop"]]
    sig = Counter(s for row in tgt["kw"] for s in row)
    print(f"keyword signals firing on true non-Coop in target subs: {dict(sig)}")


# ------------------------------- live scrape ----------------------------------
def get_token():
    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not cid or not secret:
        sys.exit("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET (create a "
                 "'script' app at https://www.reddit.com/prefs/apps).")
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    data = b"grant_type=client_credentials"
    req = urllib.request.Request(
        "https://www.reddit.com/api/v1/access_token", data=data,
        headers={"Authorization": f"Basic {auth}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())["access_token"]


def api_get(path, token):
    req = urllib.request.Request(
        "https://oauth.reddit.com" + path,
        headers={"Authorization": f"Bearer {token}", "User-Agent": UA})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < 3:
                time.sleep(5 * (i + 1)); continue
            print(f"  HTTP {e.code} {path}"); return None
    return None


def walk_comments(node, post_title, sub, out):
    """recurse the comment tree, emit parent->reply pairs (with crowd signal)"""
    if not node or node.get("kind") != "t1":
        return
    d = node["data"]
    parent = (d.get("body") or "").replace("\n", " ").strip()
    replies = d.get("replies")
    if isinstance(replies, dict):
        for child in replies["data"]["children"]:
            if child.get("kind") != "t1":
                continue
            cd = child["data"]
            reply = (cd.get("body") or "").replace("\n", " ").strip()
            if (5 < len(reply) < 500 and 5 < len(parent) < 500
                    and "[deleted]" not in reply + parent
                    and "[removed]" not in reply + parent):
                out.append({"context": parent, "utterance": reply,
                            "post_title": post_title, "subreddit": sub,
                            # Reddit's crowd signal: the strongest non-Cooperative
                            # prior in a Cooperative-dominated register.
                            "score": cd.get("score", 0),
                            "controversial": cd.get("controversiality", 0) == 1})
            walk_comments(child, post_title, sub, out)


def scrape(subs, per_sub, sort, output):
    from predict import predict
    token = get_token()
    all_pairs = []
    for sub in subs:
        print(f"r/{sub}: fetching {sort} posts...")
        listing = api_get(f"/r/{sub}/{sort}?limit=25", token)
        if not listing:
            continue
        pairs = []
        for post in listing["data"]["children"]:
            pd_ = post["data"]
            if pd_.get("stickied"):
                continue
            time.sleep(1.5)
            # sort=controversial surfaces the contested comments first
            tree = api_get(f"/comments/{pd_['id']}?limit=100&depth=3&sort=controversial", token)
            if not tree or len(tree) < 2:
                continue
            for top in tree[1]["data"]["children"]:
                walk_comments(top, pd_["title"], sub, pairs)
            if len(pairs) >= per_sub * 6:  # gather a surplus, filter down
                break
        # rank by likelihood of being non-Cooperative. PRIMARY signal is the
        # crowd (controversial / downvoted) — validated-weak keyword+model pass
        # is secondary. See --validate: text filters barely beat base rate here.
        kept = []
        for p in pairs:
            pred = predict(p["utterance"], p["context"])
            reasons = noncoop_signals(p["utterance"], p["context"])
            crowd = (["controversial"] if p["controversial"] else []) + \
                    (["downvoted"] if p["score"] <= 0 else [])
            model_flag = pred["predicted_maxim"] != "Cooperative"
            flags = crowd + (["model"] if model_flag else []) + reasons
            if flags:
                kept.append({
                    "utterance": p["utterance"], "context": p["context"],
                    "predicted_maxim": pred["predicted_maxim"],
                    "predicted_violation_type": pred["violation_type"],
                    "confidence": f"{pred['confidence']:.3f}",
                    "score": p["score"],
                    # crowd signals first so annotators triage the strongest prior
                    "flag_reason": "|".join(flags),
                    "gold_maxim": "", "gold_violation_type": "",
                    "subreddit": sub, "post_title": p["post_title"],
                })
        # prioritise crowd-flagged, then keep the requested budget
        kept.sort(key=lambda r: (0 if ("controversial" in r["flag_reason"]
                  or "downvoted" in r["flag_reason"]) else 1, r["score"]))
        kept = kept[:per_sub]
        print(f"  kept {len(kept)}/{len(pairs)} flagged "
              f"({sum('controversial' in k['flag_reason'] or 'downvoted' in k['flag_reason'] for k in kept)} crowd-flagged)")
        all_pairs.extend(kept)

    if not all_pairs:
        print("nothing kept."); return
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_pairs[0].keys()))
        w.writeheader(); w.writerows(all_pairs)
    print(f"\nWrote {len(all_pairs)} flagged pairs -> {output}")
    print(f"Predicted dist: {Counter(p['predicted_maxim'] for p in all_pairs)}")
    print("Fill gold_maxim/gold_violation_type, then merge with merge_corpus.py --csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true", help="offline filter validation, no network")
    ap.add_argument("--subs", nargs="+", default=TARGET_SUBS)
    ap.add_argument("--per-sub", type=int, default=40, help="flagged pairs to keep per sub")
    ap.add_argument("--sort", default="top", choices=["hot", "top", "new", "controversial"],
                    help="'controversial'/'top' surface more violations")
    ap.add_argument("--output", default=str(RAW / "reddit_noncoop.csv"))
    a = ap.parse_args()
    if a.validate:
        validate()
    else:
        scrape(a.subs, a.per_sub, a.sort, a.output)
