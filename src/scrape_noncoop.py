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

from labels import MAXIMS

from paths import CORPUS_PATH as CORPUS, RAW_DIR as RAW, ROOT
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


# ------------------------------- LLM pre-screen -------------------------------
# A far stronger flagger than the regex pass: ask Claude to classify each reply.
# Off by default (needs ANTHROPIC_API_KEY + credits). Default model is
# claude-opus-4-8; for a high-volume screen, claude-haiku-4-5 is ~5x cheaper and
# well-suited to classification — pass --model claude-haiku-4-5 to use it.
_llm = None
LLM_MODEL_DEFAULT = "claude-opus-5"

LLM_SYSTEM = (
    "You judge whether the REPLY in a two-turn exchange respects Grice's maxims, "
    "given the message it responds to. Output the single best label:\n"
    "- Cooperative: relevant, truthful, appropriately informative, and clear.\n"
    "- Quantity: too much or too little information (incl. non-answers / refusals).\n"
    "- Quality: says something false, unsupported, or ironic/sarcastic.\n"
    "- Relation: changes the subject / does not address the question.\n"
    "- Manner: obscure, ambiguous, disorganized, or needlessly long.\n"
    "Pick the maxim most VIOLATED; use Cooperative only if none is violated. "
    "Judge the reply's pragmatics, not its topic or subreddit."
)
LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": MAXIMS},
        "confidence": {"type": "number"},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}


def _get_llm():
    global _llm
    if _llm is None:
        import anthropic
        _llm = anthropic.Anthropic()
    return _llm


def llm_label(utterance, context, model):
    """Return (label, confidence) for the reply, or (None, None) on error."""
    try:
        resp = _get_llm().messages.create(
            model=model, max_tokens=200, system=LLM_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": LLM_SCHEMA}},
            messages=[{"role": "user", "content":
                       f"Preceding message: {context}\n\nReply to classify: {utterance}"}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        d = json.loads(text)
        return d["label"], float(d.get("confidence", 0.0))
    except Exception as e:
        print(f"  llm error: {type(e).__name__}: {e}")
        return None, None


def violation_score(label, confidence):
    """Map (label, confidence) onto a 0..1 violation axis: high = confident
    violation, ~0.5 = uncertain, low = confident Cooperative."""
    if label is None:
        return float("nan")
    return confidence if label != "Cooperative" else (1.0 - confidence)


# --------------------------- stratified batch logic ---------------------------
# Pure functions (no network / no LLM) so they're unit-testable offline.
def stratify(scored, n_conf_viol, n_mid, n_lowconf_coop, rng):
    """scored: list of dicts each with llm_label, llm_confidence. Returns the
    same dicts tagged with a 'stratum', drawn WITHOUT replacement from three
    regions of Claude's confidence space. rng is a seeded random.Random."""
    pool = [dict(r) for r in scored if r.get("llm_label") is not None]
    used = set()

    def take(cands, n, stratum):
        picked = []
        for r in cands:
            if id(r) in used:
                continue
            used.add(id(r))
            r["stratum"] = stratum
            picked.append(r)
            if len(picked) >= n:
                break
        return picked

    # (a) confident violations: label != Cooperative, highest confidence first
    viol = sorted((r for r in pool if r["llm_label"] != "Cooperative"),
                  key=lambda r: -r["llm_confidence"])
    batch = take(viol, n_conf_viol, "confident_violation")

    # (c) low-confidence Cooperative: label == Cooperative, LEAST confident first
    #     — the discard pool's most-suspect members (covert violations hide here)
    coop = sorted((r for r in pool if r["llm_label"] == "Cooperative"),
                  key=lambda r: r["llm_confidence"])
    batch += take(coop, n_lowconf_coop, "low_conf_cooperative")

    # (b) middle: whatever's left, most-uncertain first (confidence nearest 0.5)
    rest = [r for r in pool if id(r) not in used]
    rest.sort(key=lambda r: abs(r["llm_confidence"] - 0.5))
    batch += take(rest, n_mid, "mid_confidence")
    return batch


def compare_stats(pre_rows, rand_rows):
    """Both lists are annotated rows with 'gold_maxim' and 'surface_proxy_present'
    filled. Returns violation rate + surface-marker skew, prescreened vs random,
    plus per-stratum violation rate in the prescreened batch."""
    def is_viol(r):
        return str(r.get("gold_maxim", "")).strip() not in ("", "Cooperative", "nan")

    def truthy(v):
        return str(v).strip().lower() in ("true", "yes", "1", "y", "t")

    def summarize(rows):
        annotated = [r for r in rows if str(r.get("gold_maxim", "")).strip() not in ("", "nan")]
        viols = [r for r in annotated if is_viol(r)]
        surf = sum(truthy(r.get("surface_proxy_present")) for r in viols)
        return {
            "annotated": len(annotated),
            "violations": len(viols),
            "violation_rate": (len(viols) / len(annotated)) if annotated else float("nan"),
            "surface_marked_violations": surf,
            "surface_marked_rate": (surf / len(viols)) if viols else float("nan"),
        }

    out = {"prescreened": summarize(pre_rows), "random": summarize(rand_rows)}
    # per-stratum violation rate within the prescreened batch
    strata = {}
    for r in pre_rows:
        s = r.get("stratum", "?")
        strata.setdefault(s, []).append(r)
    out["per_stratum"] = {
        s: {"n": len(rs),
            "violations": sum(is_viol(x) for x in rs if str(x.get("gold_maxim","")).strip() not in ("","nan")),
            "annotated": sum(str(x.get("gold_maxim","")).strip() not in ("","nan") for x in rs)}
        for s, rs in strata.items()
    }
    return out


# ----------------------------- offline validation -----------------------------
def validate(llm_prescreen=False, model=LLM_MODEL_DEFAULT):
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
            row = {
                "sub": r.get("subreddit", "?"),
                "truth_noncoop": truth != "Cooperative",
                "truth": truth,
                "model_flag": str(r.get("predicted_maxim", "")).strip() not in ("", "Cooperative", "nan"),
                "kw": noncoop_signals(str(r["utterance"]), str(r.get("context", ""))),
                "utterance": str(r["utterance"]), "context": str(r.get("context", "")),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    df["kw_flag"] = df["kw"].map(lambda x: len(x) > 0)

    flags = ["model_flag", "kw_flag"]
    if llm_prescreen:
        print(f"LLM pre-screen with {model} over {len(df)} rows "
              f"(~{len(df)} API calls — this costs money)...")
        labels = [llm_label(u, c, model)[0] for u, c in zip(df["utterance"], df["context"])]
        df["llm_flag"] = [(l is not None and l != "Cooperative") for l in labels]
        flags.append("llm_flag")
    df["union_flag"] = df[flags].any(axis=1)

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
    for flag in flags + ["union_flag"]:
        tp, fp, fn, p, r, f = prf(flag)
        print(f"{flag:<12}{'ALL':<20}{tp:>4}{fp:>4}{fn:>4}{p:>7.2f}{r:>7.2f}{f:>7.2f}")
    print()
    for sub in TARGET_SUBS:
        for flag in flags + ["union_flag"]:
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


def _usable(*texts):
    """Shared length and deletion filter for any pair we emit."""
    blob = "".join(texts).lower()
    return (all(5 < len(t) < 500 for t in texts)
            and "[deleted]" not in blob and "[removed]" not in blob
            and "removed by reddit" not in blob)


# How to turn a comment tree into (context, utterance) pairs.
#
#   parent_reply     top-level comment -> its reply. The original behaviour.
#   title_toplevel   post title -> top-level comment. On AskReddit-style subs
#                    the title is the only actual question in the exchange.
#
# parent_reply built the first corpus, which is how comment->reply pairs ended
# up labelled "underinformative answer" with no question in them (4% of
# test_natural contexts have a question mark vs 88% of synthetic). See
# results/foundation_report.md §11. parent_reply is kept because reply chains
# are where disagreement and sarcasm live.
PAIRINGS = ("title_toplevel", "parent_reply", "both")


def walk_comments(node, post_title, sub, out, pairing="title_toplevel", depth=0):
    """Recurse the comment tree, emitting pairs according to `pairing`."""
    if not node or node.get("kind") != "t1":
        return
    d = node["data"]
    parent = (d.get("body") or "").replace("\n", " ").strip()

    # the post title answered by this top-level comment
    if pairing in ("title_toplevel", "both") and depth == 0:
        title = (post_title or "").replace("\n", " ").strip()
        if _usable(parent) and len(title) > 5:
            out.append({"context": title, "utterance": parent,
                        "post_title": post_title, "subreddit": sub,
                        "pairing": "title_toplevel",
                        "score": d.get("score", 0),
                        "controversial": d.get("controversiality", 0) == 1})

    replies = d.get("replies")
    if isinstance(replies, dict):
        for child in replies["data"]["children"]:
            if child.get("kind") != "t1":
                continue
            cd = child["data"]
            reply = (cd.get("body") or "").replace("\n", " ").strip()
            if pairing in ("parent_reply", "both") and _usable(reply, parent):
                out.append({"context": parent, "utterance": reply,
                            "post_title": post_title, "subreddit": sub,
                            "pairing": "parent_reply",
                            # Reddit's crowd signal: the strongest non-Cooperative
                            # prior in a Cooperative-dominated register.
                            "score": cd.get("score", 0),
                            "controversial": cd.get("controversiality", 0) == 1})
            walk_comments(child, post_title, sub, out, pairing, depth + 1)


def scrape(subs, per_sub, sort, output, llm_prescreen=False, model=LLM_MODEL_DEFAULT,
           pairing="title_toplevel"):
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
                walk_comments(top, pd_["title"], sub, pairs, pairing)
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
            llm = []
            if llm_prescreen:
                lab, _ = llm_label(p["utterance"], p["context"], model)
                if lab and lab != "Cooperative":
                    llm = [f"llm:{lab}"]
            flags = crowd + (["model"] if model_flag else []) + reasons + llm
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


# ------------------------- stratified batch builder ---------------------------
BATCH_COLS = ["utterance", "context", "subreddit", "post_title", "batch",
              "stratum", "llm_label", "llm_confidence", "violation_score",
              "gold_maxim", "gold_violation_type", "surface_proxy_present", "notes"]


def gather_pool(subs, per_sub, sort, token, pairing="title_toplevel"):
    """Fetch a broad pool of pairs (no scoring). sort=hot keeps it
    representative — we deliberately do NOT lead with controversial here."""
    pool = []
    for sub in subs:
        print(f"r/{sub}: gathering {sort} comments...")
        listing = api_get(f"/r/{sub}/{sort}?limit=25", token)
        if not listing:
            continue
        got = []
        for post in listing["data"]["children"]:
            pd_ = post["data"]
            if pd_.get("stickied"):
                continue
            time.sleep(1.5)
            tree = api_get(f"/comments/{pd_['id']}?limit=100&depth=3", token)
            if not tree or len(tree) < 2:
                continue
            for top in tree[1]["data"]["children"]:
                walk_comments(top, pd_["title"], sub, got, pairing)
            if len(got) >= per_sub:
                break
        pool.extend(got[:per_sub])
    return pool


def build_batches(subs, pool_per_sub, sort, model, out_prefix,
                  n_conf_viol=20, n_mid=15, n_lowconf_coop=15, n_random=40, pool=None,
                  pairing="title_toplevel"):
    """Produce two annotation CSVs from the same three subs:
       <prefix>_prescreened.csv  — LLM-scored, stratified (confident violation /
                                   mid confidence / low-confidence Cooperative)
       <prefix>_random.csv       — raw random sample, UNSCREENED, scores hidden
                                   (annotate the slow way; the bias control).
    If `pool` (a list of {utterance, context, subreddit, post_title} dicts) is
    given, it's used instead of scraping — same machinery, offline source."""
    import random
    rng = random.Random(42)
    if pool is None:
        token = get_token()
        pool = gather_pool(subs, pool_per_sub, sort, token, pairing)
    if not pool:
        print("empty pool."); return
    rng.shuffle(pool)

    # carve the random batch off FIRST, before any scoring, so it's disjoint and
    # untouched by the LLM. scores stay blank so they can't anchor the annotator.
    random_batch = pool[:n_random]
    screen_pool = pool[n_random:]

    print(f"scoring {len(screen_pool)} candidates with {model}...")
    scored = []
    for p in screen_pool:
        lab, conf = llm_label(p["utterance"], p["context"], model)
        if lab is None:
            continue
        scored.append({**p, "llm_label": lab, "llm_confidence": conf,
                       "violation_score": violation_score(lab, conf)})

    strat = stratify(scored, n_conf_viol, n_mid, n_lowconf_coop, rng)

    def row(p, batch, stratum="", scored=False):
        return {
            "utterance": p["utterance"], "context": p["context"],
            "subreddit": p["subreddit"], "post_title": p["post_title"],
            "batch": batch, "stratum": stratum,
            "llm_label": p.get("llm_label", "") if scored else "",
            "llm_confidence": f"{p['llm_confidence']:.3f}" if scored else "",
            "violation_score": f"{p['violation_score']:.3f}" if scored else "",
            "gold_maxim": "", "gold_violation_type": "",
            "surface_proxy_present": "", "notes": "",
        }

    pre_rows = [row(p, "prescreened", p["stratum"], scored=True) for p in strat]
    rand_rows = [row(p, "random", scored=False) for p in random_batch]

    for name, rows in [("prescreened", pre_rows), ("random", rand_rows)]:
        path = f"{out_prefix}_{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=BATCH_COLS)
            w.writeheader(); w.writerows(rows)
        print(f"wrote {len(rows)} -> {path}")
    print("\nStrata:", dict(Counter(r["stratum"] for r in pre_rows)))
    print("Annotate gold_maxim + surface_proxy_present in BOTH files (random "
          "first, blind), then: scrape_noncoop.py --compare <pre>.csv <rand>.csv")


def compare_batches(prescreened_csv, random_csv):
    import pandas as pd
    pre = pd.read_csv(prescreened_csv).to_dict("records")
    rand = pd.read_csv(random_csv).to_dict("records")
    s = compare_stats(pre, rand)
    p, r = s["prescreened"], s["random"]
    print(f"{'batch':<14}{'annotated':>10}{'violations':>12}{'viol_rate':>11}"
          f"{'surf_marked':>13}{'surf_rate':>11}")
    for name, d in [("prescreened", p), ("random", r)]:
        print(f"{name:<14}{d['annotated']:>10}{d['violations']:>12}"
              f"{d['violation_rate']:>11.2f}{d['surface_marked_violations']:>13}"
              f"{d['surface_marked_rate']:>11.2f}")
    print("\nPer-stratum violation rate in the prescreened batch "
          "(watch low_conf_cooperative — violations there are the LLM's misses):")
    for st, d in s["per_stratum"].items():
        rate = d["violations"] / d["annotated"] if d["annotated"] else float("nan")
        print(f"  {st:<22} {d['violations']}/{d['annotated']}  ({rate:.2f})")
    print("\nSelection-bias read: if prescreened surf_rate >> random surf_rate, "
          "the LLM screen is preferentially surfacing surface-MARKED violations "
          "and under-sampling covert ones.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true", help="offline filter validation, no network")
    ap.add_argument("--build-batches", action="store_true",
                    help="scrape a pool, emit stratified prescreened + random-unscreened annotation CSVs")
    ap.add_argument("--compare", nargs=2, metavar=("PRESCREENED", "RANDOM"),
                    help="after annotation: compare violation rate + surface-marker skew")
    ap.add_argument("--subs", nargs="+", default=TARGET_SUBS)
    ap.add_argument("--per-sub", type=int, default=40, help="flagged pairs to keep per sub (scrape mode)")
    ap.add_argument("--pool-per-sub", type=int, default=120, help="candidates to gather per sub (batch mode)")
    ap.add_argument("--sort", default="hot", choices=["hot", "top", "new", "controversial"])
    ap.add_argument("--llm-prescreen", action="store_true",
                    help="add a Claude classifier as a flagger (needs ANTHROPIC_API_KEY + credits)")
    ap.add_argument("--model", default=LLM_MODEL_DEFAULT,
                    help="LLM model; claude-haiku-4-5 is ~5x cheaper for this classification")
    ap.add_argument("--out-prefix", default=str(RAW / "reddit_batch"))
    ap.add_argument("--output", default=str(RAW / "reddit_noncoop.csv"))
    ap.add_argument("--pairing", default="title_toplevel", choices=PAIRINGS,
                    help="title_toplevel: post title -> top-level comment, a real "
                         "question-answer pair (default). parent_reply: the old "
                         "comment -> reply pairing, which produced pairs with no "
                         "question in them. both: emit each pair type.")
    a = ap.parse_args()
    if a.compare:
        compare_batches(a.compare[0], a.compare[1])
    elif a.validate:
        validate(llm_prescreen=a.llm_prescreen, model=a.model)
    elif a.build_batches:
        build_batches(a.subs, a.pool_per_sub, a.sort, a.model, a.out_prefix,
                      pairing=a.pairing)
    else:
        scrape(a.subs, a.per_sub, a.sort, a.output,
               llm_prescreen=a.llm_prescreen, model=a.model, pairing=a.pairing)
