"""
scrape_reddit.py

Pull comment-reply pairs from Reddit and pre-label them with the
fine-tuned model. Outputs a CSV for human annotation.

Uses Reddit's public JSON API (no auth needed, just append .json
to any Reddit URL). Rate-limited to be polite.

Usage:
    python scrape_reddit.py
    python scrape_reddit.py --subreddit askreddit --limit 100
    python scrape_reddit.py --subreddit cscareerquestions --output reddit_pairs.csv
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from predict import predict


USER_AGENT = "grice-maxim-classifier/1.0 (research project; pragmatics annotation)"


def fetch_json(url: str) -> dict:
    """fetch reddit JSON with rate limiting and a polite user agent."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for {url}")
        return None


def get_comment_pairs(subreddit: str, limit: int = 50) -> list:
    """
    Pull top-level comment + reply pairs from a subreddit's hot posts.
    Returns list of (context, utterance, post_title, permalink) tuples.

    The context is the parent comment, the utterance is the reply.
    This gives us natural conversational pairs where someone said
    something in response to something else — which is exactly what
    our classifier needs.
    """
    pairs = []

    # get hot posts
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
    print(f"Fetching posts from r/{subreddit}...")
    data = fetch_json(url)
    if not data:
        return pairs

    posts = data["data"]["children"]
    print(f"Found {len(posts)} posts.")

    for post in posts:
        post_data = post["data"]

        # skip pinned/stickied posts
        if post_data.get("stickied"):
            continue

        post_id = post_data["id"]
        post_title = post_data["title"]
        permalink = post_data["permalink"]

        # rate limit — reddit asks for 1 req/sec
        time.sleep(1.5)

        # fetch comments for this post
        comment_url = f"https://www.reddit.com{permalink}.json?limit=50&depth=2"
        comment_data = fetch_json(comment_url)
        if not comment_data or len(comment_data) < 2:
            continue

        comments = comment_data[1]["data"]["children"]

        for comment in comments:
            if comment["kind"] != "t1":
                continue

            parent_body = comment["data"].get("body", "")
            replies = comment["data"].get("replies", "")

            if not replies or isinstance(replies, str):
                continue

            reply_children = replies["data"]["children"]
            for reply in reply_children:
                if reply["kind"] != "t1":
                    continue

                reply_body = reply["data"].get("body", "")

                # skip deleted, removed, bot-like, or very long/short
                if any(skip in parent_body for skip in ["[deleted]", "[removed]"]):
                    continue
                if any(skip in reply_body for skip in ["[deleted]", "[removed]"]):
                    continue
                if len(reply_body) < 5 or len(reply_body) > 500:
                    continue
                if len(parent_body) < 5 or len(parent_body) > 500:
                    continue

                pairs.append({
                    "context": parent_body.replace("\n", " ").strip(),
                    "utterance": reply_body.replace("\n", " ").strip(),
                    "post_title": post_title,
                    "subreddit": subreddit,
                    "permalink": f"https://reddit.com{permalink}",
                })

                if len(pairs) >= limit:
                    return pairs

    return pairs


def scrape_and_label(subreddit: str, limit: int, output_path: str):
    """
    Scrape pairs, run them through the classifier, output CSV for annotation.
    """
    pairs = get_comment_pairs(subreddit, limit)
    print(f"\nCollected {len(pairs)} comment-reply pairs.")

    if not pairs:
        print("No pairs found. Try a different subreddit or check your connection.")
        return

    results = []
    print(f"Running classifier on {len(pairs)} pairs...\n")

    for i, pair in enumerate(pairs):
        pred = predict(pair["utterance"], pair["context"])
        results.append({
            "utterance": pair["utterance"],
            "context": pair["context"],
            "predicted_maxim": pred["predicted_maxim"],
            "predicted_violation_type": pred["violation_type"],
            "confidence": f"{pred['confidence']:.3f}",
            "gold_maxim": "",  # for you, human annotator
            "gold_violation_type": "",
            "subreddit": pair["subreddit"],
            "post_title": pair["post_title"],
        })
        print(f"  [{i+1}/{len(pairs)}] {pair['utterance'][:50]:<50} → {pred['predicted_maxim']} ({pred['confidence']:.0%})")

    # write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} pre-labeled pairs to {output_path}")
    print("Fill in gold_maxim and gold_violation_type, then merge into corpus.csv.")

    # distribution of predictions
    from collections import Counter
    pred_dist = Counter(r["predicted_maxim"] for r in results)
    print(f"\nPredicted distribution: {dict(pred_dist)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Reddit comment pairs and pre-label for annotation.",
    )
    parser.add_argument("--subreddit", default="askreddit", help="Subreddit to scrape.")
    parser.add_argument("--limit", type=int, default=50, help="Max pairs to collect.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "data" / "raw" / "reddit_pairs.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()
    scrape_and_label(args.subreddit, args.limit, args.output)
