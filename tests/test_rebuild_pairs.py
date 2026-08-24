"""Re-pairing (post_title, top-level comment) from the raw scrapes.

The guard that matters most is verify_depth_one: if a context is itself someone's
reply, pairing it with the post title invents an exchange that never happened.
"""

from src.rebuild_pairs import build, verify_depth_one


def raw(title, context, utterance, sub="askreddit"):
    return {"post_title": title, "context": context, "utterance": utterance,
            "subreddit": sub, "_file": "reddit_test.csv"}


Q = "What's the coolest place you've ever been?"
ANSWER = "The tunnels under the Magic Kingdom, they run the whole park."


def test_pairs_the_title_with_the_top_level_comment():
    pairs, _ = build([raw(Q, ANSWER, "How did you get down there?")], 6, True)
    assert len(pairs) == 1
    assert pairs[0]["context"] == Q
    assert pairs[0]["utterance"] == ANSWER
    assert pairs[0]["pairing"] == "title_to_toplevel"


def test_every_emitted_context_is_a_question():
    """The defect being fixed: 4% of the old test set had a question in it."""
    pairs, _ = build([raw(Q, ANSWER, "reply")], 6, True)
    assert all(p["context"].endswith("?") for p in pairs)


def test_non_question_titles_are_skipped_when_required():
    pairs, skipped = build([raw("TIFU by doing a thing", ANSWER, "reply")], 6, True)
    assert pairs == []
    assert skipped["title is not a question"] == 1


def test_non_question_titles_are_kept_when_allowed():
    pairs, _ = build([raw("TIFU by doing a thing", ANSWER, "reply")], 6, False)
    assert len(pairs) == 1


def test_the_same_comment_is_emitted_once_per_thread():
    """One top-level comment has many replies; it is still one answer."""
    rows = [raw(Q, ANSWER, f"reply {i}") for i in range(5)]
    pairs, skipped = build(rows, 6, True)
    assert len(pairs) == 1
    assert skipped["duplicate"] == 4


def test_per_thread_cap_limits_one_thread_from_dominating():
    rows = [raw(Q, f"a distinct top level comment number {i}", "r") for i in range(10)]
    pairs, skipped = build(rows, 3, True)
    assert len(pairs) == 3
    assert skipped["over per-thread cap"] == 7


def test_cap_applies_per_thread_not_globally():
    rows = ([raw("Q one?", f"answer {i} to one", "r") for i in range(4)]
            + [raw("Q two?", f"answer {i} to two", "r") for i in range(4)])
    pairs, _ = build(rows, 2, True)
    assert len(pairs) == 4
    assert len({p["context"] for p in pairs}) == 2


def test_short_and_overlong_answers_are_dropped():
    rows = [raw(Q, "ok", "r"), raw(Q, "x" * 2000, "r")]
    pairs, skipped = build(rows, 6, True)
    assert pairs == []
    assert skipped["answer too short or too long"] == 2


def test_labels_are_left_blank_for_the_annotator():
    """The old (comment, reply) label says nothing about (question, comment)."""
    pairs, _ = build([raw(Q, ANSWER, "r")], 6, True)
    assert pairs[0]["maxim"] == ""
    assert pairs[0]["violation_type"] == ""


def test_verify_depth_one_passes_when_no_context_is_a_reply():
    rows = [raw(Q, ANSWER, "a reply to it")]
    n, _ = verify_depth_one(rows)
    assert n == 0


def test_verify_depth_one_catches_a_mid_thread_context():
    """If a context also appears as a reply, it is not answering the title."""
    rows = [raw(Q, ANSWER, "a nested reply"),
            raw(Q, "a nested reply", "a deeper reply")]
    n, examples = verify_depth_one(rows)
    assert n == 1
    assert "a nested reply" in examples[0]
