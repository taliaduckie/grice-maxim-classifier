"""Comment-tree pairing.

The defect this guards against: pairing a top-level comment with its reply
produces a (context, utterance) pair containing no question, then labelling the
utterance "underinformative" relative to a question nobody asked. See
results/foundation_report.md §11.
"""

import pytest

from src.scrape_noncoop import _usable, walk_comments


def comment(body, score=1, controversial=0, replies=None):
    node = {"kind": "t1", "data": {"body": body, "score": score,
                                   "controversiality": controversial}}
    if replies:
        node["data"]["replies"] = {"data": {"children": replies}}
    return node


TITLE = "What's the coolest place you've ever been?"

TREE = comment(
    "The tunnels under the Magic Kingdom, they run the whole park.",
    score=120,
    replies=[
        comment("How did you get down there?", score=8,
                replies=[comment("A cast member walked us in.", score=3)]),
        comment("Pics or it didn't happen", score=-4, controversial=1),
    ],
)


def collect(pairing):
    out = []
    walk_comments(TREE, TITLE, "askreddit", out, pairing)
    return out


def test_title_toplevel_pairs_the_question_with_the_answer():
    pairs = collect("title_toplevel")
    assert len(pairs) == 1
    assert pairs[0]["context"] == TITLE
    assert pairs[0]["utterance"].startswith("The tunnels")
    assert pairs[0]["pairing"] == "title_toplevel"


def test_title_toplevel_does_not_emit_deeper_comments():
    """Only depth 0 answers the title; a reply answers its parent, not the post."""
    for p in collect("title_toplevel"):
        assert p["context"] == TITLE


def test_parent_reply_reproduces_the_old_behaviour():
    pairs = collect("parent_reply")
    contexts = {p["context"] for p in pairs}
    assert TITLE not in contexts
    assert any(p["utterance"] == "Pics or it didn't happen" for p in pairs)
    assert all(p["pairing"] == "parent_reply" for p in pairs)


def test_parent_reply_recurses_past_the_first_level():
    """The nested reply must still be reached."""
    pairs = collect("parent_reply")
    assert any(p["utterance"] == "A cast member walked us in." for p in pairs)
    assert any(p["context"] == "How did you get down there?" for p in pairs)


def test_both_emits_each_kind_once():
    pairs = collect("both")
    kinds = [p["pairing"] for p in pairs]
    assert kinds.count("title_toplevel") == 1
    assert kinds.count("parent_reply") == len(collect("parent_reply"))


def test_the_defect_being_fixed():
    """Old pairing yields pairs with no question; new pairing always has one."""
    old = collect("parent_reply")
    new = collect("title_toplevel")
    assert not any("?" in p["context"] for p in old if p["context"] == TITLE)
    assert all("?" in p["context"] for p in new)


def test_crowd_signal_comes_from_the_utterance_not_the_context():
    """The score must describe the comment being labelled."""
    top = collect("title_toplevel")[0]
    assert top["score"] == 120                      # the top-level comment's own
    downvoted = [p for p in collect("parent_reply")
                 if p["utterance"] == "Pics or it didn't happen"][0]
    assert downvoted["score"] == -4
    assert downvoted["controversial"] is True


def test_post_title_is_retained_on_every_pair():
    for p in collect("both"):
        assert p["post_title"] == TITLE


def test_deleted_and_short_bodies_are_dropped():
    assert not _usable("[deleted]")
    assert not _usable("hi")
    assert not _usable("x" * 600)
    assert _usable("a genuinely usable comment body")


def test_deleted_toplevel_comment_yields_no_title_pair():
    out = []
    walk_comments(comment("[removed]"), TITLE, "askreddit", out, "title_toplevel")
    assert out == []


def test_non_comment_nodes_are_ignored():
    out = []
    walk_comments({"kind": "more", "data": {}}, TITLE, "askreddit", out, "both")
    assert out == []


def test_missing_title_does_not_produce_a_pair():
    out = []
    walk_comments(TREE, "", "askreddit", out, "title_toplevel")
    assert out == []
