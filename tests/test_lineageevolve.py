"""Unit tests for LineageEvolve search-state logic (no LLM, no evaluator)."""

import math
import random

import pytest

from skydiscover.config import LineageEvolveDatabaseConfig
from skydiscover.search.base_database import Program
from skydiscover.search.lineageevolve.attribution import (
    build_inventory,
    code_hash,
    compute_shares,
    parse_intent,
    validate_labels,
)
from skydiscover.search.lineageevolve.database import LineageEvolveDatabase
from skydiscover.search.lineageevolve.memory import StrategyMemory
from skydiscover.search.lineageevolve.tree import AttemptRecord, LineageTree


def rec(i, parent, status="admitted", score=None, pscore=None):
    return AttemptRecord(
        id=f"A{i}", parent=parent, island=0, iteration=i, status=status, score=score,
        parent_score=pscore, delta=None if score is None else score - pscore,
    )


def test_prior_and_backup_follow_spec():
    t = LineageTree(0, gamma=0.9, kappa=2.0)
    root = t.add_root("r", 1.0)
    assert (root.alpha, root.beta) == (1.0, 1.0)

    # successful edit r -> a
    y = t.backup(root, True, 2.0, q=1.0)
    assert y == 1 and (root.alpha, root.beta) == (2.0, 1.0)
    r1 = rec(1, "r", score=2.0, pscore=1.0)
    a = t.add_child(root, "a", 2.0, r1, 1)
    assert a.mu0 == pytest.approx(2 / 3)  # parent's mean AFTER backup
    assert (a.alpha, a.beta) == (pytest.approx(4 / 3), pytest.approx(2 / 3))
    t.update_scalars(r1)
    assert t.theta == pytest.approx(1.0) and t.best_score == 2.0

    # regression a -> b (score 1.5): failure for a and for root
    y = t.backup(a, True, 1.5, q=1.0)
    assert y == 0
    assert a.beta == pytest.approx(2 / 3 + 1.0)
    assert root.beta == pytest.approx(1.0 + 0.9)
    b = t.add_child(a, "b", 1.5, rec(2, "a", score=1.5, pscore=2.0), 2)

    # repair b -> c (score 1.8): success for b only; must beat path max (2.0) for a, root
    a_alpha, root_alpha = a.alpha, root.alpha
    y = t.backup(b, True, 1.8, q=0.5)
    assert y == 1
    assert b.alpha == pytest.approx(t.kappa * b.mu0 + 0.5)
    assert a.alpha == a_alpha and root.alpha == root_alpha
    assert a.beta == pytest.approx(2 / 3 + 1.0 + 0.9 * 0.5)
    assert root.beta == pytest.approx(1.9 + 0.81 * 0.5)


def test_backup_switch_and_crash():
    t = LineageTree(0)
    root = t.add_root("r", 1.0)
    t.backup(root, True, 2.0, 1.0)
    a = t.add_child(root, "a", 2.0, rec(1, "r", score=2.0, pscore=1.0), 1)
    before = (root.alpha, root.beta)
    assert t.backup(a, False, None, 1.0, to_ancestors=False) == 0
    assert (root.alpha, root.beta) == before
    assert a.beta == pytest.approx(2 / 3 + 1.0)


def test_theta_ema_and_individual_reward():
    t = LineageTree(0, rho=0.9, theta_mode="ema")
    root = t.add_root("r", 1.0)
    assert t.individual_reward(root) == 0.0  # theta undefined
    t.update_scalars(rec(1, "r", score=2.0, pscore=1.0))
    t.update_scalars(rec(2, "r", score=1.5, pscore=1.0))
    assert t.theta == pytest.approx(0.9 * 1.0 + 0.1 * 0.5)
    t.update_scalars(rec(3, "r", score=0.5, pscore=1.0))  # not progress
    assert t.theta == pytest.approx(0.95) and t.attempts_since_progress == 1
    assert t.individual_reward(root) == pytest.approx((1.0 - 2.0) / 0.95)
    t.update_scalars(rec(4, "r", status="crashed"))
    assert t.attempts_since_progress == 2 and t.best_score == 2.0


def test_selection_backtracks_like_spec_example():
    # best node Beta(1,5), R=0; node one step behind Beta(4/3,2/3), R=-1
    t = LineageTree(0)
    best = t.add_root("best", 2.0)
    other = t.add_root("other", 1.0)
    t.theta = 1.0
    best.alpha, best.beta = 1.0, 5.0
    other.alpha, other.beta = 4 / 3, 2 / 3
    assert math.log(best.p_mean) == pytest.approx(-1.79, abs=0.01)
    rng = random.Random(0)
    picks = [t.select(rng)[0].id for _ in range(4000)]
    assert picks.count("other") > picks.count("best")
    # reproducible
    assert [t.select(random.Random(1))[0].id for _ in range(5)] == [
        t.select(random.Random(1))[0].id for _ in range(5)
    ]


def test_tree_roundtrip():
    t = LineageTree(0)
    root = t.add_root("r", 1.0)
    r1 = rec(1, "r", score=2.0, pscore=1.0)
    t.backup(root, True, 2.0, 1.0)
    t.add_child(root, "a", 2.0, r1, 1)
    root.attempts.append(r1)
    t.update_scalars(r1)
    t2 = LineageTree.from_dict(t.to_dict())
    assert t2.to_dict() == t.to_dict()


def test_hash_and_intent():
    a = "# EVOLVE-BLOCK-START\nx = 1\n\n   y  =  2\n# EVOLVE-BLOCK-END\nprint(1)"
    b = "# EVOLVE-BLOCK-START\nx = 1\ny = 2\n# EVOLVE-BLOCK-END\nprint(2)"
    assert code_hash(a) == code_hash(b)
    assert code_hash(a) != code_hash(a.replace("x = 1", "x = 3"))

    i = parse_intent("<intent>\nstrategy: new: Hex Grid Init\nsources: [S3, C1]\ndescription: Use hex grid.\n</intent>\n```python\nx=1\n```")
    assert i["strategy"] == "new:hex_grid_init" and i["sources"] == ["S3", "C1"]
    assert i["description"] == "Use hex grid." and i["parsed"]
    assert parse_intent("<intent>\nstrategy: S12\nsources: []\ndescription: d\n</intent>")["strategy"] == "S12"
    missing = parse_intent("```python\nx=1\n```")
    assert missing["strategy"] == "unspecified" and missing["sources"] == [] and not missing["parsed"]


PARENT = """import numpy as np

def f():
    a = 1
    b = 2
    return a + b

def g():
    return 3
"""
CHILD = """import numpy as np

def f():
    a = 10
    b = 2
    c = 5
    return a + b + c

def g():
    x = borrowed()
    y = borrowed2()
    return x + y
"""


def test_inventory_validation_and_shares():
    units = build_inventory(PARENT, CHILD)
    assert [u.symbol for u in units] == ["f", "g"]  # hunks in f merged
    assert sum(u.size for u in units) == units[0].size + units[1].size
    ctx = {"C1": "def g():\n    x = borrowed()\n    y = borrowed2()\n    return 0\n"}

    labels = validate_labels(
        {"U1": "ext:S3", "U2": "ext:C1"}, units, {"S3"}, {"S3"}, ctx, PARENT
    )
    assert labels == {"U1": "ext:S3", "U2": "ext:C1"}
    q, ext = compute_shares(units, labels)
    assert q == 0.0 and sum(ext.values()) == pytest.approx(1.0)

    # S3 not declared in intent sources -> lineage; C2 unknown -> lineage
    labels = validate_labels({"U1": "ext:S3", "U2": "ext:C2"}, units, {"S3"}, set(), ctx, PARENT)
    assert labels == {"U1": "lineage", "U2": "lineage"}
    # context program without the lines -> lineage; novel counts toward lineage
    labels = validate_labels({"U1": "novel", "U2": "ext:C1"}, units, set(), set(), {"C1": "pass"}, PARENT)
    assert labels == {"U1": "novel", "U2": "lineage"}
    assert compute_shares(units, labels)[0] == 1.0
    # unknown unit id / abstain -> invalid interpretation
    assert validate_labels({"U9": "lineage"}, units, set(), set(), {}, PARENT) is None
    assert validate_labels(None, units, set(), set(), {}, PARENT) is None
    assert build_inventory(PARENT, PARENT) == []


def test_memory_utility_and_retrieval():
    m = StrategyMemory()
    s1, s2, s3 = m.create("good"), m.create("bad"), m.create("fresh")
    assert m.create("good") is s1
    toks = {"p": frozenset("abcd"), "far": frozenset("wxyz")}
    m.add_evidence(s1.id, "A1", "p", 1.0, 1)
    m.add_evidence(s2.id, "A2", "p", 0.5, 0)
    m.add_evidence(s2.id, "A3", "far", 1.0, 0)  # dissimilar: weight 0
    assert m.utility(s1, toks["p"], toks.get) == (2.0, 1.0)
    assert m.utility(s2, toks["p"], toks.get) == (1.0, 1.5)
    lessons, warnings = m.retrieve(toks["p"], toks.get, {s1.id}, random.Random(0), 1, 1, forced=s3.id)
    assert [x["strategy"].id for x in lessons] == [s3.id]
    assert [x["strategy"].id for x in warnings] == [s2.id]
    assert StrategyMemory.from_dict(m.to_dict()).to_dict() == m.to_dict()


def prog(pid, score, code, parent=None):
    return Program(id=pid, solution=code, metrics={"combined_score": score}, parent_id=parent)


def make_db(**kw):
    cfg = LineageEvolveDatabaseConfig(num_islands=2, population_size=20, **kw)
    return LineageEvolveDatabase("lineageevolve", cfg)


def test_database_commit_flow_and_checkpoint(tmp_path):
    db = make_db()
    db.add(prog("init", 1.0, "x = 0"))
    db.add(prog("init2", 1.0, "x = 0"), target_island=1)
    assert db.trees[0].roots == ["init"] and db.trees[1].roots == ["init2"]
    assert db.use_paradigm_breakthrough is True  # AdaEvolve paradigm kept (default)

    parent, ctx = db.sample(3)
    assert list(parent.values())[0].id == "init"
    assert all(p.id != "init" for p in ctx[""])

    intent = {"strategy": "new:tune_x", "sources": [], "description": "set x"}
    r1 = db.commit_attempt(
        parent_id="init", island=0, iteration=1, status="admitted", intent=intent,
        prompt_sources={}, child_program=prog("c1", 2.0, "x = 1", "init"),
        interpretation={"valid": True, "q": 0.5, "ext_shares": {"C1": 0.5},
                        "lesson": {"summary": "s", "mechanism": "m", "conditions": ["c"]}},
    )
    root = db.trees[0].nodes["init"]
    assert r1.y_parent == 1 and r1.q == 0.5 and root.alpha == pytest.approx(1.5)
    child = db.trees[0].nodes["c1"]
    assert child.mu0 == pytest.approx(1.5 / 2.5) and child.depth == 1
    s = db.memory.find_by_name("tune_x")
    assert s.summary == "s" and s.evidence == [{"attempt": "A1", "program": "init", "share": 0.5, "y": 1}]
    assert r1.shares == {s.id: 0.5}  # context-program share credits no strategy
    assert db.trees[0].theta == pytest.approx(1.0) and db.best_program_id == "c1"

    visits = db.adapter.dimension_visits[0]
    h = code_hash("x = 1")
    assert h in db.hash_index
    r2 = db.commit_attempt(parent_id="c1", island=0, iteration=2, status="duplicate",
                           intent={"strategy": s.id, "sources": [], "description": "again"},
                           prompt_sources={}, candidate_hash=h)
    r3 = db.commit_attempt(parent_id="c1", island=0, iteration=3, status="crashed",
                           intent={"strategy": "unspecified"}, prompt_sources={},
                           error="timeout", candidate_hash="deadbeef")
    assert db.adapter.dimension_visits[0] == visits  # failures: reported per iteration
    evals = db.adapter.states[0].total_evaluations
    db.record_failed_visit(0)
    assert db.adapter.dimension_visits[0] == visits + 1  # visible to the scheduler
    assert db.adapter.states[0].total_evaluations == evals  # spawn productivity untouched
    assert r2.q == 1.0 and r2.shares == {s.id: 1.0} and r3.shares == {}
    assert "deadbeef" in db.hash_index
    assert child.beta == pytest.approx(2 * (1 - child.mu0) + 2.0)
    assert root.beta == pytest.approx(1.0 + 2 * 0.9)
    assert db.trees[0].attempts_since_progress == 2 and len(db.trees[0].nodes) == 2
    # strategy used in the attempt log of c1 is filtered from retrieval
    assert db.retrieve_strategies(child) == ([], [])

    # import carries the source node's posterior mean, no evidence
    db.add(prog("mig", 2.0, "x = 1", parent="c1"), target_island=1)
    mig = db.trees[1].nodes["mig"]
    assert mig.is_root and mig.mu0 == pytest.approx(child.p_mean)
    assert mig.provenance["type"] == "import" and mig.attempts == []
    assert db.trees[1].best_score == 2.0

    path = str(tmp_path / "ckpt")
    db.save(path, 3)
    db2 = make_db()
    db2.load(path)
    assert [t.to_dict() for t in db2.trees] == [t.to_dict() for t in db.trees]
    assert db2.memory.to_dict() == db.memory.to_dict()
    assert db2.hash_index == db.hash_index
    assert db2.rng.random() == db.rng.random()
    assert set(db2.node_island) <= set(db2.programs)


def test_meta_trigger_and_switches():
    db = make_db(stagnation_window=2, use_inherited_prior=False, use_source_attribution=False,
                 meta_guidance_mode="lineage", stagnation_signal="progress")
    db.add(prog("init", 1.0, "x = 0"))
    assert not db.should_trigger_meta(0)
    for i in range(2):
        db.commit_attempt(parent_id="init", island=0, iteration=i + 1, status="crashed",
                          intent={}, prompt_sources={}, error="e")
    assert db.should_trigger_meta(0)
    ids = db.register_meta_ideas(0, [{"name": "idea_a", "summary": "s"}, {"name": "idea_b", "summary": "t"}], 3)
    assert len(ids) == 2 and not db.should_trigger_meta(0)  # pending + cooldown
    lessons, _ = db.retrieve_strategies(db.trees[0].nodes["init"], forced=ids[0])
    assert lessons[0]["strategy"].id == ids[0] and len(lessons) == 2
    assert db.trees[0].pending_meta_ideas == []

    r = db.commit_attempt(parent_id="init", island=0, iteration=4, status="admitted", intent={},
                          prompt_sources={}, child_program=prog("c", 2.0, "x = 9", "init"),
                          interpretation={"valid": True, "q": 0.2, "ext_shares": {ids[0]: 0.8}})
    assert r.q == 1.0 and r.ext_shares == {}  # attribution switched off
    assert db.trees[0].nodes["c"].mu0 == 0.5  # inherited prior switched off


def test_message_text_handles_part_lists():
    from skydiscover.llm.openai import _message_text

    assert _message_text("plain") == "plain"
    assert _message_text(None) is None
    parts = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "hidden"}], "closed": True},
        {"type": "text", "text": "answer ```python\nx=1\n```"},
    ]
    assert _message_text(parts) == "answer ```python\nx=1\n```"


def test_local_evidence_max_backup_and_visits():
    t = LineageTree(0)
    root = t.add_root("r", 1.0)
    assert (root.alpha_loc, root.beta_loc, root.best_sub) == (1.0, 1.0, 1.0)
    t.record_visits(["r"])
    t.backup(root, True, 2.0, q=0.5)
    t.backup_max(root, 2.0)
    a = t.add_child(root, "a", 2.0, rec(1, "r", score=2.0, pscore=1.0), 1)
    assert (root.alpha_loc, root.beta_loc) == (1.5, 1.0)
    assert t.root_visits == 1 and root.visits == 1 and root.expand_visits == 1
    # a failed attempt deep in the tree lowers the subtree evidence of the
    # root but not its local (expand-here) evidence
    t.record_visits(["r", "a"])
    t.backup(a, False, None, q=1.0)
    assert root.beta == pytest.approx(1.0 + 0.9) and root.beta_loc == 1.0
    assert root.visits == 2 and root.expand_visits == 1 and a.expand_visits == 1
    t.backup_max(a, 3.0)
    assert root.best_sub == 3.0 and a.best_sub == 3.0
    assert LineageTree.from_dict(t.to_dict()).to_dict() == t.to_dict()


def test_puct_values_follow_spec_example():
    # PUCT spec 6.6: A is the island best with 4 failed direct attempts;
    # B is one step behind, fresh, born with mu0 = 0.67.
    t = LineageTree(0)
    a = t.add_root("A", 2.0)
    b = t.add_root("B", 1.0, mu0=2 / 3)
    t.theta = 1.0
    a.alpha_loc, a.beta_loc = 1.0, 5.0
    q_expand_a, _ = t._action_values(a, True, {})
    q_b, pi_b = t._action_values(b, False, {})
    assert q_expand_a == pytest.approx(1 / 6)
    assert q_b == pytest.approx((2 / 3) * math.exp(-1))
    assert pi_b == pytest.approx((2 / 3) * math.exp(-1))
    assert q_b > q_expand_a


def test_puct_descent_and_switches():
    t = LineageTree(0)
    root = t.add_root("r", 1.0)
    rng = random.Random(0)
    node, info = t.puct_select(rng, c_t=1.0)
    assert node.id == "r" and info["path"] == ["r"]  # leaf: only "expand"
    assert info["steps"][-1]["chosen"]["action"] == "expand"
    # a strong child pulls the descent below the root
    t.backup(root, True, 2.0, 1.0); t.backup_max(root, 2.0)
    t.add_child(root, "c", 2.0, rec(1, "r", score=2.0, pscore=1.0), 1)
    t.update_scalars(rec(1, "r", score=2.0, pscore=1.0))
    for _ in range(3):  # failures at r make expanding r unattractive
        t.record_visits(["r"]); t.backup(root, False, None, 1.0)
    node, info = t.puct_select(random.Random(1), c_t=1.0)
    assert node.id == "c" and info["path"] == ["r", "c"]
    for step in info["steps"]:
        assert step["chosen"]["score"] >= (step["best_alternative"] or {"score": -1})["score"]
    # c_t = 0: pure exploitation of Q is deterministic
    n1, _ = t.puct_select(random.Random(5), c_t=0.0)
    n2, _ = t.puct_select(random.Random(9), c_t=0.0)
    assert n1.id == n2.id
    # uniform-prior and UCT switches run
    t.puct_select(random.Random(2), c_t=1.0, switches={"puct_prior": False})
    t.puct_select(random.Random(2), c_t=1.0, switches={"uct": True, "max_backup": False})


def test_puct_database_flow_and_ucb_retrieval(tmp_path):
    from skydiscover.config import LineageEvolvePUCTDatabaseConfig

    cfg = LineageEvolvePUCTDatabaseConfig(num_islands=1, population_size=20)
    db = LineageEvolveDatabase("lineageevolve_puct", cfg)
    assert db.selection_policy == "puct" and db.retrieval_mode == "ucb"
    db.add(prog("init", 1.0, "x = 0"))
    db.total_iterations, db.current_iteration = 10, 5
    assert db.current_c() == pytest.approx(0.5)
    parent, _ = db.sample(2)
    sel = db.last_selection
    assert sel["policy"] == "puct" and sel["path"] == ["init"]
    db.commit_attempt(parent_id="init", island=0, iteration=1, status="admitted",
                      intent={"strategy": "new:a", "description": "d"}, prompt_sources={},
                      child_program=prog("c1", 2.0, "x = 1", "init"), selection_path=sel["path"])
    tree = db.trees[0]
    assert tree.root_visits == 1 and tree.nodes["init"].expand_visits == 1
    assert tree.nodes["init"].best_sub == 2.0 and tree.nodes["c1"].visits == 0
    db.memory.create("b")
    lessons, _ = db.retrieve_strategies(tree.nodes["c1"])
    assert lessons and lessons[0]["draw"] == pytest.approx(0.5 + 1.0 * math.sqrt(0.25 / 3))
    path = str(tmp_path / "ck")
    db.save(path, 1)
    db2 = LineageEvolveDatabase("lineageevolve_puct", cfg)
    db2.load(path)
    assert [t_.to_dict() for t_ in db2.trees] == [t_.to_dict() for t_ in db.trees]


def test_theta_window_tracks_recent_gains_and_stagnation_signal():
    t = LineageTree(0, theta_mode="window", theta_window=3)
    t.add_root("r", 0.0)
    t.update_scalars(rec(1, "r", score=0.9, pscore=0.0))  # huge first jump
    for i, s_ in enumerate([0.91, 0.912, 0.913], start=2):
        t.update_scalars(rec(i, "r", score=s_, pscore=s_ - (0.01 if i == 2 else 0.002 if i == 3 else 0.001)))
    assert t.theta == pytest.approx(0.002)  # median of the last 3 gains, jump forgotten
    assert t.attempts_since_best == 0
    t.update_scalars(rec(9, "r", score=0.5, pscore=0.4))  # progress, but not a new best
    assert t.attempts_since_progress == 0 and t.attempts_since_best == 1


def test_score_clone_is_recorded_but_not_a_node():
    db = make_db()
    db.add(prog("init", 1.0, "x = 0"))
    r = db.commit_attempt(parent_id="init", island=0, iteration=1, status="admitted",
                          intent={}, prompt_sources={}, child_program=prog("c", 1.0, "x = 5", "init"))
    tree = db.trees[0]
    assert "c" not in tree.nodes and "c" not in db.node_island
    assert r.y_parent == 0 and r.error.startswith("same score")
    assert "c" in db.programs and code_hash("x = 5") in db.hash_index
    assert tree.nodes["init"].attempts == [r]
    db2 = make_db(keep_score_clones=True)
    db2.add(prog("init", 1.0, "x = 0"))
    db2.commit_attempt(parent_id="init", island=0, iteration=1, status="admitted",
                       intent={}, prompt_sources={}, child_program=prog("c", 1.0, "x = 5", "init"))
    assert "c" in db2.trees[0].nodes


def test_tiny_island_best_gains_count_as_stagnation():
    t = LineageTree(0, stagnation_rel=1e-3)
    t.add_root("r", 0.79)
    for i in range(3):  # +0.0004 on 0.79 is ~5e-4 relative: a plateau
        t.update_scalars(rec(i, "r", score=0.79 + 0.0004 * (i + 1), pscore=0.79 + 0.0004 * i))
    assert t.attempts_since_best == 3 and t.attempts_since_progress == 0
    t.update_scalars(rec(9, "r", score=0.80, pscore=0.7912))  # +1.1% relative
    assert t.attempts_since_best == 0


def test_theta_floor_limits_individual_penalty():
    t = LineageTree(0, theta_min_rel=2e-3)
    t.add_root("r", 0.86)
    t.theta = 0.0002
    assert t.effective_theta() == pytest.approx(0.86 * 2e-3)
    assert t.reward_of(0.85) == pytest.approx(-0.01 / (0.86 * 2e-3))


def test_paradigm_mode_matches_adaevolve_and_links_memory():
    db = make_db()  # default meta_guidance_mode="paradigm"
    assert db.use_paradigm_breakthrough and db.paradigm_tracker is not None
    assert not db.use_meta_guidance and not db.should_trigger_meta(0)
    db.add(prog("init", 1.0, "x = 0"))
    db.add(prog("init2", 1.0, "x = 0"), target_island=1)
    db.commit_attempt(parent_id="init2", island=1, iteration=1, status="admitted", intent={},
                      prompt_sources={}, child_program=prog("best", 3.0, "x = 9", "init2"))
    assert db.best_program_id == "best"
    db.set_paradigms([{"idea": "Use LP for radii", "description": "solve an LP", "cautions": "slow"}])
    paradigm = db.get_current_paradigm()
    s = db.memory.get(paradigm["strategy_id"])
    assert s.name == "use_lp_for_radii" and s.origin == "meta_guide" and s.conditions == ["slow"]
    db.current_island = 0
    node = db.paradigm_parent_node()
    assert node.id == "best"
    sel = db.select_for_paradigm(node)
    assert db.current_island == 1 and sel["path"] == ["init2", "best"]
    db.use_paradigm(); db.use_paradigm()  # max_uses (2) per idea, as AdaEvolve
    assert db.get_current_paradigm() is None
    lineage_db = make_db(meta_guidance_mode="lineage")
    assert lineage_db.paradigm_tracker is None and lineage_db.use_meta_guidance


def test_exploration_floor_and_paradigm_parent_option():
    from skydiscover.config import LineageEvolvePUCTDatabaseConfig

    cfg = LineageEvolvePUCTDatabaseConfig(num_islands=1, population_size=20, c_min_frac=0.3,
                                          paradigm_parent="tree")
    db = LineageEvolveDatabase("lineageevolve_puct", cfg)
    db.total_iterations, db.current_iteration = 100, 95
    assert db.current_c() == pytest.approx(0.3)
    db.current_iteration = 10
    assert db.current_c() == pytest.approx(0.9)
    assert db.paradigm_parent == "tree" and db.use_paradigm_breakthrough
    with pytest.raises(ValueError):
        LineageEvolveDatabase("lineageevolve_puct",
                              LineageEvolvePUCTDatabaseConfig(num_islands=1, paradigm_parent="x"))


def test_context_only_imports():
    db = make_db(import_as_roots=False)
    db.add(prog("init", 1.0, "x = 0"))
    db.add(prog("init2", 1.0, "x = 0"), target_island=1)
    db.commit_attempt(parent_id="init", island=0, iteration=1, status="admitted", intent={},
                      prompt_sources={}, child_program=prog("c", 2.0, "x = 1", "init"))
    migrant = Program(id="m", solution="x = 1", metrics={"combined_score": 2.0}, parent_id="c",
                      metadata={"migrated_from": 0, "migrated_to": 1})
    db.add(migrant, target_island=1)
    assert "m" not in db.trees[1].nodes and "m" not in db.node_island
    assert any(p.id == "m" for p in db.get_island_population(1))  # still a context program
    assert db.trees[1].roots == ["init2"]
    db2 = make_db()  # default: spec behaviour
    db2.add(prog("init", 1.0, "x = 0"))
    db2.add(prog("init2", 1.0, "x = 0"), target_island=1)
    db2.add(Program(id="m", solution="x = 1", metrics={"combined_score": 2.0}, parent_id="init",
                    metadata={"migrated_from": 0}), target_island=1)
    assert "m" in db2.trees[1].nodes
