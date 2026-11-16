# LineageEvolve (PUCT variant) — Method Specification

This document specifies the LineageEvolve search method with an **MCTS tree policy based on PUCT**. It is written for an engineer (or coding agent) who will implement it **as a modification of AdaEvolve** in the SkyDiscover codebase (`skydiscover/search/adaevolve/`). It describes *what* the method computes and *when*; it does not prescribe code structure. Experiments and benchmark settings are out of scope.

Conventions used throughout:

- "MUST" marks behaviour the method depends on. "Default" marks a tunable value.
- All scores are the evaluator's `combined_score`, which is already higher-is-better.
- Where this document is silent, keep AdaEvolve's existing behaviour.

---

## 1. Overview

### 1.1 What stays from AdaEvolve

| AdaEvolve component | Status |
| --- | --- |
| Multiple islands, island selection by decayed UCB | Kept (two small changes, Section 10) |
| Migration between islands, dynamic island spawning | Kept (imports carry a prior, Section 10) |
| Sampling of context ("inspiration") programs shown next to the parent | Kept |
| Evaluator interface, LLM ensemble, diff / full-rewrite output modes | Kept |
| Meta-guidance (paradigm) generation on stagnation | Kept, with a new trigger and a new way of entering the prompt (Section 9.5) |
| **Within-island parent selection** (exploration intensity $I_t$, explore/exploit sampling) | **Replaced** by MCTS with a PUCT tree policy (Sections 3–7) |
| Flat per-island population | **Replaced** by a per-island lineage tree (Section 2) |
| Prompt history block ("previous attempts", top programs summary) | **Extended** by lineage context and strategy memory (Section 9) |

### 1.2 What is new

1. **Lineage tree with an explicit "expand here" action.** Every island stores its programs as a tree whose edges are evaluated LLM edits. At every node the tree policy chooses between descending into one of its children and generating a new child of the node itself.
2. **Individual Reward $R$.** How far a program (or the best program of a subtree) is behind the island's best, in units of a typical improvement step.
3. **Lineage Reward $\bar p$.** The probability that an expansion produces progress, estimated from the outcomes of edits in the lineage, weighted by genealogical distance and by source attribution.
4. **PUCT tree policy.** Action value $Q$ = expected improvement estimated from backed-up evidence; prior $P$ = the same expected improvement estimated without any visit evidence; visit counts $N$ = budget actually spent.
5. **Source attribution $q$.** For every evaluated edit, the fraction of the changed code that originates from the lineage itself rather than from external sources in the prompt.
6. **Lineage-aware Mutation.** The mutation prompt contains the path that produced the parent, the log of everything already tried at the parent, and lessons retrieved from a shared Strategy Memory whose utilities are updated through attribution.
7. **Pre-evaluation deduplication.**

### 1.3 The principle everything is derived from

The search objective is the best score found within the budget. The value of expanding node $w$ is therefore the expected improvement of its child $C'$ over the island's current best $J^*_k$:

$$V(w) = \mathbb{E}\big[(J(C') - J^*_k)_+ \mid \text{expand } w\big].$$

Write $J(C') = J(w) + \delta$. Model the edit gain: with probability $p_w$ the edit makes progress ($\delta > \varepsilon$), and then $\delta \sim \mathrm{Exp}(\text{mean } \theta_k)$; otherwise it contributes nothing to the objective. With gap $G = J^*_k - J(w) \ge 0$:

$$V(w) = p_w \int_G^\infty (x-G)\,\tfrac{1}{\theta_k}e^{-x/\theta_k}\,dx = p_w\,\theta_k\,\exp\!\Big(-\frac{J^*_k - J(w)}{\theta_k}\Big).$$

Dividing by the island constant $\theta_k$ gives a dimensionless value in $[0,1]$:

$$\frac{V(w)}{\theta_k} = \underbrace{p_w}_{\text{Lineage Reward}} \cdot \underbrace{\exp\big(R(w)\big)}_{\text{Individual Reward}}, \qquad R(w) = \frac{J(w) - J^*_k}{\theta_k} \le 0.$$

Interpretation: $\exp(R(w)) = P(\delta > G \mid \text{progress})$ is the probability that a progress step from $w$ is large enough to beat the island's best; $p_w$ is the probability that a progress step happens at all.

This quantity is used twice in the tree policy (Section 6):

- estimated **with** backed-up evidence → action value $Q$;
- estimated **without** visit evidence → prior $P$.

It is used in product form (not log form) because PUCT adds $Q$ to an exploration term whose scale is fixed by $P \in [0,1]$; $Q$ must therefore be bounded, and $V/\theta_k \in [0,1]$.

---

## 2. Search state

### 2.1 Lineage tree (one per island $k$)

- **Virtual island root** $r_k$: not a program, never expanded, holds only a visit count. Its children are the lineage roots.
- **Lineage roots**: programs with no local parent — the initial program(s) and every program imported or spawned into the island.
- **Node** = one admitted program $w$ with score $J(w)$.
- **Edge** $u \to v$ = an admitted LLM edit of $u$ that produced $v$.

Per-node fields:

| Field | Meaning |
| --- | --- |
| program id, code, score $J(w)$, metrics | as in AdaEvolve |
| `parent_node`, `depth`, creation iteration, `incoming_edit` | tree structure |
| $\mu^0_w$ | prior mean, set once at creation (Section 4.4) |
| $(\alpha_w, \beta_w)$ | **subtree** lineage evidence (Section 4.3) |
| $(\alpha^{\oplus}_w, \beta^{\oplus}_w)$ | **local** evidence: attempts made directly at $w$ (Section 4.5) |
| $\tilde J_w$ | best score in the subtree rooted at $w$ (max-backup) |
| $N(w)$ | visits: simulations whose selection path passed through $w$ |
| $N(\oplus_w)$ | number of expansion attempts made at $w$ |
| provenance | for imports: source island and source program id |

There is **no depth limit and no progressive-widening limit**: the "expand here" action (Section 6.1) decides breadth.

### 2.2 Attempt records

Every expansion attempt at a node $u$ produces one **attempt record** $e$, stored on $u$, whether or not it yields a node:

| Field | Meaning |
| --- | --- |
| `parent` | $u$ |
| `status` | `admitted`, `crashed`, `duplicate` (infrastructure failures produce no record) |
| `child` | node $v$ if admitted, else none |
| `intent` | declared before evaluation: strategy id, sources used, one-sentence description (Section 8.1) |
| `delta` | $\delta_e = J(v) - J(u)$ if admitted |
| `y_parent` | $y_e(u)$ (Section 4.1) |
| `q` | lineage share $q_e$ (Section 5) |
| `shares` | strategy shares $c_{e,s}$ (Section 5.4) |
| `prompt_sources` | ids of the lessons, context programs and meta-guidance idea that were in the prompt |
| `selection_path` | node ids from $r_k$ to $u$ |
| `iteration` | iteration index |

Status definitions:

- **Infrastructure failure** (LLM/API error, evaluator-system failure, no candidate parsed): no record, no update of any kind, no visit counted. The iteration is consumed as in AdaEvolve.
- **`duplicate`**: the candidate's normalized hash already exists in the run-wide hash index (Section 11). Not evaluated.
- **`crashed`**: evaluated and failed because of itself (syntax/runtime error, timeout, invalid output, non-finite score). No node is created.
- **`admitted`**: finite score obtained; a child node is created regardless of whether the score improved.

### 2.3 Island-level scalars

For each island $k$: best score $J^*_k$; step scale $\theta_k$ (undefined until the first progress event); counter of attempts since the last progress event (Section 9.5); AdaEvolve's existing scheduler statistics.

### 2.4 Global state

- Strategy Memory $\mathcal{M}$ (Section 9.3), shared by all islands.
- Hash index of all evaluated normalized programs (Section 11).
- Evaluator noise level $\varepsilon \ge 0$ (Section 3.2).

All state above MUST be included in checkpoints and restored on resume.

---

## 3. Individual Reward

### 3.1 Definition

For a node $w$ and for the subtree rooted at $w$:

$$R(w) = \frac{J(w) - J^*_k}{\theta_k}, \qquad \tilde R(w) = \frac{\tilde J_w - J^*_k}{\theta_k},$$

with $R \equiv \tilde R \equiv 0$ while $\theta_k$ is undefined. Both are $\le 0$. They are invariant to affine rescaling of the score, so **no score transformation or min–max normalization is applied** anywhere in the tree search.

$\tilde J_w$ is maintained by **max-backup**: when an admitted child $v$ is created, $\tilde J_v = J(v)$ and $\tilde J_a \leftarrow \max(\tilde J_a, J(v))$ for every ancestor $a$ of $v$. The max operator is used because the objective is a maximum; the progress probability (Section 4) uses a mean because it estimates a rate.

### 3.2 Step scale $\theta_k$ and noise level $\varepsilon$

A **progress event** is an admitted edit with $\delta_e > \varepsilon$ (parent-relative).

$\theta_k$ is the exponential moving average of the gains of progress events in island $k$:

- on the first progress event: $\theta_k \leftarrow \delta_e$;
- afterwards: $\theta_k \leftarrow \rho\,\theta_k + (1-\rho)\,\delta_e$, default $\rho = 0.9$.

$\theta_k$ is not updated by non-progress attempts. A newly spawned island inherits $\theta$ from the island that seeded it (mean if several).

$\varepsilon$ is the evaluator noise level. Default: $0$ plus a numerical tolerance ($10^{-12}\cdot\max(1,|J|)$) for deterministic evaluators. For noisy evaluators (e.g. scores including wall-clock time), set $\varepsilon$ to twice the standard deviation of the score over a few repeated evaluations of the initial program. $\varepsilon$ is a per-task configuration value.

### 3.3 Why a step scale instead of min–max normalization

MCTS-AHD normalizes the value by the range $q_{\max} - q_{\min}$ of all values seen. That range is fixed by the worst program ever encountered and does not shrink as the search converges, so late in the run all good nodes are compressed near 1 and the value term stops discriminating between them. $\theta_k$ tracks the size of the improvements currently being made: as gains shrink, the same absolute gap becomes a larger penalty, and selection concentrates on near-best nodes without any schedule.

---

## 4. Lineage Reward

The Lineage Reward of a node is a posterior mean of a progress probability. Two versions are kept per node: **subtree** (used when the node is chosen as a child to descend into) and **local** (used when the node itself is expanded).

### 4.1 Observation: outcome of an attempt, relative to an ancestor

Let $e$ be an attempt at node $u$, and let $a$ be $u$ or any ancestor of $u$ within the same lineage. Define the path maximum

$$m_a(u) = \max_{b \in \mathrm{path}(a \to u)} J(b),$$

including both $a$ and $u$. The outcome of $e$ with respect to $a$ is

$$y_e(a) = \begin{cases}\mathbb{1}\big[J(v) > m_a(u) + \varepsilon\big] & \text{if } e \text{ is admitted with child } v,\\ 0 & \text{if } e \text{ is crashed or duplicate.}\end{cases}$$

- For $a = u$ this is the progress event $\delta_e > \varepsilon$.
- For a more distant ancestor, the edit only counts as success if the child beats **every** program between $a$ and $u$; repairing an earlier regression is not progress for the ancestors above it.
- $y$ is binary. Gain magnitude is pooled at island level in $\theta_k$, not estimated per branch.

### 4.2 Evidence weight

$$w_e(a) = \gamma^{\,d(a,u)} \cdot q_e,$$

- $d(a,u)$ = number of edges from $a$ down to $u$ ($d(u,u)=0$);
- $\gamma \in (0,1]$, default $0.9$: genealogical decay — the tree analogue of AdaEvolve's temporal decay $\rho$ in $G_t = \rho G_{t-1} + (1-\rho)\delta_t^2$: evidence is down-weighted by its distance in the lineage instead of its age;
- $q_e \in [0,1]$: lineage share of the edit (Section 5); $q_e = 1$ for crashed and duplicate attempts and whenever no valid interpretation exists.

### 4.3 Subtree Lineage Reward

With $E(a)$ the set of attempts made at $a$ or at any descendant of $a$:

$$\alpha_a = \kappa\,\mu^0_a + \sum_{e \in E(a)} w_e(a)\,y_e(a), \qquad \beta_a = \kappa\,(1-\mu^0_a) + \sum_{e \in E(a)} w_e(a)\,\big(1-y_e(a)\big),$$

$$\bar p_a = \frac{\alpha_a}{\alpha_a + \beta_a}.$$

$\kappa$ = prior strength, default $2$ (the strength of the uniform prior $\mathrm{Beta}(1,1)$).

### 4.4 Prior mean $\mu^0$

$\mu^0_w$ is set **once**, when $w$ is created, and never changed:

| Node type | $\mu^0$ |
| --- | --- |
| Initial program of the run | $1/2$ |
| Child $v$ created by admitted edit $u \to v$ | $\bar p_u$ (subtree) evaluated **after** that edit has been backed up |
| Program imported by migration or seeding a spawned island | $\bar p$ of the source node at the time of import |

A child born from a successful edit therefore starts with a higher prior than one born from a failed edit. This is the method's momentum assumption, bounded to $\kappa$ pseudo-observations; the node's own evidence overrides it.

### 4.5 Local Lineage Reward (for expanding a node itself)

$$\alpha^{\oplus}_w = \kappa\,\mu^0_w + \sum_{e \text{ at } w} q_e\,y_e(w), \qquad \beta^{\oplus}_w = \kappa\,(1-\mu^0_w) + \sum_{e \text{ at } w} q_e\,\big(1-y_e(w)\big), \qquad \bar p^{\oplus}_w = \frac{\alpha^{\oplus}_w}{\alpha^{\oplus}_w + \beta^{\oplus}_w}.$$

Only attempts made directly at $w$ count. Rationale: the value of editing $w$ again should fall when edits of $w$ keep failing, but should not fall because edits deep inside one of $w$'s children failed; those failures already lower the value of descending into that child.

### 4.6 Visits versus evidence

$N$ counts budget spent; $(\alpha,\beta)$ count evidence learned. They differ on purpose: an edit whose changed code was mostly borrowed from a context program spends one visit at $u$ but contributes little evidence about $u$'s lineage ($q_e$ small). Visit counts are never weighted by $q$ or $\gamma$.

---

## 5. Source attribution

### 5.1 Purpose

$q_e$ answers: *to what degree is the outcome of this edit evidence about the lineage?* The mutation prompt contains the parent, retrieved lessons, context programs and possibly a meta-guidance idea. If the changed code was transplanted from a context program or implements a retrieved lesson, its success or failure is evidence about that source, not about the lineage. Attribution splits each edit among its sources; the lineage's share weights the Lineage Reward evidence, the external shares update the Strategy Memory.

### 5.2 Inputs

1. **Declared intent** (Section 8.1), written by the mutation model before evaluation.
2. **Changed-unit inventory** $U_e$: a deterministic list of changed units between parent and child. A unit is a contiguous diff hunk, merged with other hunks inside the same enclosing function/class when this can be determined; each unit has a stable id, its enclosing symbol, old/new excerpts and a size $|c|$ = added plus removed lines. Computed identically for diff-mode and full-rewrite outputs.
3. **Prompt sources**: ids and contents of the lessons, context programs and meta-guidance idea that were in the prompt.

### 5.3 Edit Interpreter and validation

One LLM call per **admitted** edit with a non-empty inventory (low temperature). Input: inventory, declared intent, prompt sources, parent/child scores, outcome label. Output: for each unit $c \in U_e$ exactly one source label $s(c)$:

- `lineage` — continues, refines or tunes a mechanism already present in the parent;
- `novel` — an idea not traceable to the parent's mechanisms or to any prompt source;
- `ext:<source id>` — implements a specific retrieved lesson or meta-guidance idea, or borrows code from a specific context program.

It also returns the lesson text fields of Section 9.3.

Validation (a label that fails is replaced by `lineage`):

| Label | Valid iff |
| --- | --- |
| `ext:<lesson or idea id>` | the id was in the prompt **and** listed in the declared intent's sources |
| `ext:<context program id>` | the id was in the prompt **and** at least half of the unit's added lines (whitespace-normalized) occur in that program and not in the parent |
| `lineage`, `novel` | always |

If the interpreter abstains, returns invalid output, or cites unknown unit ids: $q_e = 1$, whole edit credited to its declared strategy.

### 5.4 Shares

$$q_e = \frac{\sum_{c:\, s(c) \in \{\texttt{lineage},\texttt{novel}\}} |c|}{\sum_{c \in U_e} |c|}, \qquad a_{e,j} = \frac{\sum_{c:\, s(c) = \texttt{ext:}j} |c|}{\sum_{c \in U_e} |c|}, \qquad q_e + \sum_j a_{e,j} = 1.$$

- `novel` counts toward the lineage: it is what expanding $u$ produced without outside help.
- Strategy shares $c_{e,s}$: the declared strategy receives $q_e$; the strategy behind each external lesson/idea $j$ receives $a_{e,j}$; shares attributed to context programs are not credited to any strategy.
- Crashed and duplicate attempts: no interpreter call; $q_e = 1$; the declared strategy receives share $1$.

$q_e$ is the share of the **edit**, not of the **gain**. Decomposing the gain would require ablation re-evaluations and is not part of the method.

---

## 6. Tree policy (PUCT)

Executed after AdaEvolve's scheduler has chosen island $k$.

### 6.1 Actions

- At the virtual root: $A(r_k) = $ lineage roots of island $k$.
- At a program node $w$: $A(w) = \mathrm{children}(w) \cup \{\oplus_w\}$, where $\oplus_w$ means "stop here and generate a new child of $w$".

A leaf has $A(w) = \{\oplus_w\}$, so descent always terminates. Choosing between $\oplus_w$ and a child replaces progressive widening: breadth at $w$ grows exactly when generating at $w$ looks better than continuing in any existing child. (This "generation node" device is the GEN node of AB-MCTS; see Section 15.)

For each action define its **target node** $\omega(a)$: $\omega(c) = c$ for a child, $\omega(\oplus_w) = w$.

### 6.2 Action value $Q$ (with evidence)

$$Q(c) = \bar p_c \cdot \exp\big(\tilde R(c)\big), \qquad Q(\oplus_w) = \bar p^{\oplus}_w \cdot \exp\big(R(w)\big).$$

- $Q(\oplus_w)$ is exactly $V(w)/\theta_k$ of Section 1.3 with $p_w$ estimated from the attempts made at $w$.
- $Q(c)$ is the same expression for "searching inside subtree $c$": the clade's progress rate (mean-backup, discounted, attribution-weighted) times the probability that progress launched from the clade's best program beats the island's best (max-backup).
- An unvisited node has $\bar p = \mu^0$ and $\tilde J = J$, so its $Q$ equals its prior value (Section 6.3). No first-play-urgency rule or "unvisited first" rule is needed.
- $Q \in [0,1]$.

### 6.3 Prior $P$ (without visit evidence)

The prior value of an action uses only quantities fixed when its target node was created ($\mu^0$, $J$) and the current island scale:

$$\pi_0(a) = \mu^0_{\omega(a)} \cdot \exp\big(R(\omega(a))\big).$$

It is the expected-improvement estimate the search would use if it had no visits at all: the lineage estimate inherited at birth times the individual reward. The prior policy at node $u$ is

$$P(a \mid u) = (1-\eta)\,\frac{\pi_0(a)}{\sum_{a' \in A(u)} \pi_0(a')} + \frac{\eta}{|A(u)|}, \qquad \eta = 0.25 \text{ (default)}.$$

$J^*_k$ cancels in the normalization; only score differences in units of $\theta_k$ matter.

The uniform floor $\eta$ plays the role of AlphaZero's root noise (mixing fraction 0.25): without it a child that regressed by several $\theta$ would receive an exploration bonus of order $e^{-\text{several}}$ and could never be revisited. Program search benefits from occasionally developing temporarily worse programs (the motivation of MCTS-AHD), so a floor is kept.

### 6.4 Selection score and descent

$$\mathrm{score}(a \mid u) = Q(a) + c_t \cdot P(a \mid u) \cdot \frac{\sqrt{\max(N(u),\,1)}}{1 + N(a)},$$

where $N(a) = N(c)$ for a child action and $N(\oplus_w)$ for the expand action, and

$$c_t = c_0 \cdot \Big(1 - \frac{t}{H}\Big), \qquad c_0 = 1.0 \text{ (default)},$$

with $t$ the current iteration and $H$ the iteration budget (linear exploration decay, as in MCTS-AHD's exploration-decay technique).

Descent:

```
u ← r_k
loop:
    a* ← argmax_{a ∈ A(u)} score(a | u)        # ties: uniform random, seeded
    if a* = ⊕_u: return u                       # u is the parent to expand
    u ← a*
```

The returned node is the **parent**. Context programs are then sampled exactly as AdaEvolve does today (parent excluded). The random number generator MUST be seeded from the run seed.

### 6.5 What each PUCT ingredient does here

| Ingredient | Meaning in LineageEvolve | Source of information |
| --- | --- | --- |
| $Q$ | expected improvement of the action, estimated with evidence | backed-up outcomes (mean, $\gamma$, $q$) and subtree best (max) |
| $P$ | the same expected improvement, estimated without visits | prior inherited at birth, own score |
| $N$ | how much budget has gone through the action | raw visit counts |
| $c_t$ | exploration strength | constant with linear decay |

Compared with UCT (used by MCTS-AHD), PUCT scales each action's exploration bonus by its prior. UCT gives the same bonus to every rarely visited child, so children that regressed badly are revisited as eagerly as promising ones; with PUCT the bonus of a regressed child is close to the floor $\eta/|A(u)|$.

### 6.6 Worked example

Island with $\theta_k$ defined. Node A is the island best ($R = 0$, $\mu^0 = 1/2$) and has had 4 failed direct attempts: $\bar p^{\oplus}_A = 1/(1+5) = 0.17$, so $Q(\oplus_A) = 0.17$. Node B is one step behind ($R = -1$), was just created by a successful edit from a seed with posterior mean $0.67$ and has no visits: $Q(B) = 0.67 \cdot e^{-1} = 0.25$. Even before the exploration term, descending into B and expanding there is preferred over editing A a fifth time.

---

## 7. Backup

After each attempt record $e$ at node $u$ with selection path $r_k \to \dots \to u$:

1. **Visits.** $N(x) \leftarrow N(x) + 1$ for every node $x$ on the selection path, including $r_k$ and $u$; $N(\oplus_u) \leftarrow N(\oplus_u) + 1$.
2. **Subtree evidence.** Walk from $u$ up to its lineage root:
   ```
   m ← −∞
   for a in [u, parent(u), …, lineage root of u]:
       m ← max(m, J(a))
       y ← 1 if (e admitted and J(v) > m + ε) else 0
       w ← γ^(depth(u) − depth(a)) · q_e
       α_a ← α_a + w · y
       β_a ← β_a + w · (1 − y)
   ```
3. **Local evidence.** $\alpha^{\oplus}_u \leftarrow \alpha^{\oplus}_u + q_e\,y_e(u)$, $\beta^{\oplus}_u \leftarrow \beta^{\oplus}_u + q_e\,(1-y_e(u))$.
4. **Max-backup.** If admitted: $\tilde J_a \leftarrow \max(\tilde J_a, J(v))$ for $u$ and all its ancestors.
5. **New child.** If admitted, create $v$ with $\mu^0_v = \bar p_u$ (after step 2), $(\alpha_v,\beta_v) = (\alpha^{\oplus}_v,\beta^{\oplus}_v) = (\kappa\mu^0_v,\ \kappa(1-\mu^0_v))$, $\tilde J_v = J(v)$, $N(v) = N(\oplus_v) = 0$.

Backup never crosses islands; an imported node is a lineage root and backups stop there. The new child receives no direct evidence from its own creating edit; that edit is evidence about its parent and reaches the child only through $\mu^0_v$.

If iterations run concurrently, apply a virtual loss while a simulation is in flight (increment $N$ along the path at selection time and treat the pending outcome as $y = 0$ in $Q$ until the attempt completes), and apply all updates of one attempt atomically.

---

## 8. Expansion

### 8.1 Declared intent

The mutation model MUST output, before the code, a short machine-readable header:

| Field | Content |
| --- | --- |
| `strategy` | id of one strategy from the list shown in the prompt, or `new: <short_snake_case_name>` |
| `sources` | ids of the lessons / meta-guidance idea / context programs the edit draws on (may be empty) |
| `description` | one sentence describing the change |

The intent is produced before the score is known and is never rewritten. If the header is missing or unparsable: `strategy = unspecified`, `sources = []`, empty description; the candidate is still used.

### 8.2 Prompt

AdaEvolve's mutation prompt with its history section replaced by the blocks of Section 9.6. One candidate per iteration.

---

## 9. Lineage-aware Mutation

The prompt context is organised by distance from the selected parent $u$: what produced $u$ (path), what was already tried at $u$ (attempts), what other lineages have learned (memory).

### 9.1 Path trace

The incoming edits on the path from the lineage root to $u$, oldest first, one line each: declared description, strategy name, score before → after, whether it was a progress event. If too long, keep every progress edit plus the last $h$ edits (default $h = 5$) and replace omitted runs by a one-line count. For an imported root, the first line states its origin and score. No LLM call.

### 9.2 Attempt log

All attempt records at $u$, most recent first: declared description, strategy name, status, and result (`crashed: <error class>`, `duplicate`, `no change`, `regressed to …`, `improved to …`). Crashed and duplicate attempts MUST appear. If too long, keep the most recent entries plus aggregated counts. No LLM call.

### 9.3 Strategy Memory

Run-wide, shared by all islands. Each item is a **strategy** $s$:

| Field | Content |
| --- | --- |
| `id`, `name` | stable id; short snake_case name |
| `summary` | what the strategy does (1–2 sentences) |
| `mechanism` | why it is expected to help |
| `conditions` | when it applies / does not (list, appended over time) |
| `evidence` | entries $(e,\ u_e,\ c_{e,s},\ y_e)$: attempt id, program it was applied to, strategy's share, parent-relative outcome $y_e = y_e(u_e)$ |
| `origin` | `edit` or `meta_guide` |

Rules:

- `new: <name>` creates an item; text fields are written by the Edit Interpreter if admitted, else from the declared description.
- Declaring an existing strategy appends evidence; the interpreter may append at most one condition. Existing text is never rewritten.
- For every attempt $e$ and strategy $s$ with $c_{e,s} > 0$, append $(e, u_e, c_{e,s}, y_e)$.
- `strategy = unspecified` contributes no evidence for its own share.

### 9.4 Strategy utility and retrieval

Utility of strategy $s$ for parent $u$:

$$a_s(u) = 1 + \sum_{(e,u_e,c,y)} k(u,u_e)\,c\,y, \qquad b_s(u) = 1 + \sum_{(e,u_e,c,y)} k(u,u_e)\,c\,(1-y), \qquad \bar u_s(u) = \frac{a_s(u)}{a_s(u)+b_s(u)}.$$

$k(u,u') \in [0,1]$: cosine similarity of code embeddings clipped at 0, or token-set Jaccard similarity of normalized sources if no embedding model is configured. This is the same estimator as the Lineage Reward with a different kernel: genealogical distance for a question about one branch, program similarity for knowledge that must transfer across branches.

Retrieval for parent $u$, consistent with the tree policy (an upper-confidence rule instead of sampling):

1. **Filter.** Remove strategies that (a) are the declared strategy of a progress edit on $u$'s path, or (b) appear in $u$'s attempt log.
2. **Lessons to try.** Rank remaining strategies by
   $$\bar u_s(u) + c_0\,\sqrt{\frac{\bar u_s(u)\,(1-\bar u_s(u))}{a_s(u)+b_s(u)+1}}$$
   (posterior mean plus one exploration-weighted posterior standard deviation) and take the top $K_+$ (default 2).
3. **Warnings.** Among remaining strategies not chosen in step 2 with $b_s(u) - 1 > 0$, take the $K_-$ (default 1) with the largest similarity-weighted failure mass $b_s(u) - 1$; render as "failed on similar programs" with their conditions.

Shorter blocks if the memory has fewer eligible items. Retrieved ids are recorded in `prompt_sources`.

### 9.5 Meta-guidance on stagnation

- **Trigger:** island $k$ has had no progress event in its last $W$ attempts (default $W = 10$) and no unused meta-guidance idea is pending for the island. Replaces AdaEvolve's threshold on the accumulated improvement signal.
- **Input:** the selected parent $u$ with its path trace and attempt log, the task description, names of existing strategies (not the global-best program).
- **Output:** a small set of strategy-level ideas, as in AdaEvolve.
- **Use:** each idea becomes a Strategy Memory item with `origin = meta_guide` and no evidence (utility $\mathrm{Beta}(1,1)$). In the triggering iteration one idea takes one of the $K_+$ slots; afterwards ideas compete through ordinary retrieval. They are presented as lessons to try, not mandatory instructions.

### 9.6 Prompt blocks, in order

1. Task/system message (unchanged).
2. Selected parent: code, score, evaluator metrics and feedback (unchanged).
3. **Lineage: how this program was reached** — path trace.
4. **Already tried on this program** — attempt log, with an instruction not to repeat failed approaches.
5. **Lessons from other lineages** — $K_+$ strategies (name, summary, mechanism, conditions, id).
6. **Known dead ends on similar programs** — $K_-$ warnings.
7. Context programs (unchanged).
8. **Strategy list** — ids and names of existing strategies for the intent declaration (capped to the most-evidenced items if large).
9. Output instructions: intent header, then program or diff (unchanged formats).

Search statistics ($Q$, $P$, $N$, $\alpha$, $\beta$, $R$) are **never** shown to the mutation model.

---

## 10. Islands (changes to AdaEvolve's outer loop)

1. **Imports carry a prior.** An imported or seeding program becomes a lineage root in the destination island with $\mu^0$ = $\bar p$ of its source node, $(\alpha,\beta) = (\alpha^\oplus,\beta^\oplus) = (\kappa\mu^0, \kappa(1-\mu^0))$, $\tilde J = J$, $N = N(\oplus) = 0$, empty attempt log, provenance text for its path trace. Source-island evidence and visits are not copied.
2. **Crashed and duplicate attempts are visible to the scheduler**: one visit with zero reward. Infrastructure failures are not.
3. Everything else (reward definition, decay, UCB constant, migration interval, spawn rule) unchanged. $J^*_k$ is the island's own best, including imported programs.

---

## 11. Deduplication

Before evaluation, hash the normalized candidate (inside evolve-block markers if present; blank lines removed, lines trimmed, internal whitespace collapsed). If the hash is in the run-wide index:

- do not call the evaluator;
- record `status = duplicate`, $y = 0$, $q = 1$;
- run backup (Section 7) and memory update (Section 9.3) as for any failed attempt;
- the iteration is consumed.

Every evaluated candidate's hash (admitted or crashed) is added to the index. There is no other recurrence or diversity term in the tree policy.

---

## 12. One iteration, in order

1. **Island.** AdaEvolve's scheduler selects island $k$.
2. **Select.** Descend from $r_k$ by PUCT until an expand action is chosen (Section 6.4); the node reached is the parent $u$. Sample context programs as in AdaEvolve.
3. **Stagnation check.** If the trigger of Section 9.5 holds, generate meta-guidance ideas for $u$ and insert them into the Strategy Memory.
4. **Build prompt** (Section 9.6).
5. **Expand.** One LLM call returns intent header and candidate $C'$. Infrastructure failure: stop, no updates.
6. **Dedup.** If duplicate: record, go to step 9 with $y = 0$, $q = 1$.
7. **Evaluate.** Infrastructure failure: stop, no updates. Candidate failure: record `crashed`, go to step 9 with $y = 0$, $q = 1$. Otherwise record `admitted` with $\delta_e$.
8. **Interpret.** Changed-unit inventory → Edit Interpreter → validation → $q_e$, $a_{e,j}$, $c_{e,s}$ (Section 5).
9. **Backup.** Steps 1–5 of Section 7 (visits, subtree evidence, local evidence, max-backup, new child).
10. **Island scalars.** If $\delta_e > \varepsilon$: update $\theta_k$, reset the stagnation counter; else increment it. Update $J^*_k$ if $J(v) > J^*_k$.
11. **Memory.** Create/update strategy items and append evidence (Section 9.3).
12. **Outer loop.** Update the island scheduler; run migration/spawning on AdaEvolve's schedule (Section 10).

Ordering constraints that MUST hold:

- Subtree evidence (Section 7 step 2) before the new child's prior (step 5).
- Selection in the next iteration uses $\theta_k$, $J^*_k$ and all statistics as committed at the end of the previous iteration.
- Concurrent iterations: virtual loss and atomic updates (Section 7).

---

## 13. Hyperparameters

| Symbol | Meaning | Default |
| --- | --- | --- |
| $c_0$ | PUCT exploration strength (decays linearly to 0 over the budget) | 1.0 |
| $\eta$ | uniform floor in the prior policy | 0.25 |
| $\gamma$ | genealogical decay of evidence | 0.9 |
| $\kappa$ | strength of the inherited prior | 2 |
| $\rho$ | EMA factor of the step scale $\theta_k$ | 0.9 |
| $\varepsilon$ | evaluator noise level (per task) | 0 (+ numerical tolerance) |
| $K_+$, $K_-$ | lessons to try, warnings | 2, 1 |
| $h$ | recent edits kept in the path trace | 5 |
| $W$ | stagnation window (attempts) | 10 |

No depth limit, no progressive-widening parameters, no score transformation, no weighted sum of selection terms, no retrieval ranking weights.

### Component switches

Each MUST be independently switchable for later analysis:

| Switch | Effect |
| --- | --- |
| Lineage Reward off | $\bar p \equiv \bar p^{\oplus} \equiv 1/2$: $Q$ and $P$ use only the Individual Reward |
| Individual Reward off | $R \equiv \tilde R \equiv 0$: $Q$ and $P$ use only the Lineage Reward |
| Prior off | $P(a \mid u) = 1/|A(u)|$ (uniform PUCT) |
| UCT instead of PUCT | exploration term $c_t\sqrt{\ln(N(u)+1)/(1+N(a))}$ |
| Max-backup off | $\tilde J_c$ replaced by $J(c)$ in $Q(c)$ |
| Local expand value off | $Q(\oplus_w)$ uses subtree $\bar p_w$ instead of $\bar p^{\oplus}_w$ |
| Backup to ancestors off | evidence updates only the expanded node ($\gamma = 0$) |
| Inherited prior off | $\mu^0 = 1/2$ for all nodes |
| Source attribution off | $q_e = 1$; no interpreter call for attribution |
| Lineage context off | path trace and attempt log omitted from the prompt |
| Strategy Memory off | lessons, warnings, strategy list omitted; intent `strategy` ignored |
| Deduplication off | duplicates evaluated like any candidate |

---

## 14. Quantities to log per iteration

Island; full selection path; at every step of the descent, for the chosen action and the best alternative: $Q$, $P$, $N$, and the resulting score; parent id with $R$, $\tilde R$, $(\alpha,\beta)$, $(\alpha^\oplus,\beta^\oplus)$, $\mu^0$; $\theta_k$, $J^*_k$, $c_t$; attempt status, $\delta_e$, $y_e(u)$, $q_e$, strategy shares; retrieved strategy ids with $(a_s, b_s)$; whether meta-guidance triggered. The logs must suffice to reconstruct the full tree and all statistics at any iteration.

---

## 15. Provenance of design elements (for positioning)

| Element | Prior work it builds on | What LineageEvolve changes |
| --- | --- | --- |
| PUCT score $Q + c\,P\sqrt{N}/(1+N)$ | AlphaGo / AlphaZero | $Q$ and $P$ are the same expected-improvement quantity, with and without evidence |
| Expand-here action at every node | GEN node of AB-MCTS (Inoue et al., NeurIPS 2025) | Chosen by PUCT with lineage evidence instead of Thompson sampling on score models |
| Clade aggregation with depth decay | HGM (clade-metaproductivity), Clade-AHD ($\lambda^{\text{dist}}$ aggregation into Beta beliefs) | Observation is a progress event relative to the path maximum, not a normalized score; evidence weighted by attribution; prior inherited from parent |
| Exploration decay over the budget | MCTS-AHD | — |
| Improvement signal with exponential decay | AdaEvolve ($G_t$) | Decay over genealogical distance instead of time |
| Item memory with success/failure counts | ACE, ReasoningBank | Counts updated by attribution shares and weighted by program similarity |

---

## 16. Summary of symbols

| Symbol | Meaning |
| --- | --- |
| $J(w)$, $\tilde J_w$ | score of program $w$; best score in its subtree |
| $J^*_k$, $\theta_k$ | best score of island $k$; EMA of progress gains |
| $R(w)$, $\tilde R(w)$ | Individual Reward of node $w$ / of its subtree |
| $\bar p_w$, $(\alpha_w,\beta_w)$ | subtree Lineage Reward and its Beta parameters |
| $\bar p^{\oplus}_w$, $(\alpha^\oplus_w,\beta^\oplus_w)$ | local Lineage Reward (attempts at $w$ only) |
| $\mu^0_w$, $\kappa$ | prior mean set at creation; prior strength |
| $y_e(a)$, $m_a(u)$ | outcome of attempt $e$ relative to ancestor $a$; path maximum |
| $d(a,u)$, $\gamma$ | genealogical distance; decay |
| $q_e$, $a_{e,j}$, $c_{e,s}$ | lineage share; external-source shares; strategy shares |
| $A(w)$, $\oplus_w$, $\omega(a)$ | action set; expand action; target node of an action |
| $Q(a)$, $\pi_0(a)$, $P(a\mid u)$ | action value; prior value; prior policy |
| $N(w)$, $N(\oplus_w)$ | visits through $w$; attempts at $w$ |
| $c_t$, $c_0$, $\eta$ | exploration strength (decayed); its initial value; prior floor |
| $k(u,u')$, $a_s(u)$, $b_s(u)$ | program similarity; strategy utility parameters |
