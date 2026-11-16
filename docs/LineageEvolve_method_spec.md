# LineageEvolve — Method Specification

This document specifies the LineageEvolve search method. It is written for an engineer (or coding agent) who will implement it **as a modification of AdaEvolve** in the SkyDiscover codebase (`skydiscover/search/adaevolve/`). It describes *what* the method computes and *when*; it does not prescribe code structure. Experiments and benchmark settings are out of scope.

Conventions used throughout:

- "MUST" marks behaviour the method depends on. "Default" marks a tunable value.
- All scores are the evaluator's `combined_score`, which is already higher-is-better.
- Where this document is silent, keep AdaEvolve's existing behaviour.

---

## 1. Overview

### 1.1 What stays from AdaEvolve

| AdaEvolve component | Status |
| --- | --- |
| Multiple islands, island selection by decayed UCB | Kept (two small changes, Section 9) |
| Migration between islands, dynamic island spawning | Kept (imports carry a prior, Section 9) |
| Sampling of context ("inspiration") programs shown next to the parent | Kept |
| Evaluator interface, LLM ensemble, diff / full-rewrite output modes | Kept |
| Meta-guidance (paradigm) generation on stagnation | Kept, with a new trigger and a new way of entering the prompt (Section 8.5) |
| **Within-island parent selection** (exploration intensity $I_t$, explore/exploit sampling) | **Replaced** by lineage-tree selection (Sections 3–6) |
| Flat per-island population | **Replaced** by a per-island lineage tree (Section 2) |
| Prompt history block ("previous attempts", top programs summary) | **Extended** by lineage context and strategy memory (Section 8) |

### 1.2 What is new

1. **Lineage tree.** Every island stores its programs as a tree whose edges are evaluated LLM edits.
2. **Individual Reward $R(u)$.** How far program $u$ is behind the island's best, measured in units of a typical improvement step.
3. **Lineage Reward $L(u)$.** The (log) probability that expanding $u$ produces progress, estimated from the outcomes of edits in $u$'s lineage, weighted by genealogical distance and by source attribution.
4. **Selection.** Thompson sampling on $R(u) + L(u)$, which is the log expected improvement of expanding $u$.
5. **Source attribution $q$.** For every evaluated edit, the fraction of the changed code that originates from the lineage itself rather than from external sources in the prompt (retrieved lessons, context programs, meta-guidance).
6. **Lineage-aware Mutation.** The mutation prompt contains the path that produced the parent, the log of everything already tried at the parent, and lessons retrieved from a shared Strategy Memory whose utilities are updated through attribution.
7. **Pre-evaluation deduplication.**

### 1.3 The principle everything is derived from

The search objective is the best score found within the budget. The value of expanding node $u$ is therefore the expected improvement of its child $C'$ over the island's current best $J^*_k$:

$$V(u) = \mathbb{E}\big[(J(C') - J^*_k)_+ \mid \text{expand } u\big].$$

Write $J(C') = J(u) + \delta$. Model the edit gain $\delta$ as follows: with probability $p_u$ the edit makes progress ($\delta > \varepsilon$), and in that case $\delta \sim \mathrm{Exp}(\text{mean } \theta_k)$; otherwise it contributes nothing to the objective. With gap $G = J^*_k - J(u) \ge 0$:

$$V(u) = p_u \int_G^\infty (x-G)\,\tfrac{1}{\theta_k}e^{-x/\theta_k}\,dx = p_u\,\theta_k\,\exp\!\Big(-\frac{J^*_k - J(u)}{\theta_k}\Big).$$

Taking logs and dropping the constant $\log\theta_k$ shared by all nodes of the island:

$$\log V(u) = \underbrace{\frac{J(u) - J^*_k}{\theta_k}}_{R(u)\ \text{(Individual Reward)}} + \underbrace{\log p_u}_{L(u)\ \text{(Lineage Reward)}}.$$

The two rewards are two factors of one expectation, so they are added with **no weighting coefficient**. $R$ is computed from the node's own score. $p_u$ is unknown and is estimated from the lineage.

---

## 2. Search state

### 2.1 Lineage tree (one per island $k$)

- **Node** = one admitted program $v$ with score $J(v)$. Fields: program id, code, score, metrics, `parent_node` (or none), `depth`, creation iteration, `incoming_edit` (or none), Beta parameters $(\alpha_v, \beta_v)$, frozen prior mean $\mu^0_v$, and provenance (for imports: source island and source program id).
- **Root nodes** = programs with no local parent: the initial program(s) and every program imported or spawned into the island. An island may have several roots. Roots have no incoming edit.
- **Edge** $u \to v$ = an admitted LLM edit of $u$ that produced $v$.
- **Expandable set** $\mathcal{N}_k$ = all nodes of island $k$ (no depth cap, no widening limit).

### 2.2 Attempt records

Every expansion attempt at a node $u$ produces one **attempt record** $e$, stored on $u$, whether or not it yields a node:

| Field | Meaning |
| --- | --- |
| `parent` | $u$ |
| `status` | `admitted`, `crashed`, `duplicate` (infrastructure failures produce no record) |
| `child` | node $v$ if admitted, else none |
| `intent` | declared before evaluation: strategy id, sources used, one-sentence description (Section 7.1) |
| `delta` | $\delta_e = J(v) - J(u)$ if admitted |
| `y_parent` | $y_e(u)$ (Section 4.1) |
| `q` | lineage share $q_e$ (Section 5) |
| `shares` | strategy shares $c_{e,s}$ (Section 5.4) |
| `prompt_sources` | ids of the lessons, context programs and meta-guidance idea that were in the prompt |
| `iteration` | iteration index |

Definitions of the three statuses:

- **Infrastructure failure** (LLM/API error, evaluator-system failure, no candidate parsed): no record, no update of any kind. The iteration is consumed as in AdaEvolve.
- **`duplicate`**: the candidate's normalized hash already exists in the run-wide hash index (Section 10). Not evaluated.
- **`crashed`**: the candidate was evaluated and failed because of itself (syntax/runtime error, timeout, invalid output, non-finite score). No node is created.
- **`admitted`**: finite score obtained; a child node is created regardless of whether the score improved.

### 2.3 Island-level scalars

For each island $k$: best score $J^*_k$; step scale $\theta_k$ (undefined until the first progress event); counter of attempts since the last progress event (for Section 8.5); and AdaEvolve's existing scheduler statistics.

### 2.4 Global state

- Strategy Memory $\mathcal{M}$ (Section 8.3), shared by all islands.
- Hash index of all evaluated or admitted normalized programs (Section 10).
- Evaluator noise level $\varepsilon \ge 0$ (Section 3.2).

All state above MUST be included in checkpoints and restored on resume.

---

## 3. Individual Reward

### 3.1 Definition

$$R(u) = \begin{cases}\dfrac{J(u) - J^*_k}{\theta_k} & \text{if } \theta_k \text{ is defined},\\[2mm] 0 & \text{otherwise.}\end{cases}$$

$R(u) \le 0$, and $R(u) = 0$ for the island's best program. $R$ is invariant to affine rescaling of the score, so **no score transformation or normalization is applied** anywhere in the tree search.

### 3.2 Step scale $\theta_k$ and noise level $\varepsilon$

A **progress event** is an admitted edit with $\delta_e > \varepsilon$ (parent-relative).

$\theta_k$ is the exponential moving average of the gains of progress events in island $k$:

- on the first progress event: $\theta_k \leftarrow \delta_e$;
- afterwards: $\theta_k \leftarrow \rho\,\theta_k + (1-\rho)\,\delta_e$, default $\rho = 0.9$.

$\theta_k$ is not updated by non-progress attempts. A newly spawned island inherits $\theta$ from the island that seeded it; if seeded from several, use their mean.

$\varepsilon$ is the evaluator noise level. Default: $\varepsilon = 0$ plus a numerical tolerance ($10^{-12}\cdot\max(1,|J|)$) for deterministic evaluators. For noisy evaluators (e.g. scores that include wall-clock time), set $\varepsilon$ to twice the standard deviation of the score over a few repeated evaluations of the initial program. $\varepsilon$ is a per-task configuration value.

### 3.3 Behaviour to expect

- Before any progress exists in the island, $R \equiv 0$ and selection is driven only by $L$.
- $\theta_k$ shrinks as the search converges, so the same absolute gap costs more: selection automatically concentrates on near-best nodes late in the run. This replaces any fixed exploration schedule.

---

## 4. Lineage Reward

$L(u) = \log p_u$, where $p_u$ has a Beta posterior maintained on every node.

### 4.1 Observation: outcome of an attempt, relative to an ancestor

Let $e$ be an attempt at node $u$, and let $a$ be $u$ or any ancestor of $u$ in the island tree. Define the path maximum

$$m_a(u) = \max_{b \in \mathrm{path}(a \to u)} J(b),$$

where the path includes both $a$ and $u$. The outcome of $e$ with respect to $a$ is

$$y_e(a) = \begin{cases}\mathbb{1}\big[J(v) > m_a(u) + \varepsilon\big] & \text{if } e \text{ is admitted with child } v,\\ 0 & \text{if } e \text{ is crashed or duplicate.}\end{cases}$$

Notes:

- For $a = u$ this is the progress event $\delta_e > \varepsilon$.
- For a more distant ancestor, the edit only counts as success if the child beats **every** program between $a$ and $u$. An edit that merely repairs an earlier regression is a success for the regressed node but not for the ancestors above the regression.
- $y$ is binary by design. Gain magnitude is pooled at island level in $\theta_k$; it is not estimated per branch.
- $m_a(u)$ is computed incrementally while walking from $u$ up to the root.

### 4.2 Evidence weight

The weight of attempt $e$ (made at $u$) as evidence about ancestor $a$ is

$$w_e(a) = \gamma^{\,d(a,u)} \cdot q_e,$$

- $d(a,u)$ = number of edges from $a$ down to $u$ ($d(u,u) = 0$);
- $\gamma \in (0,1]$, default $0.9$: genealogical decay. This is the tree analogue of AdaEvolve's temporal decay $\rho$ in $G_t = \rho\,G_{t-1} + (1-\rho)\,\delta_t^2$: AdaEvolve down-weights evidence by its age; here evidence is down-weighted by its distance in the lineage;
- $q_e \in [0,1]$: lineage share of the edit (Section 5). $q_e = 1$ for crashed and duplicate attempts and whenever no valid interpretation exists.

### 4.3 Posterior

For every node $a$, with $E(a)$ the set of attempts made at $a$ or at any descendant of $a$:

$$\alpha_a = \kappa\,\mu^0_a + \sum_{e \in E(a)} w_e(a)\,y_e(a), \qquad \beta_a = \kappa\,(1-\mu^0_a) + \sum_{e \in E(a)} w_e(a)\,\big(1 - y_e(a)\big),$$

$$p_a \sim \mathrm{Beta}(\alpha_a, \beta_a), \qquad \bar p_a = \frac{\alpha_a}{\alpha_a + \beta_a}.$$

$\kappa$ is the prior strength, default $\kappa = 2$ (the strength of the uniform prior $\mathrm{Beta}(1,1)$).

### 4.4 Prior mean $\mu^0$

$\mu^0_a$ is set **once**, when node $a$ is created, and never changed afterwards:

| Node type | $\mu^0$ |
| --- | --- |
| Initial program of the run | $1/2$ |
| Child $v$ created by admitted edit $u \to v$ | $\bar p_u$ evaluated **after** the edit $u \to v$ has been backed up into $u$ |
| Program imported by migration, or seeded into a spawned island | $\bar p$ of the source node at the time of import |

So a child born from a successful edit starts with a higher prior than a child born from a failed one, because the edit has just moved its parent's posterior. This is the method's momentum assumption; its strength is bounded by $\kappa$ pseudo-observations, and the node's own evidence overrides it.

The child node receives **no** direct evidence from its own incoming edit; that edit is evidence about its parent (distance 0) and reaches the child only through $\mu^0$.

### 4.5 Backup procedure (after each attempt record $e$ at node $u$)

```
m ← −∞
for a in [u, parent(u), parent(parent(u)), …, root of u's lineage]:
    m ← max(m, J(a))
    y ← 1 if (e admitted and J(v) > m + ε) else 0
    w ← γ^(depth(u) − depth(a)) · q_e
    α_a ← α_a + w · y
    β_a ← β_a + w · (1 − y)
```

Backup never crosses islands. An imported node is a root: its backups stop at itself.

### 4.6 Direction of information flow

- **Upward** by backup: each attempt updates all ancestors, attenuated by $\gamma^d$.
- **Downward** by priors: each new node inherits its parent's current estimate.
- Siblings influence each other only through their common ancestors' posteriors and hence through the priors of later-born nodes.

---

## 5. Source attribution

### 5.1 Purpose

$q_e$ answers: *to what degree is the outcome of this edit evidence about the lineage?* The mutation prompt contains the parent, retrieved lessons, context programs and possibly a meta-guidance idea. If the changed code was transplanted from a context program or implements a retrieved lesson, its success or failure is evidence about that source, not about the lineage's own capacity to progress. Attribution splits each edit among its sources; the lineage's share weights the Lineage Reward evidence, and the external shares update the Strategy Memory.

### 5.2 Inputs

1. **Declared intent** (Section 7.1), written by the mutation model before evaluation.
2. **Changed-unit inventory** $U_e$: a deterministic list of changed units between parent and child. A unit is a contiguous diff hunk, merged with other hunks inside the same enclosing function/class when the language allows this to be determined; each unit has a stable id, its enclosing symbol, old/new excerpts, and a size $|c|$ = number of added plus removed lines. The inventory is computed the same way for diff-mode and full-rewrite outputs.
3. **Prompt sources**: the ids and contents of lessons, context programs and meta-guidance idea that were in the prompt.

### 5.3 Edit Interpreter and validation

One LLM call per **admitted** edit with a non-empty inventory (low temperature). It receives the inventory, the declared intent, the prompt sources, parent/child scores and the outcome label. It returns, for each unit $c \in U_e$, exactly one source label $s(c)$:

- `lineage` — continues, refines or tunes a mechanism already present in the parent;
- `novel` — introduces an idea not traceable to the parent's existing mechanisms or to any prompt source;
- `ext:<source id>` — implements a specific retrieved lesson or meta-guidance idea, or borrows code from a specific context program.

It also returns the lesson text fields of Section 8.3.

Labels are validated deterministically. A label that fails validation is replaced by `lineage`:

| Label | Valid iff |
| --- | --- |
| `ext:<lesson or idea id>` | the id was in the prompt **and** was listed in the declared intent's sources |
| `ext:<context program id>` | the id was in the prompt **and** at least half of the unit's added lines (whitespace-normalized) occur in that context program and do not occur in the parent |
| `lineage`, `novel` | always |

If the interpreter abstains, returns invalid output, or cites unknown unit ids, set $q_e = 1$ and treat the whole edit as belonging to its declared strategy.

### 5.4 Shares

$$q_e = \frac{\sum_{c:\, s(c) \in \{\texttt{lineage},\texttt{novel}\}} |c|}{\sum_{c \in U_e} |c|}, \qquad a_{e,j} = \frac{\sum_{c:\, s(c) = \texttt{ext:}j} |c|}{\sum_{c \in U_e} |c|}, \qquad q_e + \sum_j a_{e,j} = 1.$$

- `novel` counts toward the lineage: it is what expanding $u$ produced without outside help.
- Strategy shares $c_{e,s}$, used by the Strategy Memory: the edit's own declared strategy receives $q_e$; the strategy behind each external lesson/idea $j$ receives $a_{e,j}$; shares attributed to context programs are not credited to any strategy.
- For crashed and duplicate attempts no interpreter call is made: $q_e = 1$, and the declared strategy receives share $1$.

$q_e$ is the share of the **edit**, not of the **gain**. Decomposing the gain itself would require ablation re-evaluations and is not part of the method.

---

## 6. Selection

Executed after AdaEvolve's scheduler has chosen island $k$.

For every node $u \in \mathcal{N}_k$:

$$\tilde p_u \sim \mathrm{Beta}(\tau\,\alpha_u,\ \tau\,\beta_u), \qquad I(u) = R(u) + \log \tilde p_u, \qquad u^\star = \arg\max_{u \in \mathcal{N}_k} I(u).$$

- $\tau$ is a sharpening factor. Default $\tau = 1$. Optional budget schedule: $\tau = H / (H - t)$ with $H$ the total iteration budget and $t$ the current iteration (more exploitation as the budget runs out).
- Clamp $\tilde p_u$ to $[10^{-12}, 1]$ before the logarithm. Ties are broken uniformly at random.
- The random number generator MUST be seeded from the run seed so that runs are reproducible.

Properties (no additional rule is needed for any of them):

- **Exploration** comes from posterior width: nodes with little evidence occasionally draw a high $\tilde p$.
- **Widen vs. deepen** is decided by comparing $I(u)$ with $I(\text{children})$. Each failed attempt at $u$ raises $\beta_u$, so $u$ gradually yields to its descendants or to other branches.
- **Backtracking**: when the best node accumulates failures, a slightly worse node with a healthier posterior overtakes it. Example with posterior means: best node, $R = 0$, $\mathrm{Beta}(1,5)$, $\log\bar p = -1.79$; a node one step behind, $R=-1$, $\mathrm{Beta}(4/3, 2/3)$, $\log \bar p = -0.41$; indices $-1.79$ vs. $-1.41$, so the second is preferred.

The selected $u^\star$ is the **parent**. Context programs are then sampled exactly as AdaEvolve does today (the parent itself excluded).

---

## 7. Expansion

### 7.1 Declared intent

The mutation model MUST output, before the code, a short machine-readable header:

| Field | Content |
| --- | --- |
| `strategy` | the id of one strategy from the list shown in the prompt, or `new: <short_snake_case_name>` |
| `sources` | list of ids of the lessons / meta-guidance idea / context programs the edit draws on (may be empty) |
| `description` | one sentence describing the change |

The intent is produced before the score is known and is never rewritten afterwards. If the header is missing or unparsable, the candidate is still used, with `strategy = unspecified`, `sources = []` and an empty description.

### 7.2 Prompt

The prompt is AdaEvolve's mutation prompt with its history section replaced by the blocks of Section 8.6. One candidate is generated per iteration.

---

## 8. Lineage-aware Mutation

The prompt context is organised by distance from the selected parent $u$: what produced $u$ (path), what was already tried at $u$ (attempts), and what other lineages have learned (memory).

### 8.1 Path trace

The sequence of incoming edits on the path from the lineage root to $u$, oldest first, one line per edit: declared description, strategy name, score before → after, and whether it was a progress event. If the path is longer than the budget, keep every progress edit plus the last $h$ edits (default $h = 5$) and replace omitted runs by a one-line count. For an imported root, the first line states where it came from and its score.

No LLM call is used to build the path trace.

### 8.2 Attempt log

All attempt records stored on $u$, most recent first: declared description, strategy name, status, and resulting score or failure type (`crashed: <error class>`, `duplicate`, `no change`, `regressed to …`, `improved to …`). Crashed and duplicate attempts MUST appear here; this is how the generator learns what not to repeat. If the log exceeds its budget, keep the most recent entries and append aggregated counts for the rest.

No LLM call is used to build the attempt log.

### 8.3 Strategy Memory

A run-wide store shared by all islands. Each item is a **strategy** $s$:

| Field | Content |
| --- | --- |
| `id`, `name` | stable id; short snake_case name |
| `summary` | one or two sentences: what the strategy does |
| `mechanism` | why it is expected to help |
| `conditions` | when it applies / when it does not (list, appended over time) |
| `evidence` | list of entries $(e,\ u_e,\ c_{e,s},\ y_e)$: attempt id, the program it was applied to, the strategy's share, the parent-relative outcome $y_e = y_e(u_e)$ |
| `origin` | `edit` or `meta_guide` |

Creation and update rules:

- When an attempt declares `new: <name>`, a new item is created. Its text fields are written by the Edit Interpreter if the attempt is admitted; otherwise they are filled from the declared description.
- When an attempt declares an existing strategy, no new item is created; an evidence entry is appended. The interpreter may append at most one new condition to `conditions`. Existing text is never rewritten.
- For every attempt $e$ and every strategy $s$ with share $c_{e,s} > 0$ (Section 5.4), append the evidence entry $(e, u_e, c_{e,s}, y_e)$.
- Attempts with `strategy = unspecified` contribute no strategy evidence for their own share.

### 8.4 Strategy utility and retrieval

**Utility of strategy $s$ for the selected parent $u$** is a Beta posterior in which each piece of evidence is weighted by the similarity between $u$ and the program where the evidence was observed:

$$a_s(u) = 1 + \sum_{(e,u_e,c,y) \in \mathrm{evidence}(s)} k(u,u_e)\,c\,y, \qquad b_s(u) = 1 + \sum_{(e,u_e,c,y) \in \mathrm{evidence}(s)} k(u,u_e)\,c\,(1-y).$$

$k(u,u') \in [0,1]$ is a program-similarity kernel: cosine similarity of code embeddings clipped at 0; if no embedding model is configured, token-set Jaccard similarity of the normalized sources. This is the same estimator as the Lineage Reward with a different kernel: genealogical distance for a question about one branch, program similarity for knowledge that must transfer across branches.

**Retrieval for parent $u$:**

1. **Filter.** Remove strategies that (a) are the declared strategy of a progress edit on $u$'s path (already embodied in $u$), or (b) appear in $u$'s attempt log (already shown there).
2. **Lessons to try.** For each remaining strategy draw $\tilde u_s \sim \mathrm{Beta}(a_s(u), b_s(u))$ and take the $K_+$ strategies with the largest draws (default $K_+ = 2$).
3. **Warnings.** Among remaining strategies not chosen in step 2 and with $b_s(u) - 1 > 0$, take the $K_-$ with the largest similarity-weighted failure mass $b_s(u) - 1$ (default $K_- = 1$). They are rendered as "failed on similar programs", with their conditions.

If the memory has fewer eligible items than the quotas, the blocks are shorter or omitted. The ids of all retrieved strategies are recorded in the attempt's `prompt_sources`.

### 8.5 Meta-guidance on stagnation

- **Trigger:** island $k$ has had no progress event in its last $W$ attempt records (default $W = 10$), and no unused meta-guidance idea is pending for the island. This replaces AdaEvolve's threshold on the accumulated improvement signal.
- **Input:** the selected parent $u$ with its path trace and attempt log, the task description, and the names of strategies already in memory. (Not the global-best program.)
- **Output:** a small set of strategy-level ideas, as in AdaEvolve.
- **Use:** each idea is inserted into the Strategy Memory as a new item with `origin = meta_guide` and no evidence, i.e. utility $\mathrm{Beta}(1,1)$. For the iteration in which they were generated, one of these ideas takes one of the $K_+$ slots; afterwards they compete through ordinary retrieval. The prompt presents them as lessons to try, not as mandatory instructions.

### 8.6 Prompt blocks, in order

1. Task/system message (unchanged).
2. Selected parent: code, score, evaluator metrics and feedback (unchanged).
3. **Lineage: how this program was reached** — path trace (8.1).
4. **Already tried on this program** — attempt log (8.2), with an instruction not to repeat failed approaches.
5. **Lessons from other lineages** — $K_+$ strategies (name, summary, mechanism, conditions, id).
6. **Known dead ends on similar programs** — $K_-$ warnings.
7. Context programs (unchanged from AdaEvolve).
8. **Strategy list** — ids and names of existing strategies, for the intent declaration (cap the list at the most-evidenced items if it grows large).
9. Output instructions: intent header (7.1), then the program or diff (unchanged formats).

Numeric search statistics ($\alpha$, $\beta$, $\bar p$, $R$) are **never** shown to the mutation model.

---

## 9. Islands (changes to AdaEvolve's outer loop)

1. **Imports carry a prior.** A program imported by migration or used to seed a spawned island becomes a root node in the destination tree with $\mu^0$ = $\bar p$ of its source node, $(\alpha,\beta) = (\kappa\mu^0, \kappa(1-\mu^0))$, an empty attempt log, and provenance text for its path trace. Evidence from the source island is not copied.
2. **Crashed and duplicate attempts are visible to the scheduler.** They count as a visit to the island with zero reward. Infrastructure failures do not.
3. Everything else (reward definition, decay, UCB constant, migration interval, spawn rule) is unchanged. $J^*_k$ used by $R$ is the island's own best, including imported programs.

---

## 10. Deduplication

Before evaluation, compute the hash of the normalized candidate (inside evolve-block markers if present; blank lines removed, lines trimmed, internal whitespace collapsed). If the hash is in the run-wide index:

- do not call the evaluator;
- create an attempt record with `status = duplicate`, $y = 0$, $q = 1$;
- run backup (Section 4.5) and memory update (Section 8.3) as for any failed attempt;
- the iteration is consumed.

The hash of every evaluated candidate (admitted or crashed) is added to the index after evaluation. There is no other recurrence or diversity term anywhere in selection.

---

## 11. One iteration, in order

1. **Island.** AdaEvolve's scheduler selects island $k$.
2. **Select.** Sample $\tilde p_u$ for all $u \in \mathcal{N}_k$; parent $u^\star = \arg\max R(u) + \log\tilde p_u$ (Section 6). Sample context programs as in AdaEvolve.
3. **Stagnation check.** If the trigger of Section 8.5 holds, generate meta-guidance ideas for $u^\star$ and insert them into the Strategy Memory.
4. **Build prompt.** Path trace, attempt log, retrieved lessons and warnings, context programs, strategy list (Section 8.6).
5. **Expand.** One LLM call returns intent header and candidate $C'$. On infrastructure failure: stop, no updates.
6. **Dedup.** If duplicate: record, go to step 9 with $y=0$, $q=1$.
7. **Evaluate.** Infrastructure failure: stop, no updates. Candidate failure: record `crashed`, go to step 9 with $y = 0$, $q = 1$. Otherwise create child node $v$ and record `admitted` with $\delta_e$.
8. **Interpret.** Build the changed-unit inventory; call the Edit Interpreter; validate labels; compute $q_e$, $a_{e,j}$, $c_{e,s}$ (Section 5).
9. **Backup.** Update $(\alpha_a, \beta_a)$ for $u^\star$ and all its ancestors (Section 4.5).
10. **Child prior.** If admitted: $\mu^0_v \leftarrow \bar p_{u^\star}$ (post-backup), $(\alpha_v,\beta_v) \leftarrow (\kappa\mu^0_v,\ \kappa(1-\mu^0_v))$.
11. **Island scalars.** If $\delta_e > \varepsilon$: update $\theta_k$ and reset the stagnation counter; else increment it. Update $J^*_k$ if $J(v) > J^*_k$.
12. **Memory.** Create/update strategy items and append evidence (Section 8.3).
13. **Outer loop.** Update the island scheduler; run migration/spawning on AdaEvolve's schedule (Section 9).

Ordering constraints that MUST hold:

- Step 9 before step 10 (the child's prior includes the outcome of its own creating edit through its parent).
- $y_e(a)$ in step 9 uses $J^*$-independent quantities only (scores on the path), so the order of steps 9 and 11 does not matter for correctness; $\theta_k$ used by selection is always the value from the end of the previous iteration.
- If iterations run concurrently, all updates of one attempt (steps 9–12) are applied atomically; selection reads the latest committed state.

---

## 12. Hyperparameters

| Symbol | Meaning | Default |
| --- | --- | --- |
| $\gamma$ | genealogical decay of evidence | 0.9 |
| $\kappa$ | strength of the inherited prior | 2 |
| $\rho$ | EMA factor of the step scale $\theta_k$ | 0.9 |
| $\varepsilon$ | evaluator noise level (per task) | 0 (+ numerical tolerance) |
| $\tau$ | posterior sharpening | 1 (optional schedule $H/(H-t)$) |
| $K_+$, $K_-$ | lessons to try, warnings | 2, 1 |
| $h$ | recent edits kept in the path trace | 5 |
| $W$ | stagnation window (attempts) | 10 |

There are no selection weights, no exploration constant, no progressive-widening parameters, no score transformation, and no retrieval ranking weights.

### Component switches

Each of the following MUST be independently switchable, to allow later analysis:

| Switch off | Effect |
| --- | --- |
| Lineage Reward | $\tilde p_u$ drawn from $\mathrm{Beta}(1,1)$ for all nodes (selection by $R$ plus uniform noise) |
| Backup to ancestors | evidence updates only the expanded node ($\gamma = 0$) |
| Inherited prior | $\mu^0 = 1/2$ for all nodes |
| Source attribution | $q_e = 1$ for all attempts; no interpreter call for attribution |
| Lineage context | path trace and attempt log omitted from the prompt |
| Strategy Memory | lessons, warnings and strategy list omitted; intent `strategy` field ignored |
| Deduplication | duplicates are evaluated like any candidate |

---

## 13. Quantities to log per iteration

For later verification of the method's assumptions, record for every iteration: island, selected node id, $R(u^\star)$, $(\alpha_{u^\star}, \beta_{u^\star})$, $\bar p_{u^\star}$, the sampled $\tilde p_{u^\star}$, $\theta_k$, $J^*_k$, attempt status, $\delta_e$, $y_e(u^\star)$, $q_e$, strategy shares, retrieved strategy ids with their $(a_s, b_s)$, and whether meta-guidance was triggered. These logs must be sufficient to reconstruct the full tree with all posteriors at any iteration.

---

## 14. Summary of symbols

| Symbol | Meaning |
| --- | --- |
| $J(v)$ | evaluator score of program $v$ (higher is better) |
| $J^*_k$ | best score in island $k$ |
| $\delta_e$ | gain of edit $e = (u \to v)$: $J(v) - J(u)$ |
| $\theta_k$ | EMA of gains of progress events in island $k$ |
| $R(u)$ | Individual Reward |
| $p_u$, $(\alpha_u,\beta_u)$, $\bar p_u$ | progress probability of expanding $u$; its Beta parameters; posterior mean |
| $L(u)$ | Lineage Reward, $\log p_u$ |
| $y_e(a)$ | outcome of attempt $e$ relative to ancestor $a$ |
| $m_a(u)$ | maximum score on the path from $a$ to $u$ |
| $d(a,u)$ | number of edges from $a$ to $u$ |
| $q_e$, $a_{e,j}$ | lineage share and external-source shares of edit $e$ |
| $c_{e,s}$ | share of edit $e$ credited to strategy $s$ |
| $k(u,u')$ | program similarity kernel |
| $a_s(u), b_s(u)$ | utility posterior of strategy $s$ for parent $u$ |
| $I(u)$ | selection index $R(u) + \log\tilde p_u$ |
