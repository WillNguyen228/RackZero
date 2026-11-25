# Architectural starting point and research references

## Upstream review

Reference: [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general),
revision [`f1a78e0505c614205f8c60eaa82cd644a07a84b3`](https://github.com/suragnair/alpha-zero-general/tree/f1a78e0505c614205f8c60eaa82cd644a07a84b3),
MIT, copyright 2025 Surag Nair. RackZero is an independent implementation
inspired by the interfaces and training loop, not a source-code transplant.

Reading covered the six core modules in full, utility/entry-point/test files,
all game families' rules/player adapters, neural wrappers (including comparisons
of repetitive wrappers against the Othello implementation), Docker setup,
RTS configuration/visualization and all code cells of the three DotsAndBoxes
notebooks. Notebook outputs and binary assets (pretrained weights, images and
PDF writeup) were not read. No claim is made to have audited those binaries.
The games span Othello, Connect4, Gobang, TicTacToe/3D, Tafl, DotsAndBoxes,
Santorini and RTS; none supplies a ready-made Scrabble hidden-information model.

| Upstream component | Reusable idea | RackZero adaptation |
|---|---|---|
| `Game.py` | Small rules/transition/terminal interface | Separate live `Game`, observable `InformationState`, hypothetical `World` and lexicon-only `Simulator` |
| `MCTS.py` | PUCT priors, value backup, visit policy | Sampled stochastic transitions, hidden-rack sampling, distinct information-set and determinization node keys |
| `NeuralNet.py` | Train/predict/checkpoint abstraction | PyTorch variable-candidate policy plus win/margin heads |
| `Coach.py` | Self-play -> replay -> optimize -> arena -> promote | Information-only replay, natural outcomes, optimizer rollback, full iteration-boundary recovery |
| `Arena.py` | Agent callables and swapped starting seats | Same game seeds in paired seats, explicit truncation accounting and baseline metrics |
| `main.py` | Central experiment wiring | Validated YAML configurations and training/evaluation CLIs |

The upstream MCTS keeps dictionaries of Q, edge visits, state visits, priors,
terminal values and legal masks keyed by a full board serialization. It
negates values between player-relative canonical states. Its transition
interface has no chance outcome, and its policy uses a fixed action-space
mask. `Coach` collects symmetry-augmented examples, samples visit policies
with a move-count temperature schedule, trains a candidate and gates against
the preceding model. The Othello PyTorch wrapper uses policy cross-entropy
and value MSE. These are useful organizational ideas.

Direct reuse would be wrong: a Scrabble board is not a full state, hidden
racks cannot be serialized into agent tree keys, draws are stochastic, action
sets are variable and swapping player signs does not reconstruct the other
player's information. RackZero also avoids treating a training turn cutoff as
a terminal win and preserves optimizer/RNG/replay state for recovery.

Canonicalization is limited to relative scores and value perspective. No
unverified board/action symmetry transformations are used. Upstream code
contains extra-turn game adapters, but those do not address hidden tiles or
stochastic draws.

## Established methods versus project choices

Established ideas: PUCT/neural self-play [1], information-set rather than
omniscient-state search [2], simulator-based particle-belief planning [3],
and the practical usefulness and theoretical limitations of determinization
[4,5]. Tile conservation and unordered draw probabilities follow directly
from sampling a finite multiset without replacement.

Project choices, **not claims of reproducing the papers**:

- A two-player adapter around the existing Java rules engine.
- Shared MLP candidate scoring and separate terminal sign/margin targets.
- Top-score candidate pruning with reserved pass/full-exchange candidates.
- Information-set planning against an observation-only opponent policy.
- A finite-particle filter with explicit warning/reinitialization on depletion.
- Root-visit voting across a small number of determinizations.
- Sampled chance nodes rather than exhaustive chance expansion.
- Simple paired-seat model promotion and an independent greedy diagnostic.

PIMC can optimize a different continuation in each hypothetical hidden world.
That is not necessarily a single strategy executable by a real player:
this is the strategy-fusion problem. Information-set sharing and an
observation-only opponent policy reduce this source of bias in the
information-set modes here, but do not establish equilibrium play or eliminate
all approximation/non-locality errors. Retaining determinization baselines is
intentional: sophistication alone is not evidence of stronger play.

## Primary sources

1. Silver et al. (2017), **Mastering Chess and Shogi by Self-Play with a General
   Reinforcement Learning Algorithm**.
   [arXiv:1712.01815](https://arxiv.org/abs/1712.01815).
   Neural self-play in perfect-information games; not direct evidence for
   applying unchanged AlphaZero to Scrabble.
2. Cowling, Powley and Whitehouse (2012), **Information Set Monte Carlo Tree
   Search**, IEEE TCIAIG 4(2), 120-143.
   [DOI](https://doi.org/10.1109/TCIAIG.2012.2200894),
   [author repository](https://eprints.whiterose.ac.uk/id/eprint/75048/).
   Information-set trees and hidden-information search variants.
3. Silver and Veness (2010), **Monte-Carlo Planning in Large POMDPs**,
   NeurIPS 23.
   [Proceedings](https://papers.nips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html).
   Black-box simulation and particle-belief planning; a POMDP planner is not
   automatically a two-player equilibrium solver.
4. Frank and Basin (1998), **Search in games with incomplete information:
   a case study using Bridge card play**, Artificial Intelligence 100.
   [DOI](https://doi.org/10.1016/S0004-3702(97)00082-9).
   Problems arising when incomplete-information games are represented by
   ordinary perfect-information game trees.
5. Long, Sturtevant, Buro and Furtak (2010), **Understanding the Success of
   Perfect Information Monte Carlo Sampling in Game Tree Search**, AAAI.
   [Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/7562).
   Empirical conditions under which theoretically limited PIMC works well.

The reference investigation verified bibliographic/abstract sources; it was
not a full-text replication study of these papers. No competitive-strength
results are claimed. See [RACKZERO.md](RACKZERO.md) for implemented semantics,
limitations, setup and reproducible commands.
