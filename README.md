# Scrabble

[![CI](https://github.com/MasterChiefProject/Scrabble/actions/workflows/ci.yml/badge.svg)](https://github.com/MasterChiefProject/Scrabble/actions/workflows/ci.yml)

**Scrabble** is a Java 17 word-board game engine with full board validation and scoring, exact dictionary lookup, a bounded TCP dictionary service, automated tests, and a browser-playable multiplayer adaptation.

**Playable demo:** [RackZero browser demo](https://willnguyen228.github.io/RackZero/)

The browser demo is deployed from `docs/` to GitHub Pages on pushes to `main`.

The Java implementation is the primary rules engine. A JavaScript browser adaptation mirrors the main placement, scoring, exchange, and game-ending rules so the project can be evaluated without a local Java installation.

**Neural self-play:** see the [RackZero setup and architecture guide](docs/RACKZERO.md)
for imperfect-information search, PyTorch training, checkpoints and evaluation.

## Highlights

- Standard 15 x 15 board
- Standard 100-tile English distribution
- Two to four players
- Blank-tile letter assignment
- First-move center validation
- Horizontal and vertical placement validation
- Existing-tile reuse and gap validation
- Main-word and cross-word generation
- Premium-square scoring
- 50-point seven-tile bingo bonus
- Correct one-time premium-square consumption
- Tile exchange semantics
- Six-scoreless-turn game ending
- Final rack deductions and going-out bonus
- Exact offline dictionary validation
- Bloom-filter negative pre-check
- LRU and LFU caches
- Bounded loopback TCP dictionary service
- JUnit and Node.js test suites
- GitHub Actions CI on Java 17 and Java 21
- Python Gymnasium interface backed by the Java rules engine via JPype
- Seeded RL episodes, exhaustive legal-move generation, and greedy self-play

## Architecture

```text
Scrabble/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/                         # GitHub Pages browser application
│   ├── app.js                    # UI and browser persistence
│   ├── engine.js                 # Browser rules engine
│   ├── storage.js                # Validated browser saves
│   ├── index.html
│   ├── style.css
│   └── words.txt
├── src/
│   ├── main/
│   │   ├── java/com/masterchiefproject/scrabble/
│   │   │   ├── game/             # Board, scoring, bag, players, console client
│   │   │   ├── dictionary/       # Exact dictionary, Bloom filter, caches
│   │   │   └── server/           # TCP dictionary service
│   │   └── resources/dictionary/
│   └── test/java/com/masterchiefproject/scrabble/
├── tests/
│   └── web/                      # Browser-engine tests
├── package.json
├── pom.xml
└── README.md
```

## Java game engine

### Move validation

`Board` validates and scores a move before mutating board state. Invalid moves are atomic and leave the board unchanged.

A legal move satisfies the applicable Scrabble rules:

1. One to seven new rack tiles are placed.
2. Every placement stays within the board.
3. Existing tiles are never overwritten.
4. New tiles occupy a single row or column.
5. No empty gap exists between the move endpoints unless an existing tile fills it.
6. The first move covers the center square.
7. Later moves connect to the existing board.
8. At least one word of two or more letters is formed.
9. Every main word and cross-word passes the configured `WordValidator`.

`Board.preview(...)` executes the same validation and scoring path without modifying the board.

### Scoring

The engine supports:

- standard English letter values
- double-letter and triple-letter squares
- double-word and triple-word squares
- center-square double-word scoring
- multiple word multipliers in one word
- perpendicular cross-word scoring
- zero-point blank tiles
- 50-point bingo bonus for using all seven rack tiles

Premium squares apply only to newly placed tiles.

### Game coordination

`ScrabbleGame` manages player order, racks, bag state, score changes, passes, exchanges, scoreless-turn tracking, end-game conditions, and final scoring.

A game ends when either:

- the bag is empty and a player uses the final tile in their rack, or
- six consecutive scoreless turns occur

Final scoring deducts remaining rack values and awards the going-out bonus when applicable.

## Dictionary subsystem

Dictionary lookup combines exact validation with lightweight caching and a Bloom-filter pre-check:

```text
word
  |
  v
positive LRU cache
  |
  v
negative LFU cache
  |
  v
Bloom filter pre-check
  |
  v
exact word set
```

The Bloom filter is only used to reject definite negatives. Every possible match is still confirmed against the exact word set, so Bloom-filter false positives cannot legalize invalid words.

`DictionaryManager` also constrains file-backed dictionary access to configured root directories.

## TCP dictionary service

The networking layer preserves the original dictionary-server concept while adding bounded resource use and safer local defaults.

`DictionaryServer` includes:

- loopback-only binding by default
- bounded worker pool
- bounded pending-client queue
- socket read timeouts
- request-size limits
- UTF-8 request validation
- dictionary path-root enforcement
- deterministic shutdown

Protocol:

```text
Q,file1.txt,file2.txt,WORD
C,file1.txt,file2.txt,WORD
```

Responses:

```text
true
false
```

Malformed requests receive an `ERROR,...` response.

After packaging, the local server entry point is:

```bash
java -cp target/scrabble-game-2.1.0.jar \
  com.masterchiefproject.scrabble.server.DictionaryServerMain \
  6000 src/main/resources/dictionary
```

The default bind address is `127.0.0.1`.

## Browser demo

The browser implementation in `docs/` provides:

- two to four local players
- human versus Greedy, approximate DP, local Quackle, or checkpoint-backed DQN play
- agent-versus-agent matches with independent policy choices and Pause/Resume/Step
- DQN enabled only when a compatible locally trained checkpoint is available
- the standard board and tile distribution
- blank-tile assignment
- dictionary validation
- cross-word scoring
- exchange and pass actions
- final scoring and tie handling
- turn history
- autosave and manual save/load
- deep save validation and tile-conservation checks
- persistent dark/light theme
- responsive layout

Automatic and manual saves are stored independently. Valid autosave data can restore a session after reload, while a manual save remains available across new games.

### Play against Greedy or DP

To try the updated interface locally, run this from the repository root, then
open `http://localhost:8000`:

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory docs
```

Click **Play against agent**, select Greedy or DP, enter your name, and start the game. You move first:
select a rack tile, click a board square, and use **Commit move** (or exchange/pass).
The selected agent automatically takes the next turn. Its rack stays hidden and your controls
are locked during its turn. **New game → Local players** retains the original
two-to-four-player pass-and-play mode.

Greedy enumerates legal placements using a trie, uses the existing browser engine
to score each candidate, and chooses the highest immediate score with deterministic
tie-breaking. It exchanges its whole rack if no placement exists and at least
seven bag tiles remain, otherwise passes. This is **not dynamic programming**,
lookahead, or a trained model. DQN requires the separate local training setup below.

**DP is an approximate, finite-horizon dynamic-programming policy**, not an exact
Scrabble solver or a trained model. It uses this memoized Bellman recurrence:

```text
V(position, 0) = 0
V(position, depth) = max over candidates [
    immediate_score + 0.9 * V(after_placement, depth - 1)
]
```

By default, it plans up to **three own placements** and considers the **six
highest immediate-scoring placements per state**, with deterministic tie-breaking.
The same trie/engine-scoring path serves both agents. The memo key includes the
entire board (including blank flags), remaining tile multiset, and remaining depth;
the cache is scoped to one decision. Agent results include search depth, beam
width, forecast value, state count and cache hits for testing/debugging.

Forecasts spend only the current rack: **no refill, opponent move, exchange,
pass, endgame bonus/penalty, or hidden rack/bag contents are simulated**.
No continuation is worth zero. Actual moves still use all normal engine rules,
draws, and final scoring; if no legal placement exists, DP uses the same
exchange/pass fallback as Greedy. These assumptions and beam pruning keep browser
search bounded but can miss stronger moves; DP is **not guaranteed to beat Greedy**.
The browser labels these limitations in the opponent selector. The DP policy is
currently browser-only, not exposed by the Python Gymnasium package.

Move search runs in a Web Worker so the page remains responsive. Only the board,
the agent's own rack, and the bag size are sent to the agent, not your rack or bag
contents. Auto/manual saves preserve the selected opponent; restoring a save on
an agent's turn resumes that turn. Older multiplayer and Greedy saves remain compatible.
Worker failures are shown explicitly with a **Retry Greedy turn** or **Retry DP turn** button.

### Watch agents play each other

Click **Agents vs agents**, choose a policy independently for Player 1 and
Player 2, then **Start game**. Greedy vs DP, either mirror matchup, and reversed
player order are supported. The local server also enables Quackle and trained DQN.
Matches play automatically
with a short delay between turns. **Pause** cancels any in-flight search without
consuming a turn, **Step one turn** executes exactly one turn while paused, and
**Resume** continues automatic playback. Normal placement/exchange/pass controls
are disabled in spectator mode, and agent racks remain hidden during play.

Scores, move history, selected policies and pause state are saved in auto/manual
saves. A running saved match resumes automatically; a paused one stays paused.
Starting or loading another game cancels the previous match's pending work.
Natural engine game-over rules stop playback and announce the winner or tie.

This static GitHub Pages mode runs entirely in JavaScript; it does not call the
Python/Java Gymnasium interface. Future trained agents will need a browser-ready
model or a separately hosted inference service. Publishing the updated `docs/`
files is required before the live GitHub Pages site includes these controls.

## Build and run

Requirements:

- Java 17 or newer
- Maven 3.9 or newer

Full Java verification:

```bash
mvn verify
```

Executable JAR:

```bash
mvn package
java -jar target/scrabble-game-2.1.0.jar
```

The console client supports:

```text
place <row> <col> <H|V> <WORD>
exchange <LETTERS>
pass
help
quit
```

Console coordinates are 1-based.

## Browser verification

The browser engine has no npm runtime dependencies. Node.js 20 or newer is sufficient for its verification suite.

```bash
npm run check:web
npm run test:web
```

The browser tests cover tile conservation, center-square rules, scoring, atomic invalid moves, blank tiles, exchanges, game-ending conditions, save validation, and ties.

## Gymnasium / reinforcement learning

The Python package wraps the **existing Java engine**, rather than implementing a
second scoring or game-transition engine. JPype calls `ScrabbleGame`, `Board.preview`,
and `Board.place` in-process. No dictionary server, GPU, PyTorch, or Maven is needed
for this interface. The Java and browser entry points are unchanged.

Requirements: **Python 3.10+ and a matching-architecture JDK 17+**. Set `JAVA_HOME`
to the JDK directory, or put its `bin` directory on `PATH`.

From this repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m scrabble_gym.build
.\.venv\Scripts\python.exe -m unittest discover -s tests\python -v
.\.venv\Scripts\python.exe -m scrabble_gym --episodes 1 --seed 42
```

Add `--render` to print the board each turn. The demo prints a JSON result per
episode with seed, turn count, scores, winner, and termination reason. Optional
`--word-list`, `--jar-path`, and `--max-turns` flags configure experiments.
On Linux/macOS, use `.venv/bin/python` instead of the Windows executable.

The build helper compiles production Java sources with the same strict compiler
flags as Maven and packages resources into `target/scrabble-gym.jar`. Rebuild after
changing Java sources. Alternatively, `mvn package` produces a usable
`target/scrabble-game-2.1.0.jar`. The Gym JAR takes priority when both exist.
Set `SCRABBLE_JAR` or pass `jar_path=...` to use a different JAR. Python wheels do
not bundle a JVM or engine JAR; outside an editable checkout, supply the JAR
explicitly. Restart Python after rebuilding a JAR already loaded by its JVM.

### Interaction loop

```python
import gymnasium as gym
import scrabble_gym  # Registers Scrabble-v0.
from scrabble_gym import greedy_action

env = gym.make("Scrabble-v0", max_turns=200, render_mode="ansi")
observation, info = env.reset(seed=42)
try:
    terminated = truncated = False
    while not (terminated or truncated):
        action = greedy_action(env.unwrapped)
        observation, reward, terminated, truncated, info = env.step(action)
    print(info["scores"], info["end_reason"])
    print(env.render())
finally:
    env.close()
```

You can also instantiate `ScrabbleEnv` directly. One step is **one player's turn**,
not a full agent/opponent round. Both policies are controlled externally. Every
returned observation belongs to the next player, even after the final turn.
Reward is the acting player's score change, including their endgame adjustment;
it is not a win/loss or zero-sum reward.

### Observations and actions

Observations are a `spaces.Dict`; arrays are independent copies. The opponent's
rack contents and the bag order are hidden, making this a partially observable
two-player environment.

| Key                              | Shape / values                                          |
| -------------------------------- | ------------------------------------------------------- |
| `board`                          | `(15, 15)` int8; empty=0, A=1 through Z=26              |
| `board_blanks`                   | `(15, 15)` binary; placed blanks retain zero face value |
| `rack`                           | `(27,)` counts in `ABCDEFGHIJKLMNOPQRSTUVWXYZ?` order   |
| `scores`                         | `(2,)` int32, absolute player-0/player-1 scores         |
| `player`                         | 0 or 1, player to act                                   |
| `opponent_rack_size`, `bag_size` | Counts only                                             |
| `scoreless_turns`, `turn`        | Consecutive scoreless turns; total completed turns      |

The fixed `spaces.Dict` action contains `kind` (place=0, exchange=1, pass=2),
`row`/`col` (zero-based), `direction` (horizontal=0, vertical=1),
`word` (15 codes, A=1..Z=26 with trailing zero padding), `blanks` (15 binary
flags for **new** blanks), and `exchange` (27 tile counts). A placement's word
includes existing board letters; existing tiles must not be marked as blanks.
Only the fields relevant to the selected kind are used.

For convenience, direct environments also accept `Move` objects:

```python
from scrabble_gym import Move

pass_action = Move().to_action()
# Format examples; these require matching rack and board state:
place_action = Move("place", row=7, col=6, word="CAT", blanks=(1,)).to_action()
exchange_action = Move("exchange", exchange="AE?").to_action()
```

`env.unwrapped.legal_moves()` returns cached `ScoredMove(move, score)` candidates:
all distinct physical placements (including different blank assignments), all
nonempty exchange multisets when the bag has at least seven tiles, and pass.
A trie and perpendicular-word checks prune candidates; **Java previews every
placement and supplies its score**. No top-k cutoff is applied. Greedy selection
maximizes immediate points, otherwise exchanges the largest subset or passes.

`action_space.sample()` generates syntactically valid actions, **not necessarily
legal Scrabble plays**. Choose legal candidates instead. Invalid actions raise
`ValueError` without consuming a turn, tiles, or Java RNG state. Consequently,
unadapted random-action Gymnasium checkers/learning algorithms cannot be used;
the tests check the Gymnasium lifecycle with a legal pass-only action adapter,
and separately test real placement, exchange, and self-play transitions.
For DQN-style learning, score candidate moves with a shared network; candidate
list indices are not stable action IDs across states. Use a fixed-opponent
wrapper or explicit turn-aware self-play targets, not a single-agent Bellman
target applied directly across alternating turns. The optional Double DQN trainer
below implements this fixed-opponent approach.

Natural termination occurs on rack-out with an empty bag or six consecutive
scoreless turns, using Java's final rack deductions/going-out bonus. `max_turns`
causes truncation without final penalties; natural termination takes precedence.
`info` includes scores, current player and dictionary size, plus after a step:
`acting_player`, decoded `move`, `move_score`, `score_adjustments`, and
`end_reason`. Finished episodes include `winner` (0, 1, or `None` for a tie);
at truncation this only compares unfinished scores.

The default lexicon comes from the JAR's licensed demo dictionary, restricted to
2..15 letters. It includes names and is **not a tournament lexicon**. For the
teaching lexicon in your basic example or another appropriately licensed list,
pass `word_list=...`, or use `words={"AT", "CAT", ...}`. Files accept
whitespace-separated ASCII words and full-line `#` comments, uppercase entries,
remove duplicates, and reject invalid tokens. A ZIP containing exactly one `.txt`
lexicon is also accepted without extracting archive paths. WGPO hook tables with
a `//Word` tab-separated header use only the first (word) column; hooks and
statistics are not vocabulary. The `words=` API requires uppercase already.

JPype shares one JVM per Python process. `close()` releases this environment's
game references but deliberately does **not** shut down the JVM, so other
environments and subsequent resets still work. Do not fork a running JVM; for
parallel experiments, use spawned processes that create their own environments.

## Continuous integration

GitHub Actions validates:

- Maven build and JUnit tests on Java 17
- Maven build and JUnit tests on Java 21
- Java compiler warnings via `-Xlint:all -Werror`
- browser JavaScript syntax
- Node.js browser-engine tests
- Python 3.10/3.13 Java-backed Gymnasium integration tests
- native Quackle compilation, HTTP tests, and full-game Java scoring agreement
- optional PyTorch learning tests, replay/target math, checkpoint resume, and Python/browser inference parity

## Off-policy Double DQN against Quackle

The optional PyTorch trainer uses **Double DQN**, not imitation learning:
epsilon-greedy learner moves generate experience against a fixed Quackle
static-policy opponent; random minibatches from replay update `Q(observation, move)`.
Quackle does not supply labels or update its own policy. Build both Java and native
Quackle first using the setup commands above/below, then install the learning extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[learning]"
```

### Training and evaluation

For a fresh, short wiring/learning smoke test:

```powershell
.\.venv\Scripts\python.exe -m scrabble_gym.train_dqn --episodes 2 --batch-size 4 --warmup 4 --target-interval 4
```

This writes `target/dqn/checkpoint.pt`, `policy.json` for browser inference,
and per-episode `training.jsonl`. It is **not enough training to produce a
competitive opponent**. Fresh training refuses to overwrite an existing
checkpoint: use `--resume` to continue or choose a separate `--output` directory.
The schema-3 score-prior network is incompatible with older five-bit/64-32 and
schema-2 checkpoints: start a fresh output directory rather than resuming those weights.

For a longer experiment with a locally supplied dictionary, using five parallel
actor processes (each runs its own Java engine and Quackle; ~0.5 GB RAM each):

```powershell
.\.venv\Scripts\python.exe -m scrabble_gym.train_dqn --word-list WOW24v3.zip --episodes 3000 --workers 5 --threads 4 --output target\dqn-wow24
```

The supplied `WOW24v3.zip` identifies itself as **WGPO Official Words 2024
(WOW24)** from wordgameplayers.org, not NWL2023. It contains **195,383 unique
2..15-letter words**. Its notice permits non-profit use and sharing; preserve
the archive's notice and check its terms for your use. The user-supplied ZIP is
Git-ignored, not bundled or downloaded by this project. Both Java validation and
native Quackle use the selected vocabulary; a separate native DAWG is cached
under Git-ignored `target/quackle/lexicons/`, using temporary normalized words.
The shared demo dictionary and its native DAWG are not replaced.

Rebuild the native bridge after upgrading the code (`python -m
scrabble_gym.quackle_build`) so it supports selecting the generated DAWG.
`--word-list` must be supplied again for resume and evaluation; its SHA-256 must
match the checkpoint. Plain authorized NWL2023 word-list files are supported too.

Training defaults to **3,000 additional episodes**, a 500-decision replay warmup,
batch size 64, learning rate `0.00025`, `gamma=0.97`, and one minibatch update
every two learner decisions (`--train-every`). With `--workers N`, actors play
games with the latest published weights while the main process learns; one
actor plays each game end to end. `--time-limit-seconds` stops after complete
episodes and saves a checkpoint/export. Resume the resulting model:

```powershell
.\.venv\Scripts\python.exe -m scrabble_gym.train_dqn --word-list WOW24v3.zip --resume target\dqn-wow24\checkpoint.pt --episodes 1000 --workers 5
```

`--episodes` means additional episodes, not the desired total. Training alternates
the learner between first and second player, using reproducible episode seeds.
Training runs only while this command runs; browser games do not update weights.
CPU is the default with two PyTorch threads. `--device cuda` requires a compatible
PyTorch CUDA installation/GPU. Candidate enumeration and Quackle still run on CPU.

Evaluate without exploration or weight/replay updates on held-out seeds:

```powershell
.\.venv\Scripts\python.exe -m scrabble_gym.train_dqn --word-list WOW24v3.zip --evaluate target\dqn-wow24\checkpoint.pt --episodes 200 --seed 100000 --workers 5 --output target\dqn-wow24-comparison
.\.venv\Scripts\python.exe -m scrabble_gym.train_dqn --word-list WOW24v3.zip --baseline untrained --episodes 200 --seed 100000 --workers 5 --output target\dqn-wow24-comparison
.\.venv\Scripts\python.exe -m scrabble_gym.train_dqn --word-list WOW24v3.zip --baseline greedy --episodes 200 --seed 100000 --workers 5 --output target\dqn-wow24-comparison
```

Evaluation plays each seed in **both learner seats** (use an even episode count).
Use identical dictionary, seeds, episode counts and turn limits for all three
commands. The untrained baseline is a fresh network initialized with
`--baseline-network-seed` (default 42); match the original training seed for
a controlled before/after comparison. Because the residual output starts at zero,
the untrained DQN plays the highest-scoring move in its acting set, so training
must beat score-greedy play to show improvement. `--seed` selects held-out game seeds.
Greedy maximizes immediate score, otherwise exchanges the largest subset or
passes. All face the same fixed Quackle policy. Evaluation refuses seeds that
overlap the checkpoint's training seeds.

Evaluation appends `evaluation.jsonl` beside the checkpoint unless `--output`
is specified; baselines write `untrained.jsonl` and `greedy.jsonl`. Each episode logs
scores, learner seat, score differential, reward, natural termination/truncation,
and runtime; training also logs epsilon, replay size, updates and mean Huber loss.
Summary JSON files report win rate with a 95% Wilson interval, mean score
difference and its standard error, ties, per-seat game/win counts, and truncations.
Win rate and score-difference statistics include **only naturally completed
games**; ties are not wins. The Wilson interval treats games as independent;
for formal significance testing, use seed-level paired analysis because the two
seats share a seed. Compare held-out score difference and wins against both
baselines, not training loss alone. Small pilots or 20-game evaluations are
diagnostic, not sufficient to establish competitive strength.

### WOW24 CPU experiment

The schema-3 run uses the supplied WOW24 archive, initialization seed 42,
five actor processes, a 500-decision warmup, and one optimizer update per
learner decision (`--train-every 1`). It completed **2,000 training games and
25,254 optimizer updates** within a 12,600-second episode-boundary budget.
Its checkpoint is `target/dqn-wow24/checkpoint.pt`; the training summary is
`target/dqn-wow24/training-summary.json`.

Matched evaluation uses seeds 100000..100049, both seats per seed, for 100
games per policy against the same fixed Quackle opponent. Training score
improvements alone do not demonstrate improvement over the baselines.

The full-horizon (`gamma=0.97`) checkpoint produced:

| Policy | Wins against Quackle | Mean score difference |
| --- | ---: | ---: |
| Untrained score-prior DQN | 39/100 | -32.54 |
| DQN after 2,000 training games | 10/100 | -127.47 |
| Immediate-score Greedy | 39/100 | -32.54 |

All games ended naturally. The untrained and Greedy baselines coincide here
because the untrained residual is zero. The trained DQN's paired score gain
relative to Greedy was **-94.93 points** (95% seed-cluster bootstrap interval
approximately **[-120, -70]**). **This checkpoint regressed, rather than improving
on either baseline, and must not be presented as a competitive opponent.**
The summaries, invocation configuration and paired analysis are saved locally
in `target/dqn-wow24-comparison/comparison.json`.

A bounded shorter-horizon experiment resumes this checkpoint's existing replay,
sets `gamma=0.5` and learning rate `0.0001`, synchronizes the target to the online
network, and performs 1,500 further minibatch updates (batch size 64). It collects
**no additional training games**: its totals are 2,000 games and 26,754 updates.
Its checkpoint/export and exact configuration are under
`target/dqn-wow24-short-horizon/`. A separate 40-game diagnostic on seeds
200000..200019 yielded 12 wins and a -77.875 mean score difference; that sample
alone is not a matched improvement claim. The full matched evaluation is saved
under `target/dqn-wow24-comparison-short-horizon/`.

On the same 100 benchmark games, the refit won **16 games**, tied one, and had
a **-93.92** mean score difference. Its paired gain over the original trained
checkpoint was +33.55 points (95% seed-cluster interval [10.68, 56.12]), but its
gain over Greedy remained **-61.38** (interval [-84.16, -38.52]). This bounded
correction reduced the regression; **neither trained model surpassed the
baselines**. Both comparisons and the refit configuration are included in
`target/dqn-wow24-comparison/comparison.json`. More training episodes and richer
features did not, in this experiment, establish a competitive learned policy.

### Learning semantics

- The generated legal set contains **all legal physical placements**, every
  allowed nonempty exchange multiset, and pass. The DQN acts, explores and takes
  its Bellman max over **up to 64 legal moves, including pass**: start with the
  highest-scoring candidates and replace the last with pass if necessary
  (`--top-candidates`, stored in the checkpoint and `policy.json`; the browser
  applies the same cut). Without this cut, argmax over thousands of rarely sampled
  moves picked low-scoring outliers with unconstrained Q extrapolations.
  A shared scalar Q-network scores each `(observation, candidate)`; candidate
  indices are never treated as persistent action identities.
- One learner transition includes its move **and Quackle's response**, unless
  the learner ends the game. Next-state maximization therefore always uses the
  learner's next rack and legal moves, never Quackle's rack/perspective.
- Reward is the change in `(learner score - opponent score) / 100` across that
  round, including final rack penalties/bonuses. Discount `gamma=0.97` applies
  once per learner decision, not once per individual player's turn.
- Targets are
  `r + gamma * Q_target(s', argmax_legal Q_online(s', action))`.
  Natural terminals have no bootstrap. `--max-decisions` (default 100) truncates
  only after a complete round and retains a learner-state bootstrap; truncated
  games are not recorded as wins.
- `Q(s, a) = score(a)/100 + residual(s, a)`: the immediate score is a fixed
  prior and the network learns only the correction (leave value, board
  openings, opponent reply), like Quackle's score-plus-leave valuation. The
  residual's output layer is zero-initialized.
- The residual is an MLP with hidden layers **256/128/64** and ReLU. Each
  board/placement square has **35 features**: 26 one-hot letter channels,
  a physical-blank flag, normalized row/column coordinates, four directional
  neighbor-occupancy channels (north/south/west/east, with no edge wrap), and
  word/letter premium bonuses. Coordinates and premium bonuses are attached to
  occupied/placed squares; empty squares have no letter channel. Adjacent squares
  are thus explicitly related in the input, rather than only arbitrary positions
  in a flattened five-bit vector. This is **not a CNN** and does not enforce
  translation-invariant learned filters. Inputs also include own
  rack/leave/exchanged counts, public scores, opponent
  rack **size**, bag **count**, scoreless turns, action kind and immediate score.
  It never sees hidden racks or bag contents. This is a feed-forward,
  partially-observed baseline, not a recurrent belief-state Scrabble solver.
- Adam, Huber loss and gradient clipping update the online network. The target
  network stays frozen and synchronizes every 250 optimizer updates by default.
  Epsilon decays from 1.0 to 0.05 over 10,000 learner decisions (about 700
  games). 80% of exploratory moves (`--focused-exploration`) sample one of the
  10 highest-scoring candidates; the rest are uniform over the acting set.
- Replay is capped by both 5,000 transitions and 384 MiB of NumPy payload
  (Python/container overhead is additional). Compact legal candidates are stored,
  while Q evaluation reuses the state projection and multiplies sparse candidate
  features in the first layer; a minibatch's Bellman targets are computed in one
  batched sparse product per network. Old transitions are evicted as needed.
  `--replay-capacity`, `--replay-mb`, `--batch-size` and `--warmup` adjust these
  limits; a single transition exceeding the memory budget fails explicitly.

Checkpoints are saved atomically every 100 completed episodes
(`--checkpoint-every`) and when the run ends, and retain
online/target weights, optimizer, replay, counters and random-generator states.
Resume restores these and the checkpoint's gamma, learning rate, target
interval and top-candidate count. Exploration/replay limits remain command-line settings; keep them
consistent for reproducible continuation. Dictionary hash and Quackle revision
are checked; rebuild/retrain rather than silently mixing incompatible data.
Only load checkpoints you trust, even though loading uses PyTorch `weights_only`.
Generated checkpoints/logs stay in Git-ignored `target/`. `runs.jsonl` records
each training invocation's arguments, dictionary hash and opponent revision.
An interrupted run can lose progress since the last completed save; a normal
time-budget stop saves at the next episode boundary.

### Play the exported network

Run the local Quackle server and open http://127.0.0.1:8766/:

```powershell
.\.venv\Scripts\python.exe -m scrabble_gym.quackle_server --port 8766
```

The server serves `target/dqn/policy.json`; refresh the page after training.
It uses the shared demo dictionary and rejects models trained on other
vocabularies. The WOW24 experiment remains a Python training/evaluation model;
copying its export into the demo model location does not make it compatible.
Choose **DQN** in **Play against agent** or on either side of **Agents vs agents**.
The UI shows the model's training episode/update count, without promising
strength. Checkpoints with zero optimizer updates stay disabled.
Browser inference runs in the existing worker using the exported network:
no Python inference calls, hidden opponent data, or Greedy fallback.
Python/JavaScript feature, Q-value, and candidate-set parity are tested.
Pause, Step, save/load, stale-result cancellation and Retry apply to DQN too.
Saved games store the selected policy, **not a snapshot of its weights**;
reload uses the current exported model.

DQN remains disabled on GitHub Pages/ordinary static servers or with a
missing/mismatched model. A checkpoint in another output directory must be
explicitly exported/copied to `target/dqn/policy.json` for this local UI.
The browser still uses its bundled demonstration dictionary. A model trained
on WOW24/NWL2023 is intentionally unavailable there. Plain-text training sources
would require the same appropriately licensed word-list bytes in the browser;
archive-trained models additionally require archive loading/hash support that
the current browser does not implement. Copying a normalized ZIP vocabulary to
the browser alone will not satisfy the source-file hash check.

## Dictionary data

The bundled offline word list is derived from CMU Pronouncing Dictionary data and is intended for the programming demonstration rather than official tournament play.

The CMU dictionary license is included with both Java and browser runtime data:

```text
src/main/resources/dictionary/CMUDICT-LICENSE.txt
docs/CMUDICT-LICENSE.txt
```

The dictionary implementation is replaceable through the `WordValidator` abstraction.

For North American tournament experiments, NASPA's
[NWL2023](https://www.scrabbleplayers.org/w/NWL2023) is a licensed tournament
lexicon, distinct from the CMU demo data and the supplied WGPO WOW24 list.
Obtain authorized access through NASPA and provide your local file using
`--word-list`; this project does not fetch or redistribute NWL2023.

## Deployment

The browser adaptation is deployed from `docs/` through GitHub Pages.

**Live demo:** https://masterchiefproject.github.io/Scrabble/

## Local native Quackle opponent

The official [Quackle engine](https://github.com/quackle/quackle) is available as
a **local-only static-policy opponent**. It generates moves natively and ranks
them with Quackle's `ScorePlusLeaveEvaluator` and English rack-leave tables.
This is **not Quackle's Monte Carlo/simulation player**, a trained model, or an
exact endgame solver. It can exchange tiles even when a scoring play exists.

The bridge uses the same standard 15-by-15 board, blanks, scoring and
`docs/words.txt` demo lexicon as the browser. The demo lexicon is **not a
tournament word list**; Quackle's bundled tournament lexicons are not used.
The browser validates each returned move and its score before committing it.
Only the public board, acting player's rack and bag count are sent to Quackle;
opponent racks and actual bag contents are not provided.

### Build

Install the Python package as described in the Gymnasium setup above. Building
also requires Git, CMake 3.16+, and a C++17 compiler. On Windows, install
**Visual Studio 2022 Build Tools**, the **MSVC v143 x64/x86 toolset**, and a
**Windows SDK**. The build helper selects the VS 2022 x64 generator. The older
Windows headers in Strawberry MinGW 13.2 do not build this Quackle revision.
Linux/macOS can use their normal CMake C++ toolchain; no Qt is needed for this
core-only bridge (Quackle's separate desktop application does require Qt).

Quackle is a separate, Git-ignored checkout. Fresh clones need to download it:

```powershell
git clone https://github.com/quackle/quackle.git third_party\quackle
git -C third_party\quackle checkout 35e11bb21a3bf362c533541d0b5fe8ad92a0c08f
.\.venv\Scripts\python.exe -m scrabble_gym.quackle_build
```

The helper builds `native/quackle`, writes the shared demo DAWG and records the
upstream revision and dictionary hash under `target/quackle`. Rebuild after
changing `docs/words.txt`; the server refuses a mismatched dictionary.

### Play

```powershell
.\.venv\Scripts\python.exe -m scrabble_gym.quackle_server --port 8766
```

Keep that terminal running and open **http://127.0.0.1:8766/**. In **Play against
agent**, select **Quackle static**. Or choose it on either side of **Agents vs
agents**, alongside Greedy or DP. Pause, Step, save/load and Retry work with the
native policy too. Saves belong to the browser's exact origin, so saves from
another port or `localhost` are separate.

Quackle remains disabled on GitHub Pages and ordinary static servers; those
cannot execute a native engine. The local server binds only to loopback, serves
only browser assets, and accepts same-origin JSON requests. It is a local
development/play server, not a public deployment service. Each native move has a
20-second timeout, with at most one native request running at once. Pausing or
starting/loading a game aborts the browser request and ignores late replies;
an already-running native calculation can finish in the background. A brief
busy error requires **Retry**, never a silent switch to Greedy.

Run the integration tests after building the Java and Quackle engines:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests\python -p test_quackle.py -v
```

Native tests skip when no Quackle manifest exists; protocol, HTTP and dictionary
tests still run. Native tests include blank bingos, exchange/pass behavior, and
complete seeded games checked against the Java rules engine.

### Licensing

Quackle and the linked bridge in `native/quackle` are
**GPL-3.0-or-later**, not covered by this project's root license. The full
upstream license and notices remain in `third_party/quackle/LICENSE`; the bridge
notice is in `native/quackle/LICENSE`. Preserve them, provide corresponding
source as required, and review upstream strategy-data licenses before
redistributing a linked binary or combined product. CMU demo dictionary data
retains the notices in `docs/CMUDICT-LICENSE.txt`. No Quackle source or binary is
bundled in this repository's ordinary Java/Python builds.

## Project evolution

The repository began as a university Book Scrabble assignment. The current version uses a standard Maven structure, separates production code from tests and browser assets, strengthens board and dictionary behavior, preserves the networking component, and adds a polished browser demonstration with automated verification.

http://localhost:8000

## AlphaZero-style Scrabble self-play

RackZero is a modular PyTorch pipeline built on the existing Java/Gymnasium
engine. It has:

* tile-conserving, Rao-Blackwellized opponent-rack beliefs;
* five sampled-chance search variants;
* a configurable ResNet that scores the generator's legal moves (the original
  MLP is kept as a baseline);
* parallel self-play workers, batched (bf16) GPU training and full
  checkpoint/resume;
* evaluation with confidence intervals, and information-leakage tests;
* a warm start from Quackle's static player, plus AlphaZero-family training
  options: root noise, playout cap randomization, Gumbel root search, q/z value
  targets and Reanalyse. A Quackle opponent is available for evaluation.

**[Setup, architecture and commands](docs/RACKZERO.md)** |
**[A100 / NCSA Delta guide](docs/A100_TRAINING.md)** |
**[Scaling experiment plan](docs/SCALING_PLAN.md)** |
**[Audit and results report](docs/RACKZERO_REPORT.md)** |
**[AlphaZero reference review](docs/ALPHAZERO_REFERENCE.md)**

With Python 3.10+ and JDK 17+ installed, from the repository root:

```powershell
python -m pip install -e ".[learning]"
python -m scrabble_gym.build
python train.py --config configs\smoke.yaml --run-dir runs\smoke
python evaluate.py --checkpoint runs\smoke --games 4 --opponents greedy random
```

The smoke configuration needs no external lexicon. For full-dictionary
training, provide a permitted local WOW24 lexicon as documented above:

```powershell
python train.py --config configs\medium.yaml --set selfplay.workers=8
python evaluate.py --checkpoint runs\<run-directory> --opponents greedy heuristic --games 200
```

This is a functional research framework, not a claim of tournament strength.
WOW24 has separate non-profit-use terms; it is not bundled or relicensed.
Only load trusted checkpoints.
