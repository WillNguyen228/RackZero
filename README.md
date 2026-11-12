# Scrabble

[![CI](https://github.com/MasterChiefProject/Scrabble/actions/workflows/ci.yml/badge.svg)](https://github.com/MasterChiefProject/Scrabble/actions/workflows/ci.yml)

**Scrabble** is a Java 17 word-board game engine with full board validation and scoring, exact dictionary lookup, a bounded TCP dictionary service, automated tests, and a browser-playable multiplayer adaptation.

**Playable demo:** [RackZero browser demo](https://willnguyen228.github.io/RackZero/)

The browser demo is deployed from `docs/` to GitHub Pages on pushes to `main`.

The Java implementation is the primary rules engine. A JavaScript browser adaptation mirrors the main placement, scoring, exchange, and game-ending rules so the project can be evaluated without a local Java installation.

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
remove duplicates, and reject invalid tokens. The `words=` API requires uppercase
already.

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
For example, continue an existing smoke checkpoint for another 1,000 episodes:

```powershell
.\.venv\Scripts\python.exe -m scrabble_gym.train_dqn --resume target\dqn\checkpoint.pt --episodes 1000
```

`--episodes` means additional episodes, not the desired total. Training alternates
the learner between first and second player, using reproducible episode seeds.
Training runs only while this command runs; browser games do not update weights.
CPU is the default with two PyTorch threads. `--device cuda` requires a compatible
PyTorch CUDA installation/GPU. Candidate enumeration and Quackle still run on CPU.

Evaluate without exploration or weight/replay updates on held-out seeds:

```powershell
.\.venv\Scripts\python.exe -m scrabble_gym.train_dqn --evaluate target\dqn\checkpoint.pt --episodes 100 --seed 100000
```

Evaluation appends `evaluation.jsonl` beside the checkpoint. Each episode logs
scores, learner seat, score differential, reward, natural termination/truncation,
and runtime; training also logs epsilon, replay size, updates and mean Huber loss.
The summary reports win rate **only for naturally completed games**, excluding
ties from wins, and separately counts truncations. Avoid evaluation seeds used
for training. Falling training loss alone is not evidence of stronger play.

### Learning semantics

- The variable action set contains **all generated legal physical placements**,
  every allowed nonempty exchange multiset, and pass. There is no top-k cutoff.
  A shared scalar Q-network scores each `(observation, candidate)`; candidate
  indices are never treated as persistent action identities.
- One learner transition includes its move **and Quackle's response**, unless
  the learner ends the game. Next-state maximization therefore always uses the
  learner's next rack and legal moves, never Quackle's rack/perspective.
- Reward is the change in `(learner score - opponent score) / 100` across that
  round, including final rack penalties/bonuses. Discount `gamma=0.99` applies
  once per learner decision, not once per individual player's turn.
- Targets are
  `r + gamma * Q_target(s', argmax_legal Q_online(s', action))`.
  Natural terminals have no bootstrap. `--max-decisions` (default 100) truncates
  only after a complete round and retains a learner-state bootstrap; truncated
  games are not recorded as wins.
- The network is an MLP with hidden layers 64/32 and ReLU. Features encode
  each board/placement square as five letter-identity bits plus a separate
  physical-blank flag, own rack/leave/exchanged counts, public scores, opponent
  rack **size**, bag **count**, scoreless turns, action kind and immediate score.
  It never sees hidden racks or bag contents. This is a feed-forward,
  partially-observed baseline, not a recurrent belief-state Scrabble solver.
- Adam, Huber loss and gradient clipping update the online network. The target
  network stays frozen and synchronizes every 100 optimizer updates by default.
  Epsilon decays from 1.0 to 0.05 over 10,000 learner decisions.
- Replay is capped by both 5,000 transitions and 128 MiB of NumPy payload
  (Python/container overhead is additional). Compact legal candidates are stored,
  while Q evaluation runs in chunks. Old transitions are evicted as needed.
  `--replay-capacity`, `--replay-mb`, `--batch-size` and `--warmup` adjust these
  limits; a single transition exceeding the memory budget fails explicitly.

Checkpoints are saved atomically at completed episode boundaries and retain
online/target weights, optimizer, replay, counters and random-generator states.
Resume restores these and the checkpoint's gamma, learning rate and target
interval. Exploration/replay limits remain command-line settings; keep them
consistent for reproducible continuation. Dictionary hash and Quackle revision
are checked; rebuild/retrain rather than silently mixing incompatible data.
Only load checkpoints you trust, even though loading uses PyTorch `weights_only`.
Generated checkpoints/logs stay in Git-ignored `target/`.

### Play the exported network

Run the local Quackle server and open http://127.0.0.1:8766/:

```powershell
.\.venv\Scripts\python.exe -m scrabble_gym.quackle_server --port 8766
```

The server serves `target/dqn/policy.json`; refresh the page after training.
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
The lexicon remains the shared non-tournament demonstration lexicon.

## Dictionary data

The bundled offline word list is derived from CMU Pronouncing Dictionary data and is intended for the programming demonstration rather than official tournament play.

The CMU dictionary license is included with both Java and browser runtime data:

```text
src/main/resources/dictionary/CMUDICT-LICENSE.txt
docs/CMUDICT-LICENSE.txt
```

The dictionary implementation is replaceable through the `WordValidator` abstraction.

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
