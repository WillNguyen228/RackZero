// SPDX-License-Identifier: GPL-3.0-or-later
//
// Quackle static-player bridge.
//
//   quackle_bridge [lexicon.dawg] [--top N] [--serve]
//
// Input per position (whitespace separated): 225 board cells ('.' empty,
// 'A'-'Z' tile, 'a'-'z' blank), rack ('A'-'Z', '?'), bag size.
//
// Default output (one-shot, unchanged): the best move by Quackle's static
// score-plus-leave equity, as one PLACE/EXCHANGE/PASS block.
// --top N: "MOVES k", then for each of the k <= N best moves in equity order a
// line "EQUITY e" followed by its block.
// --serve: answer positions until end of input; every answer (or "ERROR msg")
// ends with a line "END" and is flushed, so one process serves many requests.
// --sim SECONDS: play Quackle's championship player (Smart Bogowin: 2-ply
// simulation over sampled opponent racks and draws, chosen by win probability;
// exact endgame player when the bag is empty) with SECONDS per turn instead of
// the static player. Each request then also carries the unseen tiles (bag plus
// opponent rack; '?' blank; "-" if none) and both scores:
//   <cells> <rack> <bag size> <unseen> <own score> <opponent score>
// and the answer is "MOVES 1", "EQUITY e" and the chosen move's block.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "alphabetparameters.h"
#include "board.h"
#include "bogowinplayer.h"
#include "boardparameters.h"
#include "datamanager.h"
#include "evaluator.h"
#include "game.h"
#include "generator.h"
#include "lexiconparameters.h"
#include "strategyparameters.h"

namespace {

void setup(Quackle::DataManager& data, const std::string& lexicon) {
    // Upstream's default EnglishBoard is a nonstandard 13x17 demonstration layout.
    auto* layout = new Quackle::BoardParameters;
    for (const auto& cell : std::vector<std::pair<int, int>>{
             {0,0},{0,7},{0,14},{7,0},{7,14},{14,0},{14,7},{14,14}})
        layout->setWordMultiplier(cell.first, cell.second, Quackle::BoardParameters::tws);
    for (int i = 1; i <= 4; ++i)
        for (const auto& cell : std::vector<std::pair<int, int>>{
                 {i,i},{i,14-i},{14-i,i},{14-i,14-i}})
            layout->setWordMultiplier(cell.first, cell.second, Quackle::BoardParameters::dws);
    layout->setWordMultiplier(7, 7, Quackle::BoardParameters::dws);
    for (const auto& cell : std::vector<std::pair<int, int>>{
             {1,5},{1,9},{5,1},{5,5},{5,9},{5,13},{9,1},{9,5},{9,9},{9,13},{13,5},{13,9}})
        layout->setLetterMultiplier(cell.first, cell.second, Quackle::BoardParameters::tls);
    for (const auto& cell : std::vector<std::pair<int, int>>{
             {0,3},{0,11},{2,6},{2,8},{3,0},{3,7},{3,14},{6,2},{6,6},{6,8},{6,12},
             {7,3},{7,11},{8,2},{8,6},{8,8},{8,12},{11,0},{11,7},{11,14},
             {12,6},{12,8},{14,3},{14,11}})
        layout->setLetterMultiplier(cell.first, cell.second, Quackle::BoardParameters::dls);
    data.setBoardParameters(layout);
    data.setAppDataDirectory("third_party/quackle/data");
    data.setUserDataDirectory("target/quackle");
    data.lexiconParameters()->loadDawg(lexicon);
    if (!data.lexiconParameters()->hasDawg())
        throw std::runtime_error("Demo lexicon not built");
    data.strategyParameters()->initialize("default_english");
    if (!data.strategyParameters()->hasWorths() || !data.strategyParameters()->hasSyn2())
        throw std::runtime_error("Quackle rack-leave strategy data unavailable");
    data.setEvaluator(new Quackle::ScorePlusLeaveEvaluator);
}

void writeMove(std::ostream& out, Quackle::DataManager& data, const Quackle::Move& move,
               const std::string& rackText) {
    auto* alphabet = data.alphabetParameters();
    if (!std::isfinite(move.equity)) throw std::runtime_error("Invalid engine equity");
    std::vector<bool> used(rackText.size(), false);
    auto rackIndex = [&](char letter) {
        for (std::size_t j = 0; j < rackText.size(); ++j) {
            if (!used[j] && rackText[j] == letter) {
                used[j] = true;
                return static_cast<int>(j);
            }
        }
        throw std::runtime_error("Engine move does not match rack");
    };
    if (move.action == Quackle::Move::Place) {
        struct Placement { int row, col, index; char letter; bool blank; };
        std::vector<Placement> placements;
        int offset = 0;
        for (auto letter : move.tiles()) {
            int row = move.startrow + (move.horizontal ? 0 : offset);
            int col = move.startcol + (move.horizontal ? offset : 0);
            ++offset;
            if (Quackle::Move::isAlreadyOnBoard(letter)) continue;
            bool blank = alphabet->isBlankLetter(letter);
            auto decoded = alphabet->userVisible(alphabet->clearBlankness(letter));
            if (decoded.size() != 1) throw std::runtime_error("Non-English engine tile");
            placements.push_back({row, col, rackIndex(blank ? '?' : decoded[0]), decoded[0], blank});
        }
        out << "PLACE " << move.score << " " << move.equity << " " << placements.size() << "\n";
        for (const auto& p : placements)
            out << p.row << " " << p.col << " " << p.index << " " << p.letter << " " << p.blank << "\n";
    } else if (move.action == Quackle::Move::Exchange) {
        std::vector<int> indexes;
        for (auto letter : move.tiles()) {
            auto decoded = alphabet->userVisible(letter);
            if (decoded.size() != 1) throw std::runtime_error("Invalid exchange tile");
            indexes.push_back(rackIndex(decoded[0]));
        }
        out << "EXCHANGE " << indexes.size();
        for (int index : indexes) out << " " << index;
        out << "\n";
    } else if (move.action == Quackle::Move::Pass) {
        out << "PASS\n";
    } else {
        throw std::runtime_error("Unsupported engine move");
    }
}

// Returns false at end of input.
bool solve(std::istream& in, std::ostream& out, Quackle::DataManager& data, int top,
           int simSeconds) {
    std::string cells, rackText, unseenText;
    int bagSize, ownScore = 0, oppScore = 0;
    if (!(in >> cells)) return false;
    if (!(in >> rackText >> bagSize) || cells.size() != 225
        || rackText.empty() || rackText.size() > 7 || bagSize < 0 || bagSize > 100)
        throw std::runtime_error("Invalid position");
    if (simSeconds > 0) {
        if (!(in >> unseenText >> ownScore >> oppScore))
            throw std::runtime_error("Sim requests need unseen tiles and both scores");
        if (unseenText == "-") unseenText.clear();
        for (char c : unseenText)
            if (c != '?' && !(c >= 'A' && c <= 'Z'))
                throw std::runtime_error("Invalid unseen letter");
        int oppSize = static_cast<int>(unseenText.size()) - bagSize;
        if (oppSize < 0 || oppSize > 7) throw std::runtime_error("Unseen tiles do not match bag size");
    }
    for (char c : cells)
        if (c != '.' && !(c >= 'A' && c <= 'Z') && !(c >= 'a' && c <= 'z'))
            throw std::runtime_error("Invalid board letter");
    for (char c : rackText)
        if (c != '?' && !(c >= 'A' && c <= 'Z'))
            throw std::runtime_error("Invalid rack letter");

    auto* alphabet = data.alphabetParameters();
    Quackle::Board board;
    board.prepareEmptyBoard();
    for (int i = 0; i < 225; ++i) {
        if (cells[i] == '.') continue;
        board.makeMove(Quackle::Move::createPlaceMove(
            i / 15, i % 15, true, alphabet->encode(std::string(1, cells[i]))));
    }
    Quackle::PlayerList players;
    players.push_back(Quackle::Player("Agent", Quackle::Player::ComputerPlayerType, 1));
    players.push_back(Quackle::Player("Unknown", Quackle::Player::HumanPlayerType, 2));
    players[0].setScore(ownScore);
    players[1].setScore(oppScore);
    Quackle::GamePosition position(players);
    // Quackle copies pregame positions with no current player until the first increment.
    position.incrementTurn();
    if (!position.setCurrentPlayer(1)) throw std::runtime_error("Cannot set player");
    position.setBoard(board);
    position.setCurrentPlayerRack(Quackle::Rack(alphabet->encode(rackText)), false);
    if (simSeconds > 0) {
        // The real unseen tiles: the opponent's rack gets the right number of them
        // (the simulator re-deals it from all unseen tiles every iteration; with
        // the bag empty it is exact) and the bag the rest.
        std::size_t oppSize = unseenText.size() - static_cast<std::size_t>(bagSize);
        Quackle::Bag bag;
        bag.clear();
        if (bagSize > 0) bag.toss(alphabet->encode(unseenText.substr(oppSize)));
        position.setBag(bag);
        position.setOppRack(Quackle::Rack(alphabet->encode(unseenText.substr(0, oppSize))), false);
        position.ensureBoardIsPreparedForAnalysis();
        Quackle::SmartBogowin player;
        Quackle::ComputerParameters parameters = player.parameters();
        parameters.secondsPerTurn = simSeconds;
        player.setParameters(parameters);
        player.setThreadCount(1, Quackle::ThreadQoS::Balanced);
        player.setPosition(position);
        // The player prints its analysis to std::cout, which carries our protocol.
        std::ostringstream discard;
        auto* saved = std::cout.rdbuf(discard.rdbuf());
        Quackle::Move move;
        try {
            move = player.move();
        } catch (...) {
            std::cout.rdbuf(saved);
            throw;
        }
        std::cout.rdbuf(saved);
        if (!std::isfinite(move.equity)) move.equity = 0;
        // Quackle's endgame player passes by "exchanging"; with fewer than 7
        // tiles in the bag an exchange is illegal, so report it as the pass it is.
        if (move.action == Quackle::Move::Exchange && bagSize < 7) {
            double equity = move.equity;
            move = Quackle::Move::createPassMove();
            move.equity = equity;
        }
        std::ostringstream answer;
        answer << "MOVES 1\nEQUITY " << move.equity << "\n";
        writeMove(answer, data, move, rackText);
        out << answer.str();
        return true;
    }
    Quackle::Bag bag;
    bag.clear();
    // Static evaluation needs only bag size, never actual hidden bag contents.
    bag.toss(Quackle::LongLetterString(bagSize, QUACKLE_BLANK_MARK));
    position.setBag(bag);
    position.ensureBoardIsPreparedForAnalysis();
    Quackle::Generator generator(position);
    generator.kibitz(std::max(1, top), bagSize < 7 ? Quackle::Generator::CannotExchange : 0);
    const auto& moves = generator.kibitzList();
    if (moves.empty()) throw std::runtime_error("Engine returned no moves");
    std::ostringstream answer;
    if (top > 0) {
        std::size_t count = std::min<std::size_t>(moves.size(), static_cast<std::size_t>(top));
        answer << "MOVES " << count << "\n";
        for (std::size_t i = 0; i < count; ++i) {
            answer << "EQUITY " << moves[i].equity << "\n";
            writeMove(answer, data, moves[i], rackText);
        }
    } else {
        writeMove(answer, data, moves.at(0), rackText);
    }
    out << answer.str();
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string lexicon = "target/quackle/demo.dawg";
    int top = 0, simSeconds = 0;
    bool serve = false;
    try {
        bool lexiconGiven = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--serve") {
                serve = true;
            } else if (arg == "--top") {
                if (i + 1 >= argc) throw std::runtime_error("--top needs a count");
                top = std::atoi(argv[++i]);
                if (top < 1 || top > 1000) throw std::runtime_error("--top must be 1..1000");
            } else if (arg == "--sim") {
                if (i + 1 >= argc) throw std::runtime_error("--sim needs seconds per turn");
                simSeconds = std::atoi(argv[++i]);
                if (simSeconds < 1 || simSeconds > 600) throw std::runtime_error("--sim must be 1..600");
            } else if (!lexiconGiven && arg.rfind("--", 0) != 0) {
                lexicon = arg;
                lexiconGiven = true;
            } else {
                throw std::runtime_error("Expected [lexicon] [--top N] [--sim SECONDS] [--serve]");
            }
        }
        Quackle::DataManager data;
        setup(data, lexicon);
        if (!serve) {
            if (!solve(std::cin, std::cout, data, top, simSeconds)) throw std::runtime_error("Invalid position");
            return 0;
        }
        std::string line, request;
        // One request per line: "<cells> <rack> <bag>".
        while (std::getline(std::cin, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            std::istringstream in(line);
            try {
                solve(in, std::cout, data, top, simSeconds);
            } catch (const std::exception& error) {
                std::cout << "ERROR " << error.what() << "\n";
            }
            std::cout << "END" << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
