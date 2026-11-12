// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "alphabetparameters.h"
#include "board.h"
#include "boardparameters.h"
#include "datamanager.h"
#include "evaluator.h"
#include "game.h"
#include "generator.h"
#include "lexiconparameters.h"
#include "strategyparameters.h"

int main() {
    try {
        std::string cells, rackText;
        int bagSize;
        if (!(std::cin >> cells >> rackText >> bagSize) || cells.size() != 225
            || rackText.empty() || rackText.size() > 7 || bagSize < 0 || bagSize > 100)
            throw std::runtime_error("Invalid position");
        for (char c : cells)
            if (c != '.' && !(c >= 'A' && c <= 'Z') && !(c >= 'a' && c <= 'z'))
                throw std::runtime_error("Invalid board letter");
        for (char c : rackText)
            if (c != '?' && !(c >= 'A' && c <= 'Z'))
                throw std::runtime_error("Invalid rack letter");

        Quackle::DataManager data;
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
        data.lexiconParameters()->loadDawg("target/quackle/demo.dawg");
        if (!data.lexiconParameters()->hasDawg())
            throw std::runtime_error("Demo lexicon not built");
        data.strategyParameters()->initialize("default_english");
        if (!data.strategyParameters()->hasWorths() || !data.strategyParameters()->hasSyn2())
            throw std::runtime_error("Quackle rack-leave strategy data unavailable");
        data.setEvaluator(new Quackle::ScorePlusLeaveEvaluator);
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
        Quackle::GamePosition position(players);
        // Quackle copies pregame positions with no current player until the first increment.
        position.incrementTurn();
        if (!position.setCurrentPlayer(1)) throw std::runtime_error("Cannot set player");
        position.setBoard(board);
        position.setCurrentPlayerRack(Quackle::Rack(alphabet->encode(rackText)), false);
        Quackle::Bag bag;
        bag.clear();
        // Static evaluation needs only bag size, never actual hidden bag contents.
        bag.toss(Quackle::LongLetterString(bagSize, QUACKLE_BLANK_MARK));
        position.setBag(bag);
        position.ensureBoardIsPreparedForAnalysis();
        Quackle::Generator generator(position);
        generator.kibitz(1, bagSize < 7 ? Quackle::Generator::CannotExchange : 0);
        const auto& move = generator.kibitzList().at(0);
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
            std::cout << "PLACE " << move.score << " " << move.equity << " " << placements.size() << "\n";
            for (const auto& p : placements)
                std::cout << p.row << " " << p.col << " " << p.index << " " << p.letter << " " << p.blank << "\n";
        } else if (move.action == Quackle::Move::Exchange) {
            std::vector<int> indexes;
            for (auto letter : move.tiles()) {
                auto decoded = alphabet->userVisible(letter);
                if (decoded.size() != 1) throw std::runtime_error("Invalid exchange tile");
                indexes.push_back(rackIndex(decoded[0]));
            }
            std::cout << "EXCHANGE " << indexes.size();
            for (int index : indexes) std::cout << " " << index;
            std::cout << "\n";
        } else if (move.action == Quackle::Move::Pass) {
            std::cout << "PASS\n";
        } else {
            throw std::runtime_error("Unsupported engine move");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
