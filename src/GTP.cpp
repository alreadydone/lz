/* 
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "GTP.h"
#include "FastBoard.h"
#include "FullBoard.h"
#include "GameState.h"
#include "Network.h"
#include "NNCache.h"
#include "SGFTree.h"
#include "SMP.h"
#include "Training.h"
#include "UCTSearch.h"
#include "Utils.h"

using namespace Utils;

// Configuration flags
bool cfg_gtp_mode;
bool cfg_allow_pondering;
int cfg_num_threads;
int cfg_max_playouts;
int cfg_max_visits;
size_t cfg_max_memory;
size_t cfg_max_tree_size;
int cfg_max_cache_ratio_percent;
TimeManagement::enabled_t cfg_timemanage;
int cfg_lagbuffer_cs;
int cfg_resignpct;

bool cfg_dyn_komi;
float cfg_max_wr;
float cfg_min_wr;
float cfg_wr_margin;
float cfg_target_komi;
int cfg_adj_positions;
float cfg_adj_pct;
bool cfg_pos;
bool cfg_neg;
bool cfg_nonslack;
bool cfg_sure_backup;
bool cfg_noshift;
bool cfg_use_symmetries;
bool cfg_orig_policy;
bool cfg_dyn_fpu;
bool cfg_backup_fpu;
bool cfg_collect_during_search;
bool cfg_always_collect;
int cfg_max_num_adjustments;
int cfg_fixed_symmetry;
bool cfg_use_root_for_diff;
bool cfg_auto_pos_neg;
float cfg_max_komi;
float cfg_min_komi;

int cfg_noise;
int cfg_random_cnt;
int cfg_random_min_visits;
float cfg_random_temp;
std::uint64_t cfg_rng_seed;
bool cfg_dumbpass;
#ifdef USE_OPENCL
std::vector<int> cfg_gpus;
bool cfg_sgemm_exhaustive;
bool cfg_tune_only;
int cfg_batch_size;
#ifdef USE_HALF
precision_t cfg_precision;
#endif
#endif
float cfg_puct;
float cfg_softmax_temp;
float cfg_fpu_reduction;
float cfg_fpu_root_reduction;
std::string cfg_weightsfile;
std::string cfg_logfile;
FILE* cfg_logfile_handle;
bool cfg_quiet;
std::string cfg_options_str;
bool cfg_benchmark;
bool cfg_cpu_only;
float cfg_virtual_loss;
float cfg_logbase;
int cfg_analyze_interval_centis;

std::unique_ptr<Network> GTP::s_network;

void GTP::initialize(std::unique_ptr<Network>&& net) {
    s_network = std::move(net);

    bool result;
    std::string message;
    std::tie(result, message) =
        set_max_memory(cfg_max_memory, cfg_max_cache_ratio_percent);
    if (!result) {
        // This should only ever happen with 60 block networks on 32bit machine.
        myprintf("LOW MEMORY SETTINGS! Couldn't set default memory limits.\n");
        myprintf("The network you are using might be too big\n");
        myprintf("for the default settings on your system.\n");
        throw std::runtime_error("Error setting memory requirements.");
    }
    myprintf(message.c_str());
    myprintf("\n");
}

void GTP::setup_default_parameters() {
    cfg_gtp_mode = false;
    cfg_allow_pondering = true;
    // we will re-calculate this on Leela.cpp
    cfg_num_threads = 0;

    cfg_max_memory = UCTSearch::DEFAULT_MAX_MEMORY;
    cfg_max_playouts = UCTSearch::UNLIMITED_PLAYOUTS;
    cfg_max_visits = UCTSearch::UNLIMITED_PLAYOUTS;
    // This will be overwriiten in initialize() after network size is known.
    cfg_max_tree_size = UCTSearch::DEFAULT_MAX_MEMORY;
    cfg_max_cache_ratio_percent = 10;
    cfg_timemanage = TimeManagement::AUTO;
    cfg_lagbuffer_cs = 100;
    cfg_weightsfile = leelaz_file("best-network");
#ifdef USE_OPENCL
    cfg_gpus = { };
    cfg_sgemm_exhaustive = false;
    cfg_tune_only = false;

    // we will re-calculate this on Leela.cpp
    cfg_batch_size = 0;
#ifdef USE_HALF
    cfg_precision = precision_t::AUTO;
#endif
#endif
    cfg_puct = 0.8f;
    cfg_softmax_temp = 1.0f;
    cfg_fpu_reduction = 0.25f;
    // see UCTSearch::should_resign
    cfg_resignpct = -1;

    cfg_dyn_komi = false;
    cfg_target_komi = 7.5f;
    cfg_adj_positions = 200;
    cfg_adj_pct = 4.0;
    cfg_pos = false;
    cfg_neg = false;
    cfg_nonslack = false;
    cfg_sure_backup = true;
    cfg_noshift = true;
    cfg_use_symmetries = true;
    cfg_orig_policy = true;
    cfg_dyn_fpu = false;
    cfg_backup_fpu = false; // to remove
    cfg_use_root_for_diff = false;
    cfg_auto_pos_neg = true;
    cfg_max_komi = std::numeric_limits<float>::max();
    cfg_min_komi = -cfg_max_komi;

    cfg_noise = false;
    cfg_fpu_root_reduction = cfg_fpu_reduction;
    cfg_random_cnt = 0;
    cfg_random_min_visits = 1;
    cfg_random_temp = 1.0f;
    cfg_dumbpass = false;
    cfg_logfile_handle = nullptr;
    cfg_quiet = false;
    cfg_benchmark = false;
#ifdef USE_CPU_ONLY
    cfg_cpu_only = true;
#else
    cfg_cpu_only = false;
#endif
    cfg_virtual_loss = 3.0f;
    cfg_logbase = 0.0f;

    cfg_analyze_interval_centis = 0;

    // C++11 doesn't guarantee *anything* about how random this is,
    // and in MinGW it isn't random at all. But we can mix it in, which
    // helps when it *is* high quality (Linux, MSVC).
    std::random_device rd;
    std::ranlux48 gen(rd());
    std::uint64_t seed1 = (gen() << 16) ^ gen();
    // If the above fails, this is one of our best, portable, bets.
    std::uint64_t seed2 = std::chrono::high_resolution_clock::
        now().time_since_epoch().count();
    cfg_rng_seed = seed1 ^ seed2;
}

int dyn_komi_test(Network & net, GameState &game, int sym) {
    // todo: configurable lower/upper limits and gap, allow black or white to move, more accurate (with raw_winrate, no bias towards pos or neg)
    auto vec = net.get_output(&game, Network::Ensemble::DIRECT, sym, true);
    auto current_komi = game.m_stm_komi;
    std::vector<float> loc_incr;
    game.m_stm_komi = -300.5f;
    auto vec_old = net.get_output(&game, Network::Ensemble::DIRECT, sym, true);
    auto accum_neg = 1.0f - vec_old.winrate;
    myprintf("komi | winrate\n");
    myprintf("---- | ----\n");
    for (auto s = -300.0f; s <= 0.0f; s = s + 0.5) {
        game.m_stm_komi = s;
        vec = net.get_output(&game, Network::Ensemble::DIRECT, sym, true);
        myprintf("%f | %f\n", s, vec.winrate);
        if (vec_old.winrate < vec.winrate) {
            loc_incr.emplace_back(s);
            accum_neg += vec.winrate - vec_old.winrate;
        }
        vec_old = vec;
    }
    auto accum_pos = 0.0f;
    for (auto s = 0.5; s <= 300.0f; s = s + 0.5) {
        game.m_stm_komi = s;
        vec = net.get_output(&game, Network::Ensemble::DIRECT, sym, true);
        myprintf("%f | %f\n", s, vec.winrate);
        if (vec_old.winrate < vec.winrate) {
            loc_incr.emplace_back(s);
            accum_pos += vec.winrate - vec_old.winrate;
        }
        vec_old = vec;
    }
    accum_pos += vec.winrate;
    game.m_stm_komi = current_komi;
    //if (loc_incr.empty()) { myprintf("Perfect weight file! 完美的权重！\n"); }
    myprintf("在以下贴目值附近胜率是上升的：Winrate increasing near ");
    for (float s : loc_incr) { myprintf("%4.1f, ", s); }
    myprintf(".\n");
    myprintf("Negative komi total score: %e\n", accum_neg);
    myprintf("Positive komi total score: %e\n", accum_pos);
    const auto thres = 0.05f;
    if (accum_neg <= thres && accum_pos <= thres) {
        myprintf("Weight file is of good quality for dynamic komi! 权重质量不错，可用于让子／不退让版。\n");
        return 0;
    }
    else if (accum_neg > thres && accum_pos > thres) {
        myprintf("Weight file is unusable for dynamic komi. Sorry. 权重质量不佳，不能用于让子／不退让版。\n");
        return 1;
    }
    else if (accum_neg <= thres) {
        myprintf("Weight file is of mediocre quality for dynamic komi. Use with the option --neg. 权重质量中等，正贴目表现不佳，推荐使用--neg参数。\n");
        return 2;
    }
    else {
        myprintf("Weight file is of mediocre quality for dynamic komi. Use with the option --pos. 权重质量中等，负贴目表现不佳，推荐使用--pos参数。\n");
        return 3;
    }
}

const std::string GTP::s_commands[] = {
    "protocol_version",
    "name",
    "version",
    "quit",
    "known_command",
    "list_commands",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "showboard",
    "undo",
    "final_score",
    "final_status_list",
    "time_settings",
    "time_left",
    "fixed_handicap",
    "place_free_handicap",
    "set_free_handicap",
    "loadsgf",
    "printsgf",
    "kgs-genmove_cleanup",
    "kgs-time_settings",
    "kgs-game_over",
    "heatmap",
    "dyn_komi_test",
    "lz-analyze",
    "lz-genmove_analyze",
    "lz-memory_report",
    "lz-setoption",
    ""
};

// Default/min/max could be moved into separate fields,
// but for now we assume that the GUI will not send us invalid info.
const std::string GTP::s_options[] = {
    "option name Maximum Memory Use (MiB) type spin default 2048 min 128 max 131072",
    "option name Percentage of memory for cache type spin default 10 min 1 max 99",
    "option name Visits type spin default 0 min 0 max 1000000000",
    "option name Playouts type spin default 0 min 0 max 1000000000",
    "option name Lagbuffer type spin default 0 min 0 max 3000",
    "option name Resign Percentage type spin default -1 min -1 max 30",
    "option name Pondering type check default true",
    ""
};

std::string GTP::get_life_list(const GameState & game, bool live) {
    std::vector<std::string> stringlist;
    std::string result;
    const auto& board = game.board;

    if (live) {
        for (int i = 0; i < board.get_boardsize(); i++) {
            for (int j = 0; j < board.get_boardsize(); j++) {
                int vertex = board.get_vertex(i, j);

                if (board.get_state(vertex) != FastBoard::EMPTY) {
                    stringlist.push_back(board.get_string(vertex));
                }
            }
        }
    }

    // remove multiple mentions of the same string
    // unique reorders and returns new iterator, erase actually deletes
    std::sort(begin(stringlist), end(stringlist));
    stringlist.erase(std::unique(begin(stringlist), end(stringlist)),
                     end(stringlist));

    for (size_t i = 0; i < stringlist.size(); i++) {
        result += (i == 0 ? "" : "\n") + stringlist[i];
    }

    return result;
}

void GTP::execute(GameState & game, const std::string& xinput) {
    std::string input;
    static auto search = std::make_unique<UCTSearch>(game, *s_network);

    bool transform_lowercase = true;

    // Required on Unixy systems
    if (xinput.find("loadsgf") != std::string::npos) {
        transform_lowercase = false;
    }

    /* eat empty lines, simple preprocessing, lower case */
    for (unsigned int tmp = 0; tmp < xinput.size(); tmp++) {
        if (xinput[tmp] == 9) {
            input += " ";
        } else if ((xinput[tmp] > 0 && xinput[tmp] <= 9)
                || (xinput[tmp] >= 11 && xinput[tmp] <= 31)
                || xinput[tmp] == 127) {
               continue;
        } else {
            if (transform_lowercase) {
                input += std::tolower(xinput[tmp]);
            } else {
                input += xinput[tmp];
            }
        }

        // eat multi whitespace
        if (input.size() > 1) {
            if (std::isspace(input[input.size() - 2]) &&
                std::isspace(input[input.size() - 1])) {
                input.resize(input.size() - 1);
            }
        }
    }

    std::string command;
    int id = -1;

    if (input == "") {
        return;
    } else if (input == "exit") {
        exit(EXIT_SUCCESS);
    } else if (input.find("#") == 0) {
        return;
    } else if (std::isdigit(input[0])) {
        std::istringstream strm(input);
        char spacer;
        strm >> id;
        strm >> std::noskipws >> spacer;
        std::getline(strm, command);
    } else {
        command = input;
    }

    /* process commands */
    if (command == "protocol_version") {
        gtp_printf(id, "%d", GTP_VERSION);
        return;
    } else if (command == "name") {
        gtp_printf(id, PROGRAM_NAME);
        return;
    } else if (command == "version") {
        gtp_printf(id, PROGRAM_VERSION);
        return;
    } else if (command == "quit") {
        gtp_printf(id, "");
        exit(EXIT_SUCCESS);
    } else if (command.find("known_command") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;     /* remove known_command */
        cmdstream >> tmp;

        for (int i = 0; s_commands[i].size() > 0; i++) {
            if (tmp == s_commands[i]) {
                gtp_printf(id, "true");
                return;
            }
        }

        gtp_printf(id, "false");
        return;
    } else if (command.find("list_commands") == 0) {
        std::string outtmp(s_commands[0]);
        for (int i = 1; s_commands[i].size() > 0; i++) {
            outtmp = outtmp + "\n" + s_commands[i];
        }
        gtp_printf(id, outtmp.c_str());
        return;
    } else if (command.find("boardsize") == 0) {
        std::istringstream cmdstream(command);
        std::string stmp;
        int tmp;

        cmdstream >> stmp;  // eat boardsize
        cmdstream >> tmp;

        if (!cmdstream.fail()) {
            if (tmp != BOARD_SIZE) {
                gtp_fail_printf(id, "unacceptable size");
            } else {
                float old_komi = game.get_komi();
                Training::clear_training();
                game.init_game(tmp, old_komi);
                gtp_printf(id, "");
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("clear_board") == 0) {
        Training::clear_training();
        game.reset_game();
        search = std::make_unique<UCTSearch>(game, *s_network);
        assert(UCTNodePointer::get_tree_size() == 0);
        gtp_printf(id, "");
        return;
    } else if (command.find("komi") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        float komi = cfg_target_komi;
        float old_komi = game.get_komi();

        cmdstream >> tmp;  // eat komi
        cmdstream >> komi;

        if (!cmdstream.fail()) {
            if (komi != old_komi) {
                game.set_komi(komi);
            }
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("play") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string color, vertex;

        cmdstream >> tmp;   //eat play
        cmdstream >> color;
        cmdstream >> vertex;

        if (!cmdstream.fail()) {
            if (!game.play_textmove(color, vertex)) {
                gtp_fail_printf(id, "illegal move");
            } else {
                gtp_printf(id, "");
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("genmove") == 0
               || command.find("lz-genmove_analyze") == 0) {
        auto analysis_output = command.find("lz-genmove_analyze") == 0;
        auto interval = 0;

        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;  // eat genmove
        cmdstream >> tmp;
        if (analysis_output) {
            cmdstream >> interval;
        }

        if (!cmdstream.fail()) {
            int who;
            if (tmp == "w" || tmp == "white") {
                who = FastBoard::WHITE;
            } else if (tmp == "b" || tmp == "black") {
                who = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "syntax error");
                return;
            }
            if (analysis_output) {
                // Start of multi-line response
                cfg_analyze_interval_centis = interval;
                if (id != -1) gtp_printf_raw("=%d\n", id);
                else gtp_printf_raw("=\n");
            }
            // start thinking
            {
                game.set_to_move(who);
                // Outputs winrate and pvs for lz-genmove_analyze
                int move = search->think(who);
                game.play_move(move);

                std::string vertex = game.move_to_text(move);
                if (!analysis_output) {
                    gtp_printf(id, "%s", vertex.c_str());
                } else {
                    gtp_printf_raw("play %s\n", vertex.c_str());
                }
            }
            if (cfg_allow_pondering) {
                // now start pondering
                if (!game.has_resigned()) {
                    // Outputs winrate and pvs through gtp for lz-genmove_analyze
                    search->ponder(false);
                }
            }
            if (analysis_output) {
                // Terminate multi-line response
                gtp_printf_raw("\n");
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        analysis_output = false;
        return;
    } else if (command.find("lz-analyze") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        auto who = game.board.get_to_move();

        cmdstream >> tmp; // eat lz-analyze
        cmdstream >> tmp; // eat side to move or interval
        if (!cmdstream.fail()) {
            if (tmp == "w" || tmp == "white") {
                who = FastBoard::WHITE;
            } else if (tmp == "b" || tmp == "black") {
                who = FastBoard::BLACK;
            } else {
                // Not side to move, must be interval
                try {
                    cfg_analyze_interval_centis = std::stoi(tmp);
                } catch(...) {
                    gtp_fail_printf(id, "syntax not understood");
                    return;
                }
            }
            if (tmp == "w" || tmp == "b" || tmp == "white" || tmp == "black") {
                // We got a color, so the interval must come now.
                int interval;
                cmdstream >> interval;
                if (!cmdstream.fail()) {
                    cfg_analyze_interval_centis = interval;
                } else {
                    gtp_fail_printf(id, "syntax not understood");
                    return;
                }
            }
        }
        // Start multi-line response.
        if (id != -1) gtp_printf_raw("=%d\n", id);
        else gtp_printf_raw("=\n");
        // Now start pondering.
        if (!game.has_resigned()) {
            // Outputs winrate and pvs through gtp
            game.set_to_move(who);
            search->ponder(true);
        }
        cfg_analyze_interval_centis = 0;
        // Terminate multi-line response
        gtp_printf_raw("\n");
        return;
    } else if (command.find("kgs-genmove_cleanup") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;  // eat kgs-genmove
        cmdstream >> tmp;

        if (!cmdstream.fail()) {
            int who;
            if (tmp == "w" || tmp == "white") {
                who = FastBoard::WHITE;
            } else if (tmp == "b" || tmp == "black") {
                who = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "syntax error");
                return;
            }
            game.set_passes(0);
            {
                game.set_to_move(who);
                int move = search->think(who, UCTSearch::NOPASS);
                game.play_move(move);

                std::string vertex = game.move_to_text(move);
                gtp_printf(id, "%s", vertex.c_str());
            }
            if (cfg_allow_pondering) {
                // now start pondering
                if (!game.has_resigned()) {
                    search->ponder(false);
                }
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("undo") == 0) {
        if (game.undo_move()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "cannot undo");
        }
        return;
    } else if (command.find("showboard") == 0) {
        gtp_printf(id, "");
        game.display_state();
        return;
    } else if (command.find("final_score") == 0) {
        float ftmp = game.final_score();
        /* white wins */
        if (ftmp < -0.1) {
            gtp_printf(id, "W+%3.1f", float(fabs(ftmp)));
        } else if (ftmp > 0.1) {
            gtp_printf(id, "B+%3.1f", ftmp);
        } else {
            gtp_printf(id, "0");
        }
        return;
    } else if (command.find("final_status_list") == 0) {
        if (command.find("alive") != std::string::npos) {
            std::string livelist = get_life_list(game, true);
            gtp_printf(id, livelist.c_str());
        } else if (command.find("dead") != std::string::npos) {
            std::string deadlist = get_life_list(game, false);
            gtp_printf(id, deadlist.c_str());
        } else {
            gtp_printf(id, "");
        }
        return;
    } else if (command.find("time_settings") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int maintime, byotime, byostones;

        cmdstream >> tmp >> maintime >> byotime >> byostones;

        if (!cmdstream.fail()) {
            // convert to centiseconds and set
            game.set_timecontrol(maintime * 100, byotime * 100, byostones, 0);

            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("time_left") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, color;
        int time, stones;

        cmdstream >> tmp >> color >> time >> stones;

        if (!cmdstream.fail()) {
            int icolor;

            if (color == "w" || color == "white") {
                icolor = FastBoard::WHITE;
            } else if (color == "b" || color == "black") {
                icolor = FastBoard::BLACK;
            } else {
                gtp_fail_printf(id, "Color in time adjust not understood.\n");
                return;
            }

            game.adjust_time(icolor, time * 100, stones);

            gtp_printf(id, "");

            if (cfg_allow_pondering) {
                // KGS sends this after our move
                // now start pondering
                if (!game.has_resigned()) {
                    search->ponder(false);
                }
            }
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("auto") == 0) {
        do {
            int move = search->think(game.get_to_move(), UCTSearch::NORMAL);
            game.play_move(move);
            game.display_state();

        } while (game.get_passes() < 2 && !game.has_resigned());

        return;
    } else if (command.find("go") == 0) {
        int move = search->think(game.get_to_move());
        game.play_move(move);

        std::string vertex = game.move_to_text(move);
        myprintf("%s\n", vertex.c_str());
        return;
    } else if (command.find("heatmap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string symmetry;

        cmdstream >> tmp;   // eat heatmap
        cmdstream >> symmetry;

        Network::Netresult vec;
        if (cmdstream.fail()) {
            // Default = DIRECT with no symmetric change
            vec = s_network->get_output(
                &game, Network::Ensemble::DIRECT,
                Network::IDENTITY_SYMMETRY, true);
        } else if (symmetry == "all") {
            for (auto s = 0; s < Network::NUM_SYMMETRIES; ++s) {
                vec = s_network->get_output(
                    &game, Network::Ensemble::DIRECT, s, true);
                Network::show_heatmap(&game, vec, false);
            }
        } else if (symmetry == "average" || symmetry == "avg") {
            vec = s_network->get_output(
                &game, Network::Ensemble::AVERAGE,
                Network::NUM_SYMMETRIES, true);
        } else {
            vec = s_network->get_output(
                &game, Network::Ensemble::DIRECT, std::stoi(symmetry), true);
        }

        if (symmetry != "all") {
            Network::show_heatmap(&game, vec, false);
        }

        gtp_printf(id, "");
        return;
    }
    else if (command.find("dyn_komi_test") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string symmetry;

        cmdstream >> tmp;   // eat dyn_komi_test
        cmdstream >> symmetry;

        int sym;
        if (cmdstream.fail()) {
            sym = Network::IDENTITY_SYMMETRY;
        }
        else {
            sym = std::stoi(symmetry);
        }
        dyn_komi_test(*s_network, game, sym);
    } else if (command.find("fixed_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int stones;

        cmdstream >> tmp;   // eat fixed_handicap
        cmdstream >> stones;

        if (!cmdstream.fail() && game.set_fixed_handicap(stones)) {
            auto stonestring = game.board.get_stone_list();
            gtp_printf(id, "%s", stonestring.c_str());
        } else {
            gtp_fail_printf(id, "Not a valid number of handicap stones");
        }
        return;
    } else if (command.find("place_free_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int stones;

        cmdstream >> tmp;   // eat place_free_handicap
        cmdstream >> stones;

        if (!cmdstream.fail()) {
            game.place_free_handicap(stones, *s_network);
            auto stonestring = game.board.get_stone_list();
            gtp_printf(id, "%s", stonestring.c_str());
        } else {
            gtp_fail_printf(id, "Not a valid number of handicap stones");
        }

        return;
    } else if (command.find("set_free_handicap") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp;   // eat set_free_handicap

        do {
            std::string vertex;

            cmdstream >> vertex;

            if (!cmdstream.fail()) {
                if (!game.play_textmove("black", vertex)) {
                    gtp_fail_printf(id, "illegal move");
                } else {
                    game.set_handicap(game.get_handicap() + 1);
                }
            }
        } while (!cmdstream.fail());

        std::string stonestring = game.board.get_stone_list();
        gtp_printf(id, "%s", stonestring.c_str());

        return;
    } else if (command.find("loadsgf") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;
        int movenum;

        cmdstream >> tmp;   // eat loadsgf
        cmdstream >> filename;

        if (!cmdstream.fail()) {
            cmdstream >> movenum;

            if (cmdstream.fail()) {
                movenum = 999;
            }
        } else {
            gtp_fail_printf(id, "Missing filename.");
            return;
        }

        auto sgftree = std::make_unique<SGFTree>();

        try {
            sgftree->load_from_file(filename);
            game = sgftree->follow_mainline_state(movenum - 1);
            gtp_printf(id, "");
        } catch (const std::exception&) {
            gtp_fail_printf(id, "cannot load file");
        }
        return;
    } else if (command.find("kgs-chat") == 0) {
        // kgs-chat (game|private) Name Message
        std::istringstream cmdstream(command);
        std::string tmp;

        cmdstream >> tmp; // eat kgs-chat
        cmdstream >> tmp; // eat game|private
        cmdstream >> tmp; // eat player name
        do {
            cmdstream >> tmp; // eat message
        } while (!cmdstream.fail());

        gtp_fail_printf(id, "I'm a go bot, not a chat bot.");
        return;
    } else if (command.find("kgs-game_over") == 0) {
        // Do nothing. Particularly, don't ponder.
        gtp_printf(id, "");
        return;
    } else if (command.find("kgs-time_settings") == 0) {
        // none, absolute, byoyomi, or canadian
        std::istringstream cmdstream(command);
        std::string tmp;
        std::string tc_type;
        int maintime, byotime, byostones, byoperiods;

        cmdstream >> tmp >> tc_type;

        if (tc_type.find("none") != std::string::npos) {
            // 30 mins
            game.set_timecontrol(30 * 60 * 100, 0, 0, 0);
        } else if (tc_type.find("absolute") != std::string::npos) {
            cmdstream >> maintime;
            game.set_timecontrol(maintime * 100, 0, 0, 0);
        } else if (tc_type.find("canadian") != std::string::npos) {
            cmdstream >> maintime >> byotime >> byostones;
            // convert to centiseconds and set
            game.set_timecontrol(maintime * 100, byotime * 100, byostones, 0);
        } else if (tc_type.find("byoyomi") != std::string::npos) {
            // KGS style Fischer clock
            cmdstream >> maintime >> byotime >> byoperiods;
            game.set_timecontrol(maintime * 100, byotime * 100, 0, byoperiods);
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return;
        }

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("netbench") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp;
        int iterations;

        cmdstream >> tmp;  // eat netbench
        cmdstream >> iterations;

        if (!cmdstream.fail()) {
            s_network->benchmark(&game, iterations);
        } else {
            s_network->benchmark(&game);
        }
        gtp_printf(id, "");
        return;

    } else if (command.find("printsgf") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        cmdstream >> tmp;   // eat printsgf
        cmdstream >> filename;

        auto sgf_text = SGFTree::state_to_string(game, 0);
        // GTP says consecutive newlines terminate the output,
        // so we must filter those.
        boost::replace_all(sgf_text, "\n\n", "\n");

        if (cmdstream.fail()) {
            gtp_printf(id, "%s\n", sgf_text.c_str());
        } else {
            std::ofstream out(filename);
            out << sgf_text;
            out.close();
            gtp_printf(id, "");
        }

        return;
    } else if (command.find("load_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        // tmp will eat "load_training"
        cmdstream >> tmp >> filename;

        Training::load_training(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("save_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        // tmp will eat "save_training"
        cmdstream >> tmp >>  filename;

        Training::save_training(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("dump_training") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, winner_color, filename;
        int who_won;

        // tmp will eat "dump_training"
        cmdstream >> tmp >> winner_color >> filename;

        if (winner_color == "w" || winner_color == "white") {
            who_won = FullBoard::WHITE;
        } else if (winner_color == "b" || winner_color == "black") {
            who_won = FullBoard::BLACK;
        } else {
            gtp_fail_printf(id, "syntax not understood");
            return;
        }

        Training::dump_training(who_won, filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("dump_debug") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, filename;

        // tmp will eat "dump_debug"
        cmdstream >> tmp >> filename;

        Training::dump_debug(filename);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }

        return;
    } else if (command.find("dump_supervised") == 0) {
        std::istringstream cmdstream(command);
        std::string tmp, sgfname, outname;

        // tmp will eat dump_supervised
        cmdstream >> tmp >> sgfname >> outname;

        Training::dump_supervised(sgfname, outname);

        if (!cmdstream.fail()) {
            gtp_printf(id, "");
        } else {
            gtp_fail_printf(id, "syntax not understood");
        }
        return;
    } else if (command.find("lz-memory_report") == 0) {
        auto base_memory = get_base_memory();
        auto tree_size = add_overhead(UCTNodePointer::get_tree_size());
        auto cache_size = add_overhead(s_network->get_estimated_cache_size());

        auto total = base_memory + tree_size + cache_size;
        gtp_printf(id,
            "Estimated total memory consumption: %d MiB.\n"
            "Network with overhead: %d MiB / Search tree: %d MiB / Network cache: %d\n",
            total / MiB, base_memory / MiB, tree_size / MiB, cache_size / MiB);
        return;
    } else if (command.find("lz-setoption") == 0) {
        return execute_setoption(*search.get(), id, command);
    }
    gtp_fail_printf(id, "unknown command");
    return;
}

std::pair<std::string, std::string> GTP::parse_option(std::istringstream& is) {
    std::string token, name, value;

    // Read option name (can contain spaces)
    while (is >> token && token != "value")
        name += std::string(" ", name.empty() ? 0 : 1) + token;

    // Read option value (can contain spaces)
    while (is >> token)
        value += std::string(" ", value.empty() ? 0 : 1) + token;

    return std::make_pair(name, value);
}

size_t GTP::get_base_memory() {
    // At the moment of writing the memory consumption is
    // roughly network size + 85 for one GPU and + 160 for two GPUs.
#ifdef USE_OPENCL
    auto gpus = std::max(cfg_gpus.size(), size_t{1});
    return s_network->get_estimated_size() + 85 * MiB * gpus;
#else
    return s_network->get_estimated_size();
#endif
}

std::pair<bool, std::string> GTP::set_max_memory(size_t max_memory,
    int cache_size_ratio_percent) {
    if (max_memory == 0) {
        max_memory = UCTSearch::DEFAULT_MAX_MEMORY;
    }

    // Calculate amount of memory available for the search tree +
    // NNCache by estimating a constant memory overhead first.
    auto base_memory = get_base_memory();

    if (max_memory < base_memory) {
        return std::make_pair(false, "Not enough memory for network. " +
            std::to_string(base_memory / MiB) + " MiB required.");
    }

    auto max_memory_for_search = max_memory - base_memory;

    assert(cache_size_ratio_percent >= 1);
    assert(cache_size_ratio_percent <= 99);
    auto max_cache_size = max_memory_for_search *
        cache_size_ratio_percent / 100;

    auto max_cache_count =
        (int)(remove_overhead(max_cache_size) / NNCache::ENTRY_SIZE);

    // Verify if the setting would not result in too little cache.
    if (max_cache_count < NNCache::MIN_CACHE_COUNT) {
        return std::make_pair(false, "Not enough memory for cache.");
    }
    auto max_tree_size = max_memory_for_search - max_cache_size;

    if (max_tree_size < UCTSearch::MIN_TREE_SPACE) {
        return std::make_pair(false, "Not enough memory for search tree.");
    }

    // Only if settings are ok we store the values in config.
    cfg_max_memory = max_memory;
    cfg_max_cache_ratio_percent = cache_size_ratio_percent;
    // Set max_tree_size.
    cfg_max_tree_size = remove_overhead(max_tree_size);
    // Resize cache.
    s_network->nncache_resize(max_cache_count);

    return std::make_pair(true, "Setting max tree size to " +
        std::to_string(max_tree_size / MiB) + " MiB and cache size to " +
        std::to_string(max_cache_size / MiB) +
        " MiB.");
}

void GTP::execute_setoption(UCTSearch & search,
                            int id, const std::string &command) {
    std::istringstream cmdstream(command);
    std::string tmp, name_token;

    // Consume lz_setoption, name.
    cmdstream >> tmp >> name_token;

    // Print available options if called without an argument.
    if (cmdstream.fail()) {
        std::string options_out_tmp("");
        for (int i = 0; s_options[i].size() > 0; i++) {
            options_out_tmp = options_out_tmp + "\n" + s_options[i];
        }
        gtp_printf(id, options_out_tmp.c_str());
        return;
    }

    if (name_token.find("name") != 0) {
        gtp_fail_printf(id, "incorrect syntax for lz-setoption");
        return;
    }

    std::string name, value;
    std::tie(name, value) = parse_option(cmdstream);

    if (name == "maximum memory use (mib)") {
        std::istringstream valuestream(value);
        int max_memory_in_mib;
        valuestream >> max_memory_in_mib;
        if (!valuestream.fail()) {
            if (max_memory_in_mib < 128 || max_memory_in_mib > 131072) {
                gtp_fail_printf(id, "incorrect value");
                return;
            }
            bool result;
            std::string reason;
            std::tie(result, reason) = set_max_memory(max_memory_in_mib * MiB,
                cfg_max_cache_ratio_percent);
            if (result) {
                gtp_printf(id, reason.c_str());
            } else {
                gtp_fail_printf(id, reason.c_str());
            }
            return;
        } else {
            gtp_fail_printf(id, "incorrect value");
            return;
        }
    } else if (name == "percentage of memory for cache") {
        std::istringstream valuestream(value);
        int cache_size_ratio_percent;
        valuestream >> cache_size_ratio_percent;
        if (cache_size_ratio_percent < 1 || cache_size_ratio_percent > 99) {
            gtp_fail_printf(id, "incorrect value");
            return;
        }
        bool result;
        std::string reason;
        std::tie(result, reason) = set_max_memory(cfg_max_memory,
            cache_size_ratio_percent);
        if (result) {
            gtp_printf(id, reason.c_str());
        } else {
            gtp_fail_printf(id, reason.c_str());
        }
        return;
    } else if (name == "visits") {
        std::istringstream valuestream(value);
        int visits;
        valuestream >> visits;
        cfg_max_visits = visits;

        // 0 may be specified to mean "no limit"
        if (cfg_max_visits == 0) {
            cfg_max_visits = UCTSearch::UNLIMITED_PLAYOUTS;
        }
        // Note that if the visits are changed but no
        // explicit command to set memory usage is given,
        // we will stick with the initial guess we made on startup.
        search.set_visit_limit(cfg_max_visits);

        gtp_printf(id, "");
    } else if (name == "playouts") {
        std::istringstream valuestream(value);
        int playouts;
        valuestream >> playouts;
        cfg_max_playouts = playouts;

        // 0 may be specified to mean "no limit"
        if (cfg_max_playouts == 0) {
            cfg_max_playouts = UCTSearch::UNLIMITED_PLAYOUTS;
        } else if (cfg_allow_pondering) {
            // Limiting playouts while pondering is still enabled
            // makes no sense.
            gtp_fail_printf(id, "incorrect value");
            return;
        }

        // Note that if the playouts are changed but no
        // explicit command to set memory usage is given,
        // we will stick with the initial guess we made on startup.
        search.set_playout_limit(cfg_max_visits);

        gtp_printf(id, "");
    } else if (name == "lagbuffer") {
        std::istringstream valuestream(value);
        int lagbuffer;
        valuestream >> lagbuffer;
        cfg_lagbuffer_cs = lagbuffer;
        gtp_printf(id, "");
    } else if (name == "pondering") {
        std::istringstream valuestream(value);
        std::string toggle;
        valuestream >> toggle;
        if (toggle == "true") {
            if (cfg_max_playouts != UCTSearch::UNLIMITED_PLAYOUTS) {
                gtp_fail_printf(id, "incorrect value");
                return;
            }
            cfg_allow_pondering = true;
        } else if (toggle == "false") {
            cfg_allow_pondering = false;
        } else {
            gtp_fail_printf(id, "incorrect value");
            return;
        }
        gtp_printf(id, "");
    } else if (name == "resign percentage") {
        std::istringstream valuestream(value);
        int resignpct;
        valuestream >> resignpct;
        cfg_resignpct = resignpct;
        gtp_printf(id, "");
    } else {
        gtp_fail_printf(id, "Unknown option");
    }
    return;
}
