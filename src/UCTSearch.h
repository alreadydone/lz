/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Gian-Carlo Pascutto

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

#ifndef UCTSEARCH_H_INCLUDED
#define UCTSEARCH_H_INCLUDED

#include <list>
#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <future>

#include "ThreadPool.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GameState.h"
#include "UCTNode.h"
#include "Network.h"

namespace TimeManagement {
    enum enabled_t {
        AUTO = -1, OFF = 0, ON = 1, FAST = 2, NO_PRUNING = 3
    };
};

struct BackupData {
    struct NodeFactor {
        UCTNode* node;
        float factor;
        NodeFactor(UCTNode* node, float factor) : node(node), factor(factor) {}
    };
    float eval{ -1.0f };
    std::vector<NodeFactor> path;
    int symmetry;
    std::unique_ptr<GameState> state;
    std::atomic<int>* pending_counter;
    //int multiplicity{1};
};

class UCTWorker {
public:
    UCTWorker(UCTSearch * search, int thread_num)
        : m_search(search), m_thread_num(thread_num) {}
    //void operator()();
private:
    UCTSearch * m_search;
    int m_thread_num;
};

class UCTSearch {
public:
    /*
        Depending on rule set and state of the game, we might
        prefer to pass, or we might prefer not to pass unless
        it's the last resort. Same for resigning.
    */
    using passflag_t = int;
    static constexpr passflag_t NORMAL   = 0;
    static constexpr passflag_t NOPASS   = 1 << 0;
    static constexpr passflag_t NORESIGN = 1 << 1;

    /*
        Default memory limit in bytes.
        ~1.6GiB on 32-bits and about 5.2GiB on 64-bits.
    */
    static constexpr size_t DEFAULT_MAX_MEMORY =
        (sizeof(void*) == 4 ? 1'600'000'000 : 5'200'000'000);

    /*
        Minimum allowed size for maximum tree size.
    */
    static constexpr size_t MIN_TREE_SPACE = 100'000'000;

    /*
        Value representing unlimited visits or playouts. Due to
        concurrent updates while multithreading, we need some
        headroom within the native type.
    */
    static constexpr auto UNLIMITED_PLAYOUTS =
        std::numeric_limits<int>::max() / 2;

    UCTSearch(GameState& g, Network & network);
    ~UCTSearch();
    void search(int gnum, int i);
    int think(int color, passflag_t passflag = NORMAL);
    void set_playout_limit(int playouts);
    void set_visit_limit(int visits);
    void ponder();
    bool is_running() const;
    void increment_playouts();
    void play_simulation(std::unique_ptr<GameState> currstate, UCTNode* node, 
        std::atomic<int>* pending_counter, int gnum, int i);
    std::atomic<int> m_positions{0};
    std::atomic<bool> m_run{false};
    std::mutex m_mutex;
    std::condition_variable m_cv;

    void backup(BackupData& bd, Netresult_ptr netresult);
    
    std::atomic<int> pending_netresults{0};
    std::atomic<int> max_pending_netresults;
    std::atomic<int> min_pending_netresults;
private:
    
    friend class UCTWorker;
#ifdef ACCUM_DEBUG
    std::atomic<int> failed_simulations{0};
    std::atomic<uint16_t> max_leaf_vl;
    std::atomic<uint16_t> max_vl;
    std::atomic<int> pending_backups{0};
    std::atomic<int> max_pending_backups;
    std::atomic<int> pending_w_mult{0};
    std::atomic<int> max_pending_w_mult;

    std::string m_debug_string = "";
#endif
    
    std::atomic<uint8_t> m_root_lock{0};
    void acquire_reader();
    void release_reader();
    void acquire_writer();
    void release_writer();
    std::atomic<int>* m_pending_counter{nullptr};
    GameState m_rootstate;
    GameState & m_gtpstate;
    std::unique_ptr<GameState> m_last_rootstate;
    std::unique_ptr<UCTNode> m_root;
    std::atomic<int> m_playouts;
    int m_maxplayouts;
    int m_maxvisits;

    Utils::ThreadGroup m_search_threads;
    std::atomic<bool> m_terminate{false};
    //std::atomic<bool> m_must_run{false};
    std::atomic<bool> m_root_prepared{false};

    //std::list<Utils::ThreadGroup> m_delete_futures;
    Utils::ThreadGroup m_delete_futures;

    void dump_stats(FastState& state, UCTNode& parent);
    void tree_stats(UCTNode& node);
    std::string get_pv(FastState& state, UCTNode& parent);
    void dump_analysis(int playouts);
    bool should_resign(passflag_t passflag, float besteval);
    bool have_alternate_moves(int elapsed_centis, int time_for_move);
    int est_playouts_left(int elapsed_centis, int time_for_move) const;
    size_t prune_noncontenders(int elapsed_centis = 0, int time_for_move = 0,
                               bool prune = true);
    bool stop_thinking(int elapsed_centis = 0, int time_for_move = 0) const;
    int get_best_move(passflag_t passflag);
    void update_root();
    bool advance_to_new_rootstate(std::list<UCTNode*>& to_delete);
    void output_analysis(FastState & state, UCTNode & parent);

    Network & m_network;

    void backup(BackupData& bd, uint16_t vl);
    void failed_simulation(BackupData& bd, uint16_t vl, bool incr = false);
};

#endif
