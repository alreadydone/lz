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

#include "config.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Random.h"
#include "Utils.h"

using namespace Utils;

UCTNode::UCTNode(int vertex, float policy) : m_move(vertex), m_policy(policy) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0.0;
}

std::array<std::array<int, NUM_INTERSECTIONS>,
    Network::NUM_SYMMETRIES> Network::symmetry_nn_idx_table;

void UCTNode::create_children(Network::Netresult& raw_netlist0,
                              int symmetry,
                              const GameState& state, 
                              float min_psa_ratio) {

    Network::Netresult raw_netlist;
    m_net_eval = raw_netlist.winrate = raw_netlist0.winrate;
    const auto to_move = state.board.get_to_move();
    // our search functions evaluate from black's point of view
    if (state.board.white_to_move()) {
        m_net_eval = 1.0f - m_net_eval;
    }

    if (!expandable(min_psa_ratio)) {
        acquire_writer();
        return;
    }

    for (auto idx = size_t{ 0 }; idx < NUM_INTERSECTIONS; ++idx) {
        const auto sym_idx = Network::symmetry_nn_idx_table[symmetry][idx];
        raw_netlist.policy[idx] = raw_netlist0.policy[sym_idx];
    }
    raw_netlist.policy_pass = raw_netlist0.policy_pass;

    std::vector<Network::PolicyVertexPair> nodelist;

    auto legal_sum = 0.0f;
    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state.board.get_vertex(x, y);
        if (state.is_move_legal(to_move, vertex)) {
            nodelist.emplace_back(raw_netlist.policy[i], vertex);
            legal_sum += raw_netlist.policy[i];
        }
    }
    nodelist.emplace_back(raw_netlist.policy_pass, FastBoard::PASS);
    legal_sum += raw_netlist.policy_pass;

    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    }
    else {
        // This can happen with new randomized nets.
        auto uniform_prob = 1.0f / nodelist.size();
        for (auto& node : nodelist) {
            node.first = uniform_prob;
        }
    }

    link_nodelist(nodelist, min_psa_ratio);
    return;
}

void UCTNode::link_nodelist(std::vector<Network::PolicyVertexPair>& nodelist,
                            float min_psa_ratio) {
    assert(min_psa_ratio < m_min_psa_ratio_children);

    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));
    std::vector<bool> to_append(nodelist.size());

    //for ()

    const auto max_psa = nodelist[0].first;
    const auto old_min_psa = max_psa * m_min_psa_ratio_children;
    const auto new_min_psa = max_psa * min_psa_ratio;

    acquire_writer();
    if (new_min_psa > 0.0f) {
        m_children.reserve(
            std::count_if(cbegin(nodelist), cend(nodelist),
                [=](const auto& node) { return node.first >= new_min_psa; }
            )
        );
    } else {
        m_children.reserve(nodelist.size());
    }

    auto skipped_children = false;
    for (const auto& node : nodelist) {
        if (node.first < new_min_psa) {
            skipped_children = true;
            break;
        } else if (node.first < old_min_psa) {
            m_children.emplace_back(node.second, node.first);
        }
    }

    m_min_psa_ratio_children = skipped_children ? min_psa_ratio : 0.0f;
}

const std::vector<UCTNodePointer>& UCTNode::get_children() const {
    return m_children;
}


int UCTNode::get_move() const {
    return m_move;
}

void UCTNode::virtual_loss(uint16_t vl) {
    m_virtual_loss += vl;
}

void UCTNode::virtual_loss_undo(uint16_t vl) {
    if (vl != 0) { m_virtual_loss -= vl; }
}

void UCTNode::update(float eval, uint16_t vl, float factor, float sel_factor) {
    atomic_add(m_visits, double(factor));
    atomic_add(m_blackevals, double(eval*factor));
    //atomic_add(m_sel_visits, double(sel_factor));
    virtual_loss_undo(vl);
}

bool UCTNode::has_children() const {
    return m_min_psa_ratio_children <= 1.0f;
}

bool UCTNode::expandable(const float min_psa_ratio) const {
    return min_psa_ratio < m_min_psa_ratio_children;
}

float UCTNode::get_policy() const {
    return m_policy;
}

void UCTNode::set_policy(float policy) {
    m_policy = policy;
}

double UCTNode::get_visits(visit_type type) const {
    //if (type == SEL) { return m_sel_visits; } else 
    if (type == WR) { return m_visits; }
    else if (type == VL) { return m_visits + m_virtual_loss * VIRTUAL_LOSS_COUNT; }
}

float UCTNode::get_raw_eval(int tomove, double virtual_loss) const {
    auto visits = get_visits(WR) + virtual_loss;
    assert(visits > 0);
    auto blackeval = get_blackevals();
    if (tomove == FastBoard::WHITE) {
        blackeval += virtual_loss;
    }
    auto eval = static_cast<float>(blackeval / double(visits));
    if (tomove == FastBoard::WHITE) {
        eval = 1.0f - eval;
    }
    return eval;
}

float UCTNode::get_eval(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    return get_raw_eval(tomove, m_virtual_loss * VIRTUAL_LOSS_COUNT);
}

float UCTNode::get_net_eval(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return 1.0f - m_net_eval;
    }
    return m_net_eval;
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_blackevals, double(eval));
}

float uct_value(float q, float p, double v, double v_total) {
    return q + cfg_puct * p * sqrt(v_total) / (1.0 + v);
}

float binary_search_visits(std::function<float(float)> f, float v_init) {
    auto low = 0.0;
    auto high = v_init;
    if (f(low) > 0.0) { return 0.0; }
    while (f(high) < 0.0) { low = high; high = 2.0 * high; }
    while (true) {
        auto mid = (low + high) / 2.0;
        auto fmid = f(mid);
        if (abs(fmid) < 0.01) { return mid; } // 0.0001
        if (fmid < 0.0) { low = mid; }
        else { high = mid; }
    }
}

float factor(float q_c, float p_c, float v_c, float q_a, float p_a, float v_a, float v_total) {
    auto v_additional = binary_search_visits(
        [q_c, p_c, v_c, q_a, p_a, v_a, v_total](float x) {
        return uct_value(q_c, p_c, v_c, v_total + x) - uct_value(q_a, p_a, v_a + x, v_total + x); },
        1.0 + v_total);

    auto factor_ = v_total / (v_total + v_additional);
    if (factor_ < 0.0) {
        myprintf("chosen: %f, actual best: %f policy\n", p_c, p_a);
        myprintf("chosen: %f, actual best: %f visits\n", v_c, v_a);
        myprintf("chosen: %f, actual best: %f Q\n", q_c, q_a);
        myprintf("chosen: %f, actual best: %f Q+U before addiaional visits\n",
            uct_value(q_c, p_c, v_c, v_total),
            uct_value(q_a, p_a, v_a, v_total));
        myprintf("chosen: %f, actual best: %f Q+U after %f additional visits\n",
            uct_value(q_c, p_c, v_c, v_total + v_additional),
            uct_value(q_a, p_a, v_a + v_additional, v_total + v_additional),
            v_additional);
        myprintf("parentvisits: %f, factor: %f\n\n", v_total, factor_);
    }

    return factor_;
}

std::pair<UCTNode*, float> UCTNode::uct_select_child(int color, bool is_root) {
    // add (float) everywhere in front of visits..
    /*
    // Count parentvisits manually to avoid issues with transpositions.
    auto parentvisits = 0.0;
    auto actual_parentvisits = 0.0;
    for (const auto& child : m_children) {
        if (child.valid()) {
            actual_parentvisits += child.get_visits();
            if (cfg_vl_in_parentvisits) {
                parentvisits += child.get_visits(VL);
            }
            if (child.get_visits(VL) == 0.0) { // VL
                break; // children are ordered by policy (initially) or by visits (NodeComp), so this is good.
            }
        }
    }
    if (!cfg_vl_in_parentvisits) { parentvisits = actual_parentvisits; }
    */
    auto actual_parentvisits = get_visits(); 
    // will be somewhat smaller than sum of children visits due to fractional backup
    auto parentvisits = actual_parentvisits;
    if (cfg_vl_in_parentvisits) { parentvisits = get_visits(VL); }
 
    auto numerator = std::sqrt(parentvisits);
    auto actual_numerator = std::sqrt(actual_parentvisits);
    auto parent_eval = get_raw_eval(color); // get_visits(WR) > 0.0 ? get_raw_eval(color) : get_net_eval(color);

    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto actual_best = best;
    auto best_value = std::numeric_limits<double>::lowest();
    auto best_actual_value = best_value;
    auto actual_value_of_best = best_value;
    auto q_of_best = 0.0;
    auto q_of_actual_best = 0.0;
    auto total_visited_policy = 0.0f;
    auto policy_of_best = 0.0f;
    auto policy_of_actual_best = 0.0f;
    auto visits_of_best = 0.0;
    auto visits_of_actual_best = 0.0;
    auto num_children = m_children.size();
    auto wt = std::min(1.0f, std::pow(float(parentvisits), cfg_exponent) / cfg_uniform_visits);

#ifdef UCT_SOFTMAX
    auto accum = 0.0;
    auto accum_vector = std::vector<double>{};
    auto index_vector = std::vector<int>{};
    auto q_vector = std::vector<double>{};
    accum_vector.reserve(m_children.size());
    index_vector.reserve(m_children.size());
    q_vector.reserve(m_children.size());
#endif

#ifdef LOCK_DEBUG
    auto size = m_children.size();
    std::vector<float> winrates;
    std::vector<float> pucts;
#endif

    for (auto i = 0; i < m_children.size(); i++) {
        auto& child = m_children[i];
        if (!child.active()) {
            continue;
        }

        auto winrate = parent_eval;
        // Estimated eval for unknown nodes = parent eval - reduction
        // Lower the expected eval for moves that are likely not the best.
        // Do not do this if we have introduced noise at this node exactly
        // to explore more.
        if (child.is_inflated() && child.get()->m_accumulated_vl >= 48)
            winrate = -1.0f;
        winrate -= (is_root? cfg_fpu_root_reduction : cfg_fpu_reduction) * std::sqrt(total_visited_policy);

        auto actual_winrate = winrate;
        auto visits = child.get_visits();
        if (visits > 0.0) {
            winrate = child.get_eval(color);
            actual_winrate = child.get_raw_eval(color);
        }
        auto psa = child.get_policy();
        psa = wt / num_children + (1.0f - wt) * psa;
        total_visited_policy += (1.0f - wt) * psa;
        auto denom = 1.0 + child.get_visits(VL);
        auto actual_denom = 1.0 + visits;
        auto puct = cfg_puct * psa * (numerator / denom);
        auto actual_puct = cfg_puct * psa * (actual_numerator / actual_denom);
        auto value = winrate + puct;
        auto actual_value = actual_winrate + actual_puct;

#ifdef UCT_SOFTMAX
        if (cfg_uct_temp > 0.0) { 
            accum += std::exp(value / cfg_uct_temp); // takes too long!!
            accum_vector.emplace_back(accum); 
            index_vector.emplace_back(i);
            q_vector.emplace_back(actual_winrate);
        }
#endif

#ifdef LOCK_DEBUG
        winrates.emplace_back(winrate);
        pucts.emplace_back(cfg_puct);
        pucts.emplace_back(psa);
        pucts.emplace_back(parentvisits);
        pucts.emplace_back(actual_parentvisits);
        pucts.emplace_back(denom);
#endif
        
        if (actual_value > best_actual_value) {
            best_actual_value = actual_value;
            q_of_actual_best = actual_winrate;
            policy_of_actual_best = psa;
            visits_of_actual_best = visits;
            actual_best = &child;
        }
        if (value > best_value) {
            best = &child;
            best_value = value;
            actual_value_of_best = actual_value;
            q_of_best = actual_winrate;
            policy_of_best = psa;
            visits_of_best = visits;
            if (visits == 0.0) break;
        }
    }

#ifdef UCT_SOFTMAX
    if (cfg_uct_temp > 0.0) {
        auto distribution = std::uniform_real_distribution<double>{ 0.0, accum };
        auto pick = distribution(Random::get_Rng());
        for (auto i = 0; i < accum_vector.size(); i++) {
            if (pick < accum_vector[i]) {
                best = &(m_children[index_vector[i]]);
                q_of_best = q_vector[i];
                policy_of_best = best->get_policy();
                visits_of_best = best->get_visits();
                break;
            }
        }
    }
#endif

    assert(best != nullptr);
#ifdef LOCK_DEBUG
    // if (is_root) { myprintf("%f, %f\n", parentvisits, best_value); }//
    if (best == nullptr) {
        myprintf("%d, %f,,, %f, %f, %f, %f ", size, parent_eval, pucts[0], pucts[1],pucts[2],pucts[3]);
        //for (auto val : pucts) { myprintf("%f, ", val); }
        //for (auto val : winrates) { myprintf("%f, ", val); }
        return std::make_pair(nullptr, 1.0f);
    }
#endif
    best->inflate();
    if (best == actual_best || !cfg_frac_backup) return std::make_pair(best->get(), 1.0f);
#ifdef UCT_SOFTMAX
    return std::make_pair(best->get(), 1.0f);
#else
    return std::make_pair(best->get(), factor(q_of_best, policy_of_best, visits_of_best,
                                              q_of_actual_best, policy_of_actual_best, visits_of_actual_best,
                                              actual_parentvisits));
#endif
}

class NodeComp : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    NodeComp(int color) : m_color(color) {};
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {

        auto a_visits = a.get_visits();
        auto b_visits = b.get_visits();

        // if visits are not same, sort on visits
        if (a_visits!= b_visits) {
            return a_visits < b_visits;
        }

        // neither has visits, sort on policy prior
        if (a_visits == 0) {
            return a.get_policy() < b.get_policy();
        }

        // both have same non-zero number of visits
        return a.get_raw_eval(m_color) < b.get_raw_eval(m_color);
    }
private:
    int m_color;
};

void UCTNode::sort_children(int color) {
    while (!pre_acquire_writer()) {}
    acquire_writer();
    std::stable_sort(rbegin(m_children), rend(m_children), NodeComp(color));
    release_writer();
}

UCTNode& UCTNode::get_best_root_child(int color, bool running) {
    acquire_reader();

    assert(!m_children.empty());

    auto ret = std::max_element(begin(m_children), end(m_children),
                                NodeComp(color));
    release_reader();
    ret->inflate();

    return *(ret->get());
}

void UCTNode::invalidate() {
    m_status = INVALID;
}

void UCTNode::set_active(const bool active) {
    if (valid()) {
        m_status = active ? ACTIVE : PRUNED;
    }
}

bool UCTNode::valid() const {
    return m_status != INVALID;
}

bool UCTNode::active() const {
    return m_status == ACTIVE;
}

float UCTNode::get_min_psa_ratio() {
    const auto mem_full = UCTNodePointer::get_tree_size() / static_cast<float>(cfg_max_tree_size);
    // If we are halfway through our memory budget, start trimming
    // moves with very low policy priors.
    if (mem_full > 0.5f) {
        // Memory is almost exhausted, trim more aggressively.
        if (mem_full > 0.95f) {
            // if completely full just stop expansion by returning an impossible number
            if (mem_full >= 1.0f) {
                return 2.0f;
            }
            return 0.01f;
        }
        return 0.001f;
    }
    return 0.0f;
}

void UCTNode::acquire_reader() {
    //myprintf("%d acquire\n", m_lock.load());
    while (true) {
        if (m_lock >= 170) { continue; }
        if (m_lock.fetch_add(1) >= 170) {
            --m_lock;
            continue;
        }
        //virtual_loss();
        return;
    }
}

void UCTNode::release_reader(uint16_t vl, bool incr) {
    //myprintf("%d release\n", m_lock.load());
    if (incr) { virtual_loss(vl); }
    else { virtual_loss_undo(vl); }
    --m_lock;
}

bool UCTNode::pre_acquire_writer() {
    //myprintf("%d pre-acquire\n", m_lock.load());
    std::uint8_t expected;
    do {
        expected = m_lock;
        if (expected >= 85) {
            return false;
        }
    } while (!m_lock.compare_exchange_strong(expected, 86 + expected));
    return true;
}
// readers can still come in, but not other writers (if success)
// long time gap (NN eval) before writer privilege actually needed

void UCTNode::acquire_writer() {
    //myprintf("%d acquire-writer\n", m_lock.load());
    m_lock += 85;
    while (m_lock != 171) {}
}
// only called after pre-acquired writer
// after this no other readers or writers can come in
// virtual_loss won't be incremented any more
// can now undo all virtual loss at this node (and matching amount from each ancestor)

void UCTNode::release_writer() {
    //myprintf("%d release-writer\n", m_lock.load());
    m_lock -= 171;
}

UCTNode::Action UCTNode::get_action(bool is_root) {
#ifdef LOCK_DEBUG
    /*
    std::uint8_t lk = m_lock;
    if (3 <= lk && lk <= 84) { myprintf("problem! %d\n", lk); }
    if (88 <= lk && lk <= 169) { myprintf("problem+! %d\n", lk); }
    */
    if (m_lock == 84) {
        myprintf("problem!\n");
        std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    }
#endif
    while (true) {
        auto lock = m_lock.load();
        if (lock >= 170) { continue; }
        if (lock < 85 &&
            (m_visits == 0.0 || is_root || expandable(get_min_psa_ratio())) &&
            pre_acquire_writer()) {
            virtual_loss();
            m_accumulated_vl++;
            return WRITE;
        }
        is_root = false;
        lock = m_lock.fetch_add(1);
        if (lock >= 170) { --m_lock; continue; }
        //myprintf("%d get-action\n", lock);
        virtual_loss();
        if (has_children()) { while (m_visits == 0.0) {}; return READ; }
        else {
            if (lock >= 85) {
                m_accumulated_vl++;
                release_reader();
                return FAIL;
            }
            else {
                myprintf("No children due to low memory!\n");
                release_reader();
                return BACKUP;
            }
        }
    }
}