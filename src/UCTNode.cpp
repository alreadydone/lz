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
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Utils.h"

using namespace Utils;

UCTNode::UCTNode(int vertex, float policy) : m_move(vertex), m_policy(policy) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

std::array<std::array<int, NUM_INTERSECTIONS>,
    Network::NUM_SYMMETRIES> Network::symmetry_nn_idx_table;

void UCTNode::create_children(Network::Netresult& raw_netlist0,
                               int symmetry,
                               std::atomic<int>& nodecount,
                               GameState& state, 
                               float min_psa_ratio) {

    Network::Netresult raw_netlist;
    m_net_eval = raw_netlist.winrate = raw_netlist0.winrate;
    const auto to_move = state.board.get_to_move();
    // our search functions evaluate from black's point of view
    if (state.board.white_to_move()) {
        m_net_eval = 1.0f - m_net_eval;
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

    link_nodelist(nodecount, nodelist, min_psa_ratio);
    return;
}

void UCTNode::link_nodelist(std::atomic<int>& nodecount,
                            std::vector<Network::PolicyVertexPair>& nodelist,
                            float min_psa_ratio) {
    assert(min_psa_ratio < m_min_psa_ratio_children);

    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));

    const auto max_psa = nodelist[0].first;
    const auto old_min_psa = max_psa * m_min_psa_ratio_children;
    const auto new_min_psa = max_psa * min_psa_ratio;
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
            ++nodecount;
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

void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}

void UCTNode::update(float eval, float factor, float sel_factor) {
    atomic_add(m_visits, double(factor));
    atomic_add(m_blackevals, double(eval*factor));
    atomic_add(m_sel_visits, double(sel_factor));
    virtual_loss_undo();
}

bool UCTNode::has_children() const {
    return m_min_psa_ratio_children <= 1.0f;
}

bool UCTNode::expandable(const float min_psa_ratio) const {
#ifndef NDEBUG
    if (m_min_psa_ratio_children == 0.0f) {
        // If we figured out that we are fully expandable
        // it is impossible that we stay in INITIAL state.
        assert(m_expand_state.load() != ExpandState::INITIAL);
    }
#endif
    return min_psa_ratio < m_min_psa_ratio_children;
}

float UCTNode::get_policy() const {
    return m_policy;
}

void UCTNode::set_policy(float policy) {
    m_policy = policy;
}

double UCTNode::get_visits(visit_type type) const {
    if (type == SEL) { return m_sel_visits; }
    else { return m_visits; }
}

float UCTNode::get_raw_eval(int tomove, int virtual_loss) const {
    auto visits = get_visits(WR) + virtual_loss;
    assert(visits > 0);
    auto blackeval = get_blackevals();
    if (tomove == FastBoard::WHITE) {
        blackeval += static_cast<double>(virtual_loss);
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
    return get_raw_eval(tomove, m_virtual_loss);
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

double binary_search_visits(std::function<double(double)> f, double v_init) {
    auto low = 0.0;
    auto high = v_init;
    while (f(high) < 0.0) { low = high; high = 2.0 * high; }
    while (true) {
        auto mid = (low + high) / 2.0;
        auto fmid = f(mid);
        if (abs(fmid) < 0.000001) { return mid; }
        if (fmid < 0.0) { low = mid; }
        else { high = mid; }
    }
}

float factor(float q_c, float p_c, double v_c, float q_a, float p_a, double v_a, double v_total) {
    auto v_additional = binary_search_visits(
        [q_c, p_c, v_c, q_a, p_a, v_a, v_total](double x) {
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
    wait_expanded();

    // Count parentvisits manually to avoid issues with transpositions.
    auto total_visited_policy = 0.0f;
    auto parentvisits = 0.0;
    for (const auto& child : m_children) {
        if (child.valid()) {
            parentvisits += child.get_visits();
            if (child.get_visits(WR) > 0.0) {
                total_visited_policy += child.get_policy();
            }
            else {
                break; // children are ordered by policy (initially) or by visits (NodeComp), so this is good.
            }
        }
    }
    auto numerator = std::sqrt(parentvisits);
    auto parent_eval = get_visits(WR) > 0.0 ? get_raw_eval(color) : get_net_eval(color);

    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto actual_best = best;
    auto best_value = std::numeric_limits<double>::lowest();
    auto best_actual_value = best_value;
    auto actual_value_of_best = best_value;
    auto q_of_best = 0.0;
    auto q_of_actual_best = 0.0;
    total_visited_policy = 0.0f;
    auto policy_of_best = 0.0f;
    auto policy_of_actual_best = 0.0f;
    auto visits_of_best = 0.0;
    auto visits_of_actual_best = 0.0;

    for (auto& child : m_children) {
        if (!child.active()) {
            continue;
        }

        auto winrate = parent_eval;
        // Estimated eval for unknown nodes = parent eval - reduction
        // Lower the expected eval for moves that are likely not the best.
        // Do not do this if we have introduced noise at this node exactly
        // to explore more.
        if (!cfg_noise) {
            winrate -= (is_root ? cfg_fpu_root_reduction : cfg_fpu_reduction) * std::sqrt(total_visited_policy);
        }
        auto actual_winrate = winrate;
        bool has_visits = false;
        if (child.get_visits(WR) > 0.0) {
            winrate = child.get_eval(color);
            actual_winrate = child.get_raw_eval(color);
            has_visits = true;
        }
        auto psa = child.get_policy();
        total_visited_policy += psa;
        auto denom = 1.0 + child.get_visits();
        auto puct = cfg_puct * psa * (numerator / denom);
        auto value = winrate + puct;
        auto actual_value = actual_winrate + puct;
        /*
        if (child.is_inflated() && child->m_expand_state.load() == ExpandState::EXPANDING) {
            // Someone else is expanding this node, never select it
            // if we can avoid so, because we'd block on it.
            value = -1.0f - cfg_fpu_reduction;
        }
        */
        
        if (actual_value > best_actual_value) {
            best_actual_value = actual_value;
            q_of_actual_best = actual_winrate;
            policy_of_actual_best = psa;
            visits_of_actual_best = child.get_visits();
            actual_best = &child;
        }
        auto to_expand = false;
        if (value > best_value) {
            if (!has_visits) {
                child.inflate();
                if (child->acquire_expanding()) {
                    to_expand = true;
                }
                else {
                    continue;
                }
            }
            best = &child;
            best_value = value;
            actual_value_of_best = actual_value;
            q_of_best = actual_winrate;
            policy_of_best = psa;
            visits_of_best = child.get_visits();
            if (to_expand) { break; }
        }
    }

    //assert(best != nullptr);
    if (best == nullptr) return std::make_pair(nullptr, 1.0f);
    //best->inflate();
    if (best == actual_best || !cfg_frac_backup) return std::make_pair(best->get(), 1.0f);
    return std::make_pair(best->get(), factor(q_of_best, policy_of_best, visits_of_best,
                                              q_of_actual_best, policy_of_actual_best, visits_of_actual_best,
                                              parentvisits));
}

class NodeComp : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    NodeComp(int color) : m_color(color) {};
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {
        // if visits are not same, sort on visits
        if (a.get_visits() != b.get_visits()) {
            return a.get_visits() < b.get_visits();
        }

        // neither has visits, sort on policy prior
        if (a.get_visits() == 0) {
            return a.get_policy() < b.get_policy();
        }

        // both have same non-zero number of visits
        return a.get_eval(m_color) < b.get_eval(m_color);
    }
private:
    int m_color;
};

void UCTNode::sort_children(int color) {
    std::stable_sort(rbegin(m_children), rend(m_children), NodeComp(color));
}

UCTNode& UCTNode::get_best_root_child(int color) {
    wait_expanded();

    assert(!m_children.empty());

    auto ret = std::max_element(begin(m_children), end(m_children),
                                NodeComp(color));
    ret->inflate();

    return *(ret->get());
}

size_t UCTNode::count_nodes_and_clear_expand_state() {
    auto nodecount = size_t{0};
    nodecount += m_children.size();
    if (expandable()) {
        m_expand_state = ExpandState::INITIAL;
    }
    for (auto& child : m_children) {
        if (child.is_inflated()) {
            nodecount += child->count_nodes_and_clear_expand_state();
        }
    }
    return nodecount;
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

bool UCTNode::acquire_expanding() {
    auto expected = ExpandState::INITIAL;
    auto newval = ExpandState::EXPANDING;
    return m_expand_state.compare_exchange_strong(expected, newval);
}

void UCTNode::expand_done() {
    auto v = m_expand_state.exchange(ExpandState::EXPANDED);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::expand_cancel() {
    auto v = m_expand_state.exchange(ExpandState::INITIAL);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::wait_expanded() {
    while (m_expand_state.load() == ExpandState::EXPANDING) {}
    auto v = m_expand_state.load();
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDED);
}

