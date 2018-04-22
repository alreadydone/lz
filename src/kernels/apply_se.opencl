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

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.

R"(
    __kernel void apply_se(
                  const int channels,
                  __global const net_t * restrict input,
                  __global net_t * restrict residual,
                  __constant const net_t * restrict scales,
                  __constant const net_t * restrict prelu_alphas) {

        const int col = get_global_id(0);  // column
        const int c = get_global_id(1);  // channel

        if (c < channels && col < BOARD_SIZE) {
            const real prelu_alpha = vload_net_t(c, prelu_alphas);
            real sig_scale = vload_net_t(c, scales);
            sig_scale = 1.0f/(1.0f + exp(-sig_scale));

            for ( int i = 0; i < BOARD_SIZE; i++) {
                const int idx = c * BOARD_SQUARES + col * BOARD_SIZE + i;
                const real in = vload_net_t(idx, input);
                const real res = vload_net_t(idx, residual);

                real val = sig_scale * in + res;

                val = val > 0.0f ? val : prelu_alpha * val;

                vstore_net_t(val, idx, residual);
            }
        }
    }
// End of the C++11 raw string literal
)"
