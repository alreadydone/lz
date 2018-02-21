/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Henrik Forsten

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

#ifndef CUDNN_H_INCLUDED
#define CUDNN_H_INCLUDED

#include "config.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <cudnn.h>

struct conv_descriptor {
	cudnnTensorDescriptor_t input_descriptor;
	cudnnTensorDescriptor_t output_descriptor;
	cudnnTensorDescriptor_t bias_descriptor;
	cudnnFilterDescriptor_t kernel_descriptor;
	cudnnActivationDescriptor_t activation_descriptor;
	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnConvolutionFwdAlgo_t convolution_algorithm;
};

class CuDNN;

class Layer {
    friend class OpenCL_Network;
private:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    bool is_input_convolution{false};
    bool is_residual_block{false};
    bool is_convolve1{false};
	conv_descriptor conv_desc;
    std::vector<net_t*> weights;
	size_t workspace_size;
};

class ThreadData {
    friend class CuDNN;
    friend class OpenCL_Network;
private:
    float *m_workspace;
    float *m_InBuffer;
    float *m_OutBuffer;
    float *m_ResidualBuffer;
    bool m_is_initialized{false};
    bool m_buffers_allocated{false};
};

class OpenCL_Network {
public:
    OpenCL_Network(CuDNN & opencl) : m_cudnn(opencl) {}
    CuDNN & getCuDNN() {
        return m_cudnn;
    }

    void push_input_convolution(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights,
                       const std::vector<float>& biases);

    void push_residual(unsigned int filter_size,
                       unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights_1,
                       const std::vector<float>& biases_1,
                       const std::vector<float>& weights_2,
                       const std::vector<float>& biases_2);

    void push_convolve1(unsigned int channels,
                       unsigned int outputs,
                       const std::vector<float>& weights);

    size_t get_layer_count() const {
        return m_layers.size();
    }

    void forward(const std::vector<net_t>& input,
            std::vector<net_t>& output_pol,
            std::vector<net_t>& output_val);

private:

    void push_weights(size_t layer, const std::vector<float>& weights) {
        add_weights(layer, weights.size(), weights.data());
    }
    void add_weights(size_t layer, size_t size, const float* weights);

    CuDNN & m_cudnn;
    std::vector<Layer> m_layers;

};

class CuDNN {
    friend class OpenCL_Network;
public:
    void initialize(const int channels, const std::vector<int> & gpus,
                    bool silent = false);
    void ensure_thread_initialized(void);
    std::string get_device_name();

    std::vector<size_t> get_sgemm_tuners(void);

private:

void convolve(net_t *bufferIn,
			  net_t *bufferOut,
			  net_t *weights,
			  net_t *workspace,
			  size_t workspace_bytes,
			  const conv_descriptor& conv_desc);

void convolveActivation(net_t *bufferIn,
					 net_t *bufferOut,
					 net_t *weights,
					 net_t *residualBuffer,
					 net_t *biases,
					 net_t *workspace,
					 size_t workspace_bytes,
					 const conv_descriptor& conv_desc);

size_t convolve_init(int channels, int outputs, int kernel_size,
		conv_descriptor& conv_desc);

	cudnnHandle_t m_handle;
    bool m_init_ok{false};
};

extern thread_local ThreadData opencl_thread_data;
#endif
