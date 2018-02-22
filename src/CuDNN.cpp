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

#include "config.h"

#ifdef USE_CUDNN

#include <cassert>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <iterator>
#include <limits>
#include <stdexcept>

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "Network.h"
#include "GTP.h"
#include "Utils.h"
#include "CuDNN.h"

using namespace Utils;

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

thread_local ThreadData opencl_thread_data;

void CuDNN::initialize(const int channels, const std::vector<int> & gpus,
                       bool silent) {

	/* For compatibility with OpenCL implementation */
	(void)channels;

    auto best_bandwidth = 0.0f;
    auto found_device = false;
    auto nDevices = 0;
    auto best_device_id = 0;
    cudaDeviceProp best_device;

    cudaGetDeviceCount(&nDevices);

    if (!silent) {
        myprintf("Detected %d CUDA devices.\n", nDevices);
    }

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        auto bandwidth = 2.0f*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6;
        if (!silent) {
            myprintf("Device Number: %d\n", i);
            myprintf("  Device name: %s\n", prop.name);
            myprintf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            myprintf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                   bandwidth);
        }

        bool preferred =
            std::find(cbegin(gpus), cend(gpus), i) != cend(gpus);

        if ( (bandwidth > best_bandwidth) || preferred) {
            best_bandwidth = bandwidth;
            best_device = prop;
            best_device_id = i;
            if (preferred) {
                best_bandwidth = std::numeric_limits<decltype(best_bandwidth)>::max();
            } else {
                best_bandwidth = bandwidth;
            }
            found_device = true;
        }
    }

    if (!found_device) {
        throw std::runtime_error("No suitable CUDA device found.");
    }

    myprintf("Selected device: %s\n", best_device.name);
    myprintf("with compute capability %d.%d.\n", best_device.major, best_device.minor);

    cudaSetDevice(best_device_id);

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    m_handle = cudnn;
    m_init_ok = true;
}

void OpenCL_Network::push_input_convolution(unsigned int filter_size,
				   unsigned int channels,
				   unsigned int outputs,
				   const std::vector<float>& weights,
				   const std::vector<float>& biases) {
	size_t layer = get_layer_count();
	push_weights(layer, weights);
	push_weights(layer, biases);
	m_layers[layer].is_input_convolution = true;
	m_layers[layer].outputs = outputs;
	m_layers[layer].filter_size = filter_size;
	m_layers[layer].channels = channels;

	conv_descriptor conv_desc;
	auto wsize = m_cudnn.convolve_init(channels, outputs, 3, conv_desc);
	m_layers[layer].conv_desc = conv_desc;
	m_layers[layer].workspace_size = wsize;
}

void OpenCL_Network::push_residual(unsigned int filter_size,
				   unsigned int channels,
				   unsigned int outputs,
				   const std::vector<float>& weights_1,
				   const std::vector<float>& biases_1,
				   const std::vector<float>& weights_2,
				   const std::vector<float>& biases_2) {
	size_t layer = get_layer_count();
	push_weights(layer, weights_1);
	push_weights(layer, biases_1);
	push_weights(layer, weights_2);
	push_weights(layer, biases_2);
	m_layers[layer].is_residual_block = true;
	m_layers[layer].outputs = outputs;
	m_layers[layer].filter_size = filter_size;
	m_layers[layer].channels = channels;

	conv_descriptor conv_desc;
	auto wsize = m_cudnn.convolve_init(channels, outputs, 3, conv_desc);
	m_layers[layer].conv_desc = conv_desc;
	m_layers[layer].workspace_size = wsize;

}

void OpenCL_Network::push_convolve1(unsigned int channels,
				   unsigned int outputs,
				   const std::vector<float>& weights) {
	size_t layer = get_layer_count();
	push_weights(layer, weights);
	m_layers[layer].is_convolve1 = true;
	m_layers[layer].outputs = outputs;
	m_layers[layer].channels = channels;

	conv_descriptor conv_desc;
	auto wsize = m_cudnn.convolve_init(channels, outputs, 1, conv_desc);
	m_layers[layer].conv_desc = conv_desc;
	m_layers[layer].workspace_size = wsize;
}

void CuDNN::convolve(net_t *bufferIn,
					 net_t *bufferOut,
					 net_t *weights,
					 net_t *workspace,
					 size_t workspace_bytes,
					 const conv_descriptor& conv_desc) {

	const float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnConvolutionForward(m_handle,
                                   &alpha,
                                   conv_desc.input_descriptor,
                                   bufferIn,
                                   conv_desc.kernel_descriptor,
                                   weights,
                                   conv_desc.convolution_descriptor,
                                   conv_desc.convolution_algorithm,
                                   workspace,
                                   workspace_bytes,
                                   &beta,
                                   conv_desc.output_descriptor,
                                   bufferOut));
}

void CuDNN::convolveActivation(net_t *bufferIn,
					 net_t *bufferOut,
					 net_t *weights,
					 net_t *residualBuffer,
					 net_t *biases,
					 net_t *workspace,
					 size_t workspace_bytes,
					 const conv_descriptor& conv_desc) {

	const float alpha1 = 1.0f;
	float alpha2 = 0.0f;

	net_t *residual = bufferOut;

	if (residualBuffer != nullptr) {
		alpha2 = 1.0f;
		residual = residualBuffer;
	}

	checkCUDNN(cudnnConvolutionBiasActivationForward(
			m_handle,
			&alpha1,
			conv_desc.input_descriptor,
		    bufferIn,
			conv_desc.kernel_descriptor,
			weights,
			conv_desc.convolution_descriptor,
            conv_desc.convolution_algorithm,
            workspace,
            workspace_bytes,
			&alpha2,
			conv_desc.output_descriptor,
			residual,
			conv_desc.bias_descriptor,
			biases,
			conv_desc.activation_descriptor,
			conv_desc.output_descriptor,
		    bufferOut));
}

size_t CuDNN::convolve_init(int channels, int outputs, int kernel_size,
		conv_descriptor& conv_desc) {

	checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv_desc.input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/channels,
                                      /*image_height=*/19,
                                      /*image_width=*/19));

	checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv_desc.output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/outputs,
                                      /*image_height=*/19,
                                      /*image_width=*/19));

	checkCUDNN(cudnnCreateTensorDescriptor(&conv_desc.bias_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv_desc.bias_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/outputs,
                                      /*image_height=*/1,
                                      /*image_width=*/1));

	checkCUDNN(cudnnCreateFilterDescriptor(&conv_desc.kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(conv_desc.kernel_descriptor,
									  /*dataType=*/CUDNN_DATA_FLOAT,
									  /*format=*/CUDNN_TENSOR_NCHW,
									  /*out_channels=*/outputs,
									  /*in_channels=*/channels,
									  /*kernel_height=*/kernel_size,
									  /*kernel_width=*/kernel_size));

	checkCUDNN(cudnnCreateActivationDescriptor(&conv_desc.activation_descriptor));
	checkCUDNN(cudnnSetActivationDescriptor(conv_desc.activation_descriptor,
									  /*mode=*/CUDNN_ACTIVATION_RELU,
									  /*NanPropagation=*/CUDNN_NOT_PROPAGATE_NAN,
									  /*coef=*/0));

	auto pad_size = 0;

	if (kernel_size == 1) {
		pad_size = 0;
	} else if (kernel_size == 3) {
		pad_size = 1;
	}

	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc.convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc.convolution_descriptor,
											   /*pad_height=*/pad_size,
											   /*pad_width=*/pad_size,
											   /*vertical_stride=*/1,
											   /*horizontal_stride=*/1,
											   /*dilation_height=*/1,
											   /*dilation_width=*/1,
											   /*mode=*/CUDNN_CROSS_CORRELATION,
											   /*computeType=*/CUDNN_DATA_FLOAT));

	checkCUDNN(
		cudnnGetConvolutionForwardAlgorithm(m_handle,
											conv_desc.input_descriptor,
											conv_desc.kernel_descriptor,
											conv_desc.convolution_descriptor,
											conv_desc.output_descriptor,
											CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
											/*memoryLimitInBytes=*/0,
											&conv_desc.convolution_algorithm));


	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_handle,
													   conv_desc.input_descriptor,
													   conv_desc.kernel_descriptor,
													   conv_desc.convolution_descriptor,
													   conv_desc.output_descriptor,
													   conv_desc.convolution_algorithm,
													   &workspace_bytes));
	return workspace_bytes;
}


void OpenCL_Network::add_weights(size_t layer,
                                 size_t size,
                                 const float * weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(Layer());
    }

    auto converted_weights = std::vector<net_t>();
    for(auto i = size_t{0}; i < size; i++) {
        converted_weights.emplace_back(weights[i]);
    }

    auto weightSize = size * sizeof(decltype(converted_weights)::value_type);

	net_t *device_mem;
	auto err = cudaMalloc((void**)&device_mem, weightSize);
	if (err != cudaSuccess) {
		throw std::runtime_error("cudaMalloc failed.");
	}
	err = cudaMemcpy(device_mem, (net_t*)&converted_weights[0], weightSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw std::runtime_error("cudaMemcpy failed.");
	}
    m_layers.back().weights.emplace_back(device_mem);
}

void OpenCL_Network::forward(const std::vector<net_t>& input,
                             std::vector<net_t>& output_pol,
                             std::vector<net_t>& output_val) {
    constexpr auto width = 19;
    constexpr auto height = 19;
    constexpr auto one_plane = width * height * sizeof(net_t);
    const auto finalSize_pol = m_layers[m_layers.size()-2].outputs * one_plane;
    const auto finalSize_val = m_layers.back().outputs * one_plane;

    if (!opencl_thread_data.m_buffers_allocated) {
        auto max_wsize = size_t{0};
        auto max_channels = unsigned{0};
        for (const auto& layer : m_layers) {
            max_wsize = std::max(max_wsize, layer.workspace_size);
			max_channels = std::max(max_channels,
								std::max(layer.channels, layer.outputs));
        }
		auto alloc_insize = max_channels * one_plane;

		net_t *d_workspace;
		auto err = cudaMalloc((void**)&d_workspace, max_wsize);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
		}

		net_t *d_InBuffer;
		err = cudaMalloc((void**)&d_InBuffer, alloc_insize);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
		}

		net_t *d_OutBuffer;
		err = cudaMalloc((void**)&d_OutBuffer, alloc_insize);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
		}

		net_t *d_ResidualBuffer;
		err = cudaMalloc((void**)&d_ResidualBuffer, alloc_insize);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
		}
		opencl_thread_data.m_workspace = d_workspace;
		opencl_thread_data.m_InBuffer = d_InBuffer;
		opencl_thread_data.m_OutBuffer = d_OutBuffer;
		opencl_thread_data.m_ResidualBuffer = d_ResidualBuffer;
        opencl_thread_data.m_buffers_allocated = true;
    }

	auto workspace = opencl_thread_data.m_workspace;
	auto InBuffer = opencl_thread_data.m_InBuffer;
	auto OutBuffer = opencl_thread_data.m_OutBuffer;
	auto ResidualBuffer = opencl_thread_data.m_ResidualBuffer;

	const auto inSize = sizeof(net_t) * input.size();
	auto err = cudaMemcpy(InBuffer, (net_t*)&input[0], inSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
	}

	for (auto iter = cbegin(m_layers); iter != cend(m_layers); iter++) {
        const auto& layer = *iter;
        const auto niter = std::next(iter);

        if (layer.is_input_convolution) {
            assert(niter != cend(m_layers));
            auto conv_weights = begin(layer.weights);
            auto conv_biases = begin(layer.weights) + 1;
            m_cudnn.convolveActivation(InBuffer,
					 OutBuffer,
                     conv_weights[0],
					 nullptr,
                     conv_biases[0],
					 workspace,
					 layer.workspace_size,
					 layer.conv_desc);
			std::swap(InBuffer, OutBuffer);
        } else if (layer.is_residual_block) {
            assert(layer.channels == layer.outputs);
            assert(niter != cend(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto conv1_biases = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 2;
            auto conv2_biases = begin(layer.weights) + 3;

            m_cudnn.convolveActivation(InBuffer,
					 OutBuffer,
                     conv1_weights[0],
					 nullptr,
                     conv1_biases[0],
					 workspace,
					 layer.workspace_size,
					 layer.conv_desc);

			m_cudnn.convolveActivation(OutBuffer,
					 ResidualBuffer,
                     conv2_weights[0],
					 InBuffer,
                     conv2_biases[0],
					 workspace,
					 layer.workspace_size,
					 layer.conv_desc);
			std::swap(InBuffer, ResidualBuffer);

        } else {
            assert(layer.is_convolve1);

			m_cudnn.convolve(InBuffer,
				 OutBuffer,
				 layer.weights[0],
				 workspace,
				 layer.workspace_size,
				 layer.conv_desc);

            if (niter == cend(m_layers)) {
				/* Value */
				auto err = cudaMemcpy((net_t*)&output_val[0], OutBuffer, finalSize_val, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
				}
            } else {
				/* Policy */
				auto err = cudaMemcpy((net_t*)&output_pol[0], OutBuffer, finalSize_pol, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
				}
            }
        }
    }

}


#endif
