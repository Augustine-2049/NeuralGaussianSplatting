#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);



std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GetIdxMapCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& normal,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& feature_vector,
	const torch::Tensor& sh,
	const torch::Tensor& campos,
	const bool debug);

std::tuple<torch::Tensor>
GetIdxMapBackwardCUDA(
	const torch::Tensor& means3D,
	const int image_height,
	const int image_width,
	const torch::Tensor& campos,
	const torch::Tensor& out_idx, // [H, W]
	const torch::Tensor& out_feature, // [H, W, NUM_CHANNEL]
	const bool debug);

