#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>


namespace RASTER
{
	void preprocess(
		const int P,
		const float* orig_points,
		const float* viewmatrix,
		const float* projmatrix,
		const int W,
		const int H,
		const float tan_fovx,
		const float tan_fovy,
		int* radii,
		ushort2* pixels,
		float2* img_xy,
		float* radii_float,
		float* depths,
		uint32_t* touched
	);

}
