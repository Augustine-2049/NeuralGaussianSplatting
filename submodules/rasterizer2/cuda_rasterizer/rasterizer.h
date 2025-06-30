#pragma once
#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
		
		static int getidxmap(
			std::function<char* (size_t)> idxBuffer,
			std::function<char* (size_t)> BinImgFuncBuffer,
			std::function<char* (size_t)> PixelBuffer,
			const int P,
			const int M,
			const int W,
			const int H,
			const float* means3D,
			const float* normal,
			const float* scales,
			const float* rotations,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx,
			const float tan_fovy,
			const float* feature_vector,
			const float* sh,
			const float* campos,
			int* out_idx,
			float* out_color,
			float* out_depth,
			float* out_feature,
			int* radii,
			bool debug);

		static void getidxmap_backward(
			const int p,
			const int w,
			const int h,
			const float* means3d,
			const float* campos,
			int* out_idx,
			float* out_feature,
			float* dl_dfeature_vector,
			bool debug);
	};
};