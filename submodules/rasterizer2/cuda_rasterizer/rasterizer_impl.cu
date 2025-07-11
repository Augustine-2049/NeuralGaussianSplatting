
#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "raster.h"



uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}


// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	// 获取当前线程的全局索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
    // 对于不可见的高斯模型（半径为 0），不生成键值对
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
        // 计算当前高斯模型在输出缓冲区中的起始偏移量
        // 高斯模型：gaussian_keys_unsorted / gaussian_values_unsorted
        // 如果是第一个高斯模型，偏移量为 0；否则使用前缀和数组中的值
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// 计算当前高斯模型覆盖的 tile 范围
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth.
        // 遍历当前高斯模型覆盖的所有 tile
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// 生成键值对的键
				// 键的结构：|  tile ID  |      depth      |
				// tile ID 是当前 tile 的索引，计算方式为 y * grid.x + x
				uint64_t key = y * grid.x + x;
				// 将 tile ID 左移 32 位，为深度值腾出空间
				key <<= 32;
				// 将深度值（float 类型）转换为 uint32_t，并将其与 tile ID 组合成完整的键
				// &depths[idx] : 深度值所在地址 - float*
				// 解引用转换后的指针，读取该内存位置的 `uint32_t` 值（即浮点数的二进制表示直接当作整数处理，以支持|操作）。
				key |= *((uint32_t*)&depths[idx]);
				// 将生成的键和值写入输出数组
				gaussian_keys_unsorted[off] = key;          // 存储键
				gaussian_values_unsorted[off] = idx;        // 存储值（高斯模型的 ID）
                // 更新偏移量，指向下一个键值对的位置
				off++;
			}
		}
	}
}

__global__ void identifyTileRanges(
int L,                     // 排序后的高斯模型列表的长度 num_renders
uint64_t* point_list_keys, // 排序后的高斯模型键列表（每个键包含 tile ID 和深度信息）
uint2* ranges              // 输出数组，存储每个 tile 的工作范围（起始和结束位置）
)
{
    // 获取当前线程的全局索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
    // 从当前键中提取 tile ID
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;  // 右移 32 位，提取 tile ID
    // 如果当前是列表的第一个元素，设置当前 tile 的起始位置为 0
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
        // 提取前一个键的 tile ID
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        // 如果当前 tile ID 与前一个 tile ID 不同，说明当前元素是新的 tile 的起始位置
		if (currtile != prevtile)
		{
            // 设置前一个 tile 的结束位置为当前索引
			ranges[prevtile].y = idx;
            // 设置当前 tile 的起始位置为当前索引
			ranges[currtile].x = idx;
		}
	}
    // 如果当前是列表的最后一个元素，设置当前 tile 的结束位置为列表长度
	if (idx == L - 1)
		ranges[currtile].y = L;
}


CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;  // 创建一个 GeometryState 对象

    // 从内存块 chunk 中分配对齐的内存，并将指针赋值给 geom 的成员变量
    // 每个 obtain 调用都会更新 chunk 指针，使其指向下一个可用内存位置
	// 指针，数据类型，大小，对齐块长度
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
    // 分配每个高斯模型覆盖的 tile 数量数据（tiles_touched），大小为 P，对齐长度为 128
	obtain(chunk, geom.tiles_touched, P, 128);
    // 使用 CUB 库计算 tiles_touched 的前缀和，并确定扫描所需的空间大小
	cub::DeviceScan::InclusiveSum(nullptr, 
		geom.scan_size,  // 输出：临时存储所需大小 
		geom.tiles_touched, // 输入数组（每个高斯覆盖的 tile 数量） 
		geom.tiles_touched, // 输出数组（原地计算，输入输出可相同） 
		P                 // 数组长度（高斯数量）
	);

	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    // 分配点偏移数据（point_offsets），大小为 P，对齐长度为 128
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;  // 返回初始化好的 GeometryState 对象
}

CudaRasterizer::Idx2Img CudaRasterizer::Idx2Img::fromChunk(char*& chunk, size_t P)
{
	Idx2Img geom;  // 创建一个 Idx2Img 对象

	// 从内存块 chunk 中分配对齐的内存，并将指针赋值给 geom 的成员变量
	// 每个 obtain 调用都会更新 chunk 指针，使其指向下一个可用内存位置
	// 指针，数据类型，大小，对齐块长度
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.pixels, P, 128);
	obtain(chunk, geom.img_xy, P, 128);
	obtain(chunk, geom.radii_float, P, 128);
	obtain(chunk, geom.touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr,
		geom.scan_size,  // 输出：临时存储所需大小 
		geom.touched, // 输入数组（每个高斯覆盖的 tile 数量） 
		geom.touched, // 输出数组（原地计算，输入输出可相同） 
		P                 // 数组长度（高斯数量）
	);

	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	// 分配点偏移数据（point_offsets），大小为 P，对齐长度为 128
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;  // 返回初始化好的 GeometryState 对象
}

CudaRasterizer::BinningImg CudaRasterizer::BinningImg::fromChunk(char*& chunk, size_t P)
{
	BinningImg binning;  // 创建一个 BinningState 对象

	// 从内存块 chunk 中分配对齐的内存，并将指针赋值给 binning 的成员变量

	// 分配点列表数据（point_list），大小为 P，对齐长度为 128
	obtain(chunk, binning.point_list, P, 128);

	// 分配未排序的点列表数据（point_list_unsorted），大小为 P，对齐长度为 128
	obtain(chunk, binning.point_list_unsorted, P, 128);

	// 分配点列表键值数据（point_list_keys），大小为 P，对齐长度为 128
	obtain(chunk, binning.point_list_keys, P, 128);

	// 分配未排序的点列表键值数据（point_list_keys_unsorted），大小为 P，对齐长度为 128
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);

	// 使用 CUB 库对点列表进行排序，并确定排序所需的空间大小
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);

	// 分配排序所需的空间（list_sorting_space），大小为 sorting_size，对齐长度为 128
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;  // 返回初始化好的 BinningState 对象
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;  // 创建一个 ImageState 对象

    // 从内存块 chunk 中分配对齐的内存，并将指针赋值给 img 的成员变量

    // 分配累积透明度数据（accum_alpha），大小为 N，对齐长度为 128
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);

    // 分配每个 tile 的工作范围数据（ranges），大小为 N，对齐长度为 128
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::PixelState CudaRasterizer::PixelState::fromChunk(char*& chunk, size_t N)
{
	PixelState img;  // 创建一个 ImageState 对象

	// 从内存块 chunk 中分配对齐的内存，并将指针赋值给 img 的成员变量

	// 分配每个 pixel 的工作范围数据（ranges），大小为 N，对齐长度为 128
	obtain(chunk, img.ranges, N, 128);
	return img;
}


CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;  // 创建一个 BinningState 对象

    // 从内存块 chunk 中分配对齐的内存，并将指针赋值给 binning 的成员变量

    // 分配点列表数据（point_list），大小为 P，对齐长度为 128
	obtain(chunk, binning.point_list, P, 128);

    // 分配未排序的点列表数据（point_list_unsorted），大小为 P，对齐长度为 128
	obtain(chunk, binning.point_list_unsorted, P, 128);

    // 分配点列表键值数据（point_list_keys），大小为 P，对齐长度为 128
	obtain(chunk, binning.point_list_keys, P, 128);

    // 分配未排序的点列表键值数据（point_list_keys_unsorted），大小为 P，对齐长度为 128
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);

    // 使用 CUB 库对点列表进行排序，并确定排序所需的空间大小
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);

    // 分配排序所需的空间（list_sorting_space），大小为 sorting_size，对齐长度为 128
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;  // 返回初始化好的 BinningState 对象
}

int CudaRasterizer::Rasterizer::forward(
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
	int* radii,
	bool debug)
{
	// ### step1 : 分配空间，预处理 ###################
	// 计算焦距
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	// 计算 GeometryState 所需的内存大小，并分配内存
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	// 从内存块初始化 GeometryState
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	// 如果 radii 为空指针，则使用几何状态内部的 radii
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

    // 定义 CUDA 线程块和网格的维度
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
    // 动态调整图像辅助缓冲区的大小
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// ### step2 : 运行高斯模型的预处理 ###################
    // 运行高斯模型的预处理（变换、边界计算、SH 到 RGB 的转换）
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

    // ### step3 : 计算高斯模型覆盖的 tile 数量的前缀和 ########
		// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomState.scanning_space, // 临时存储空间指针
		geomState.scan_size,      // 临时存储空间大小
		geomState.tiles_touched,  // 输入数组：每个高斯点影响的区块数
		geomState.point_offsets,  // 输出数组：累计偏移量
		P), debug)

	// ### step4 : 获取渲染的高斯实例总数并调整辅助缓冲区大小 #####
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);  // 计算分箱缓冲区所需的大小
	char* binning_chunkptr = binningBuffer(binning_chunk_size);      // 分配分箱缓冲区
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);  // 初始化分箱状态

	// ### step5 : 为每个渲染实例生成 [ tile | depth ] 键 #####
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)
	int bit = getHigherMsb(tile_grid.x * tile_grid.y); // 获取 tile 网格的最高有效位

	// ### step6 : 对高斯索引列表按键进行排序 #####
    // 键 = [像素id|Depth]
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

    // 初始化每个 tile 的工作范围
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// ### step7 : 确定排序列表中每个 tile 的工作范围 #####
    // 构造出 imgState.ranges[start, end], 表示每个像素处理的高斯点范围
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// ### step8 : 并行渲染每个 tile 的高斯模型 #####
	const float* feature_ptr = geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color), debug)

    // 返回渲染的高斯实例总数
	return num_rendered;
}

void CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,
	const float* dL_dpix,  // input
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	// 复用已有的内存空间
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

    // 定义 CUDA 线程块和网格的维度
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);  // [616, 409] -> [39, 26, 1]个tiled
	const dim3 block(BLOCK_X, BLOCK_Y, 1);  // [16, 16, 1]

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	// step1 : 渲染，计算mean2D、conic、透明度、颜色的损失
	const float* color_ptr = geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,  // input
		(float3*)dL_dmean2D,  // output
		(float4*)dL_dconic,  // output
		dL_dopacity,  // output
		dL_dcolor  // output
		), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,  // render中修改
		dL_dcov3D,	// 
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}


__global__ void duplicateIdxDepthImg(
	int P,
	const ushort2* pixels,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii)
{
	// 获取当前线程的全局索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || radii[idx] == 0)
		return;
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

	// high [ x | y || depth ] low
	// 解引用转换后的指针，读取该内存位置的 `uint32_t` 值（即浮点数的二进制表示直接当作整数处理，以支持|操作）。
	uint64_t key = pixels[idx].x;  // ushort16
	key <<= 16;
	key |= pixels[idx].y;  // ushort16
	key <<= 32;
	key |= *((uint32_t*)&depths[idx]);  // float32

	// high [depth || x | y] low
	//uint64_t key = *((uint32_t*)&depths[idx]); // float32
	//key <<= 16;
	//key |= pixels[idx].x; // ushort16
	//key <<= 16;
	//key |= pixels[idx].y; // ushort16

	// 将生成的键和值写入输出数组
	gaussian_keys_unsorted[off] = key;          // 存储键
	gaussian_values_unsorted[off] = idx;        // 存储值（高斯模型的 ID）
	// 更新偏移量，指向下一个键值对的位置
}
__global__ void duplicateIdxDepthImg(
	int P,
	int W, int H,
	const ushort2* pixels,
	const float2* img_xy,
	const float* radii_float,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii)
{
	// 获取当前线程的全局索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || radii[idx] == 0)
		return;
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	int2 point_image_topleft = { (int)(fmaxf(0.0f, img_xy[idx].x - radii_float[idx])), (int)(fmaxf(0.0f, img_xy[idx].y - radii_float[idx])) };
	int2 point_image_bottomright = { (int)(fminf((float)W, img_xy[idx].x + radii_float[idx] + 1.0f)), (int)(fminf((float)H, img_xy[idx].y + radii_float[idx] + 1.0f)) };
	for (short i = point_image_topleft.x; i < point_image_bottomright.x; i++)
	{
		for (short j = point_image_topleft.y; j < point_image_bottomright.y; j++)
		{
			// high [ x | y || depth ] low
			// 解引用转换后的指针，读取该内存位置的 `uint32_t` 值（即浮点数的二进制表示直接当作整数处理，以支持|操作）。
			uint64_t key = i;
			key <<= 16;
			key |= j;
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);  // float32
			// 将生成的键和值写入输出数组
			gaussian_keys_unsorted[off] = key;          // 存储键
			gaussian_values_unsorted[off] = idx;        // 存储值（高斯模型的 ID）
			// 更新偏移量，指向下一个键值对的位置
			off ++;
		}
	}
}



__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	return glm::max(result, 0.0f);
}

__device__ glm::vec3 computeColorFromSH(int idx, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	result += 0.5f;

	return glm::max(result, 0.0f);
}



__global__ void identifyPixelRanges(
	int L, int width,          // 排序后的高斯模型列表的长度 num_renders
	uint64_t* point_list_keys, // 排序后的高斯模型键列表（每个键包含 深度信息 & pixel id）
	uint2* ranges              // 输出数组，存储每个 pixel 的工作范围（起始和结束位置）
)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// uint32_t key = point_list_keys[idx] & 0xFFFFFFFF;  // idx

	uint32_t key = point_list_keys[idx] >> 32; // point_list_keys[idx] & 0xFFFFFFFF;  // idx

	uint16_t x = key >> 16;
	uint16_t y = key & 0xFFFF;
	uint32_t currtile = (uint32_t)y * (uint32_t)width + (uint32_t)x;




	// 如果当前是列表的第一个元素，设置当前 tile 的起始位置为 0
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		// 提取前一个键的 tile ID
		// uint32_t prekey = point_list_keys[idx - 1] & 0xFFFFFFFF;  // idx

		uint32_t prekey = point_list_keys[idx - 1] >> 32; // &0xFFFFFFFF;  // idx

		uint16_t px = prekey >> 16;
		uint16_t py = prekey & 0xFFFF;
		uint32_t prevtile = (uint32_t)py * (uint32_t)width + (uint32_t)px;

		// 如果当前 tile ID 与前一个 tile ID 不同，说明当前元素是新的 tile 的起始位置
		if (currtile != prevtile)
		{
			// 设置前一个 tile 的结束位置为当前索引
			ranges[prevtile].y = idx;
			// 设置当前 tile 的起始位置为当前索引
			ranges[currtile].x = idx;
		}
	}
	// 如果当前是列表的最后一个元素，设置当前 tile 的结束位置为列表长度
	if (idx == L - 1)
		ranges[currtile].y = L;
}



__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
GETMAP(
	const uint2* ranges,
	const uint32_t* point_list,
	int P,
	const int M,
	int width,
	int height,
	const float* means3D,
	const float* normal,
	const float* scales,
	const float* rotations,
	const ushort2* pixels,
	const float* depths,
	const float* feature_vector,
	const float* sh,
	const float* campos,
	int* out_idx,
	float* out_color,
	float* out_depth,
	float* out_feature
) {
	// 获取当前线程的全局索引
	int x = blockIdx.x * blockDim.x + threadIdx.x;  // 全局 x 坐标
	int y = blockIdx.y * blockDim.y + threadIdx.y;  // 全局 y 坐标
	if (x >= width || y >= height)
		return;

	int pix_idx = y * width + x; // tile id

	// -1 表示没有命中点，并返回
	if (ranges[pix_idx].y == ranges[pix_idx].x) {
		out_idx[pix_idx] = -1;
		return;
	}
	int gaussian_idx = point_list[ranges[pix_idx].x]; // ranges[tile id].x = tile idx, point_list[tile idx] = point idx
	
	// ranges[pix_idx].y != ranges[pix_idx].x，则找最近的点point_list[ranges[pix_idx].x];
	// === begin 背面剔除 === // 
	glm::vec3 pos = ((glm::vec3*)means3D)[gaussian_idx];
	glm::vec3 dir = pos - *(glm::vec3*)campos;
	dir = dir / glm::length(dir);
	glm::vec3 nor = ((glm::vec3*)normal)[gaussian_idx];
	//if (glm::dot(dir, nor) >= 0) { // 最近的点可能是个背面，背面剔除
	//	out_idx[pix_idx] = -1;
	//	return;
	//}
	// === end 背面剔除 === // 
	
	out_idx[pix_idx] = gaussian_idx;
	out_feature[pix_idx * NUM_FEATURES] = depths[gaussian_idx];  // 0

	// 对方向向量进行位置编码
	float encoded_dir[POSITIONAL_ENCODING_DIMS];
	positional_encoding_3d(dir, encoded_dir);
	
	// 将位置编码结果存储到featuremap的1-24位置
	for (int i = 0; i < POSITIONAL_ENCODING_DIMS; i++) {
		out_feature[pix_idx * NUM_FEATURES + 1 + i] = encoded_dir[i];
	}
	
	// 将原始方向向量存储到colmap中
	out_color[pix_idx * 3 + 0] = dir.x;
	out_color[pix_idx * 3 + 1] = dir.y;
	out_color[pix_idx * 3 + 2] = dir.z;

	for (int i = 25; i < NUM_FEATURES; i++) {
		out_feature[pix_idx * NUM_FEATURES + i] = feature_vector[gaussian_idx * NUM_FEATURES + i];
	}
	out_depth[pix_idx] = depths[gaussian_idx];
	// for (int i = 0; i < 3; i++) {
	// 	out_color[pix_idx * 3 + i] = (normal[gaussian_idx * 3 + i] + 1) * 0.5f;
	// 	// out_color[pix_idx * 3 + i] = depths[gaussian_idx];
	// }
	//glm::vec3 res = computeColorFromSH(gaussian_idx, M,
	//	(glm::vec3*)means3D, *(glm::vec3*)campos, sh); // 这个定义在本函数上侧，而非forward中
	//for (int i = 0; i < 3; i++) {
	//	out_color[pix_idx * 3 + i] = res[i];
		//out_feature[pix_idx * NUM_FEATURES + i] = res[i];  // 0 1 2
		//out_feature[pix_idx * NUM_FEATURES + 3 + i] = scales[gaussian_idx * 3 + i]; // 3 4 5
		//out_feature[pix_idx * NUM_FEATURES + 6 + i] = rotations[gaussian_idx * 4 + i];  // 6 7 8
		//out_feature[pix_idx * NUM_FEATURES + 11 + i] = campos[i];  // 11 12 13
		//out_feature[pix_idx * NUM_FEATURES + 14 + i] = means3D[gaussian_idx * 3 + i];  // 14 15 16
	// }
	// out_feature[pix_idx * NUM_FEATURES + 9] = rotations[gaussian_idx * 4 + 3];  // 9

}


__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
CopyFeature(
	const int P,
	const int W,
	const int H,
	const float* means3D,
	const float* campos,
	int* out_idx,
	float* out_feature,
	float* dL_dfeature_vector
) {
	// 获取当前线程的全局索引
	int x = blockIdx.x * blockDim.x + threadIdx.x;  // 全局 x 坐标
	int y = blockIdx.y * blockDim.y + threadIdx.y;  // 全局 y 坐标
	if (x >= W || y >= H)
		return;


	int pix_idx = y * W + x;
	int gaussian_idx = out_idx[pix_idx];
	if (gaussian_idx < 0 || gaussian_idx >= P)
		return;


	for (int i = 25; i < NUM_FEATURES; i++) {
		dL_dfeature_vector[gaussian_idx * NUM_FEATURES + i] += out_feature[pix_idx * NUM_FEATURES + i];
	}
}



int CudaRasterizer::Rasterizer::getidxmap(
	std::function<char* (size_t)> idxBuffer,
	std::function<char* (size_t)> BinImgFuncBuffer,
	std::function<char* (size_t)> PixelBuffer,
	const int P,
	const int M,
	const int width,
	const int height,
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
	bool debug)
{
	// ### step1 : 分配空间，预处理 ###################
	// 计算焦距
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	// 计算 GeometryState 所需的内存大小，并分配内存
	size_t chunk_size = required<Idx2Img>(P);
	char* chunkptr = idxBuffer(chunk_size);
	// 从内存块初始化 GeometryState
	Idx2Img idxState = Idx2Img::fromChunk(chunkptr, P);
	// 定义 CUDA 线程块和网格的维度
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// 动态调整pixel辅助缓冲区的大小
	size_t pixel_chunk_size = required<PixelState>(width * height);
	char* pixel_chunkptr = PixelBuffer(pixel_chunk_size);
	PixelState pixelState = PixelState::fromChunk(pixel_chunkptr, width * height);


	CHECK_CUDA(RASTER::preprocess(
		P,
		means3D,
		viewmatrix, projmatrix,
		width, height,
		tan_fovx, tan_fovy,
		radii,
		idxState.pixels,
		idxState.img_xy,
		idxState.radii_float,
		idxState.depths, // 每个像素的深度
		idxState.touched  // 每个像素是否被碰到
	), debug)


	// ### step3 : 计算高斯模型覆盖的 tile 数量的前缀和 ########
		// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		idxState.scanning_space, // 临时存储空间指针
		idxState.scan_size,      // 临时存储空间大小
		idxState.touched,  // 输入数组：每个高斯点影响的区块数
		idxState.point_offsets,  // 输出数组：累计偏移量
		P), debug)

	// ### step4 : 获取渲染的高斯实例总数并调整辅助缓冲区大小 #####
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, idxState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	//return num_rendered;
	size_t binning_chunk_size = required<BinningImg>(num_rendered);  // 计算分箱缓冲区所需的大小
	char* binning_chunkptr = BinImgFuncBuffer(binning_chunk_size);      // 分配分箱缓冲区
	BinningImg binningimgState = BinningImg::fromChunk(binning_chunkptr, num_rendered);  // 初始化分箱状态



	//// ### step5 : 为每个渲染实例生成 [ tile | depth ] 键 #####
	duplicateIdxDepthImg << <(P + 255) / 256, 256 >> > (
		P,
		width, height,
		idxState.pixels,
		idxState.img_xy,
		idxState.radii_float,
		idxState.depths,
		idxState.point_offsets,
		binningimgState.point_list_keys_unsorted,
		binningimgState.point_list_unsorted,
		radii);

	// ### step6 : 对高斯索引列表按键进行排序 #####
	// 键 = h[ Depth | x | y ]l -> h[ x | y | Depth ]l
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningimgState.list_sorting_space,
		binningimgState.sorting_size,
		binningimgState.point_list_keys_unsorted, binningimgState.point_list_keys,
		binningimgState.point_list_unsorted, binningimgState.point_list,
		num_rendered, 0, 63), debug)


		// 初始化每个 tile 的工作范围
		CHECK_CUDA(cudaMemset(pixelState.ranges, 0, width * height * sizeof(uint2)), debug);
	if (num_rendered > 0)
		identifyPixelRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered, width,
			binningimgState.point_list_keys,
			pixelState.ranges);

	GETMAP << <tile_grid, block >> > (
		pixelState.ranges,
		binningimgState.point_list,
		P, M,
		width, height,
		means3D,
		normal,
		scales,
		rotations,
		idxState.pixels,
		idxState.depths,
		feature_vector,
		sh,
		campos,
		out_idx,
		out_color,
		out_depth,
		out_feature
		);
	return num_rendered;
}


void CudaRasterizer::Rasterizer::getidxmap_backward(
	const int P,
	const int W,
	const int H,
	const float* means3D,
	const float* campos,
	int* out_idx,
	float* out_feature,
	float* dL_dfeature_vector,
	bool debug)
{

	dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	CopyFeature << <tile_grid, block >> > (
		P, W, H,
		means3D,
		campos,
		out_idx,
		out_feature,
		dL_dfeature_vector
		);
}