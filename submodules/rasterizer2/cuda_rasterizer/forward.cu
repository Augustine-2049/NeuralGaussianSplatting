#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
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

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// 调用 : computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix)
// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
    // 将 3D 点从世界空间变换到视图空间
	float3 t = transformPoint4x3(mean, viewmatrix);

    // 限制点的 x 和 y 坐标在视锥范围内
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

    // 计算雅可比矩阵 J，用于将 3D 协方差矩阵投影到 2D 屏幕空间
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

    // 提取视图矩阵的 3x3 线性部分（旋转和缩放）
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    // 计算变换矩阵 T = W * J
	glm::mat3 T = W * J;

    // 构建 3D 协方差矩阵 Vrk
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

    // 计算 2D 协方差矩阵：cov = T^T * Vrk^T * T
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
    // 应用低通滤波：确保每个高斯分布在屏幕空间中至少有一个像素的宽度/高度
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;

    // 返回 2D 协方差矩阵的上三角部分（3 个元素）
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,  // target : 屏幕空间半径 int * P
	float2* points_xy_image,  // target : 屏幕坐标 2float * P
	float* depths,  // target : 相机坐标系深度 float * P
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,  // target 4float * P
	const dim3 grid,  // [616, 409] -> [39, 26, 1]个tiled
	uint32_t* tiles_touched,  // target 1float * P
	bool prefiltered)
{
	// 每个线程处理一个gaussian，通过idx辨认
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// 初始化半径 + tiled是否可达
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// ### step1 : 视锥剔除
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// ### step2 : 高斯点3D投影到2D屏幕空间[-1, 1]
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// ### step3 : 计算 3D / [2D屏幕空间] 协方差 矩阵，求逆。接收协方差矩阵的上三角部分
	const float* cov3D;
	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
	cov3D = cov3Ds + idx * 6;
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// 2D 协方差矩阵求逆 (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// ### step4 : 通过特征值计算高斯点在屏幕空间的半径。计算高斯点覆盖的屏幕空间瓦片范围。如果范围为空，则直接返回。
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// [[-1,1], [-1,1]] -> [[0, W], [0, H]]
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	// grid = [616, 409] -> [39, 26, 1]个tiled
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// ### step5 : 通过球谐函数（SH）计算颜色。
	glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
	rgb[idx * C + 0] = result.x;
	rgb[idx * C + 1] = result.y;
	rgb[idx * C + 2] = result.z;

	// ### step6 : 处理后的数据存储到输出缓冲区中。
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// 主光栅化方法。每个线程块协作处理一个 tile，每个线程处理一个像素。
// 交替进行数据获取和光栅化。
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,              // 每个 tile 的高斯模型范围 [start, end)
    const uint32_t* __restrict__ point_list,       // 排序后的高斯模型索引列表
    int W, int H,                                  // 图像的宽度和高度
    const float2* __restrict__ points_xy_image,    // 高斯模型在图像空间中的 2D 位置
    const float* __restrict__ features,            // 高斯模型的特征（如颜色）
    const float4* __restrict__ conic_opacity,      // 高斯模型的圆锥不透明度（conic 矩阵 + 不透明度）
    float* __restrict__ final_T,                   // 每个像素的最终透射率 （transmittance）
    uint32_t* __restrict__ n_contrib,              // 每个像素的贡献高斯模型数量
    const float* __restrict__ bg_color,            // 背景颜色
    float* __restrict__ out_color                  // 输出的渲染颜色
    )
{
	// Identify current tile and associated min/max pixel range.
    // 获取当前线程块和线程的索引
	auto block = cg::this_thread_block();
    // 计算水平方向的 tile 数量
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    // 当前 tile 的最小和最大像素的范围[block.group_index().x, pix_min.x + BLOCK_X)
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };

	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };  // 该线程处理的像素坐标

	uint32_t pix_id = W * pix.y + pix.x;  // 该像素的全局索引
	float2 pixf = { (float)pix.x, (float)pix.y }; // 该像素的浮点坐标

	// Check if this thread is associated with a valid pixel or outside.
    // 检查当前线程是否处理有效的像素（在图像范围内）
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
    // 如果线程处理的像素无效，则标记为完成
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
    // 加载当前 tile 需要处理的高斯模型 ID 范围 [start, end)
    // tile id = block.group_index().y * horizontal_blocks + block.group_index().x
    // [tile id, depth]
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// rounds打包，表示256批量处理多少轮
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x; // tile id 相同的 不同depth对应不同高斯点的信息

	// Allocate storage for batches of collectively fetched data.
    // 为批量获取的数据分配共享内存
    // 高斯模型ID
	__shared__ int collected_id[BLOCK_SIZE];
	// 高斯模型的 2D 位置
	__shared__ float2 collected_xy[BLOCK_SIZE];
	// 高斯模型的圆锥不透明度
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;              // 透射率（初始为 1.0）
	uint32_t contributor = 0;    // 当前处理的贡献高斯模型数量
	uint32_t last_contributor = 0; // 最后一个贡献的高斯模型
	float C[CHANNELS] = { 0 };   // 当前像素的颜色累加值

	// Iterate over batches until all done or range is complete
    // 迭代处理每个批次，直到所有高斯模型处理完成或范围结束 rounds 个高斯点
    // 一个像素只对应一个tile id，会有多个高斯点。
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
        // 查看块中多少线程已经执行结束，如果都结束了，则完成光栅化
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
        // 协作从全局内存中批量获取高斯模型数据到共享内存
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress]; // 获取高斯模型 ID，也就是高斯点的idx
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id]; // 获取 2D 位置
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id]; // 获取圆锥 2D协方差 + 不透明度
		}
		// 同步线程块，保证256个线程将这个批次所需的信息都放在了共享内存中。
		// 并行读取，因为256个像素对应的一个tile，按tile对高斯点切分分类的
		block.sync();

		// Iterate over current batch
		// 迭代处理当前批次的高斯点
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			// 使用圆锥矩阵重新采样（参考 Zwicker 等人的 "Surface Splatting" 论文）
			float2 xy = collected_xy[j];  // 中心 2D 位置
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };  // 向量
			float4 con_o = collected_conic_opacity[j];  // 圆锥 2D协方差 + 不透明度
			// **马氏距离（Mahalanobis Distance）** 的简化计算，用于衡量 **当前像素（`pixf`）相对于高斯点椭圆（由 `con_o` 定义）的位置**。
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			// **像素距离高斯点中心太远**（超出 1σ 范围），可以跳过该高斯点的贡献。
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
            // 3D 高斯泼溅论文中的公式 (2)
            // 通过高斯不透明度和指数衰减计算 alpha
            // 避免数值不稳定（参考论文附录）
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) // 基本透明，不用管这个点
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) // 透射率过低，该像素点完成
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
            // 3D 高斯泼溅论文中的公式 (3)
            // 累加当前高斯模型对像素颜色的贡献
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	// 所有处理有效像素的线程将最终的渲染数据写入帧缓冲区和辅助缓冲区
	if (inside)
	{
		final_T[pix_id] = T; // 该像素 有贡献的 最终透射率
		n_contrib[pix_id] = last_contributor; // 该像素 有贡献的 高斯模型数量
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // 写入最终颜色
	}
}

void FORWARD::render(
	const dim3 grid,  // [616, 409] -> [39, 26, 1]个tiled
	dim3 block, // [16, 16]
	const uint2* ranges,
	const uint32_t* point_list,  // 存储的排序后 高斯点 在[0, P - 1] 上的索引值
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	// 逐像素处理
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,  // 每个tile对应的 point_list_keys 上的范围 [start, end)
		point_list,
		// 下标：num_rendered，值：存储的排序后 高斯点 在[0, P - 1] 上的索引值
		W, H,
		means2D,  // 每个高斯点在像素平面上的像素坐标
		colors,
		conic_opacity, // 2D协方差逆矩阵 + 不透明度
		final_T,       // 透明度累计
		n_contrib,    // 高斯贡献数量
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,  // target : 屏幕空间半径 int * P
	float2* means2D,  // target : 屏幕坐标 2float * P
	float* depths,  // target : 相机坐标系深度 float * P
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,  // target 4float * P
	const dim3 grid,// [616, 409] -> [39, 26, 1]个tiled
	uint32_t* tiles_touched,  // target 1float * P
	bool prefiltered)
{
	// NUM_CHANNELS=3在全局变量中声明
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,  // target : 屏幕空间半径 int * P
		means2D,  // target : 屏幕坐标 2float * P
		depths,  // target : 相机坐标系深度 float * P
		cov3Ds,
		rgb,
		conic_opacity,  // target 4float * P
		grid,  // [616, 409] -> [39, 26, 1]个tiled
		tiles_touched,  // target 1float * P
		prefiltered
		);
}