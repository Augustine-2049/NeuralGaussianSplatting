#include "raster.h"
#include "auxiliary.h"
#include "aux_new.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template<int C>
__global__ void preprocessCUDA(
	const int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, 
	const int H,
	const float tan_fovx, 
	const float tan_fovy,
	int* radii,
	float S,
	float2* img_xy,
	float* radii_float,
	ushort2* pixels,  // target : 屏幕坐标 2short * P
	float* depths,  // target : 相机坐标系深度 float * P
	uint32_t* touched
	)
{
	// 每个线程处理一个gaussian，通过idx辨认
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// 初始化半径 == 是否可达
	radii[idx] = 0;
	touched[idx] = 0;

	// ### step1 : 视锥剔除 + 高斯点3D投影到2D屏幕空间[-1, 1]
	// ps : 融合原步骤1 + 2，去掉in_frustum冗余操作。
	float3 p_view;
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);
	if (p_view.z <= 0.2f) return;

	// ### step2 : 根据点的深度，修改idx
	int2 point_image = { (int)ndc2Pix(p_proj.x, W), (int)ndc2Pix(p_proj.y, H) };
	radii_float[idx] = S / p_view.z;  // 0.5 - 6 都有
	img_xy[idx] = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	int2 point_image_topleft = { (int)(fmaxf(0.0f, img_xy[idx].x - radii_float[idx])), (int)(fmaxf(0.0f, img_xy[idx].y - radii_float[idx])) };
	int2 point_image_bottomright = { (int)(fminf((float)W, img_xy[idx].x + radii_float[idx] + 1.0f)), (int)(fminf((float)H, img_xy[idx].y + radii_float[idx] + 1.0f)) };
	if (point_image.x < 0 || point_image.x >= W || point_image.y < 0 || point_image.y >= H)
		return;
	// ### step3 : 处理后的数据存储到输出缓冲区中。
	depths[idx] = p_view.z;
	pixels[idx].x = point_image.x;
	pixels[idx].y = point_image.y;
	radii[idx] = 1;
	touched[idx] = (point_image_bottomright.x - point_image_topleft.x) * (point_image_bottomright.y - point_image_topleft.y);

}

void RASTER::preprocess(
	const int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, 
	const int H,
	const float tan_fovx, 
	const float tan_fovy,
	int* radii,
	ushort2* pixels,  // target : 屏幕坐标 2float * P
	float2* img_xy,
	float* radii_float,
	float* depths,
	uint32_t* touched
)  // target : 相机坐标系深度 float * P
{
	// NUM_CHANNELS=3在全局变量中声明
	float S = 3.0f;
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		orig_points,
		viewmatrix,
		projmatrix,
		W, H,
		tan_fovx, tan_fovy,
		radii,
		S,
		img_xy,
		radii_float,
		pixels,  // target : 屏幕坐标 2float * P
		depths,  // target : 相机坐标系深度 float * P
		touched
		);
}