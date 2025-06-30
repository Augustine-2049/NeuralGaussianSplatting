from typing import NamedTuple
import torch.nn as nn
import torch

from . import _C

def rasterize_gaussians2(
    means3D,
    normal,
    means2D,
    feature_vector,
    sh,
    # opacities,
    scales,
    rotations,
    raster_settings,
):
    return _GetIdxMap2.apply(
        means3D,
        normal,
        means2D,
        feature_vector,
        sh,
        # opacities,
        scales,
        rotations,
        raster_settings,
    )


class _GetIdxMap2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        normal,
        means2D,
        feature_vector,
        sh,
        # opacities,
        scales,
        rotations,
        raster_settings,
    ):


        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            normal,
            scales,
            rotations,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            feature_vector,
            sh,
            raster_settings.campos,
            raster_settings.debug
        )

        num_rendered, idxmap, colmap, depthmap, featuremap = _C.get_idxmap2(*args)
        # 保留需要反向的tensors
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(means3D, normal, scales, rotations, feature_vector, sh, idxmap, featuremap)
        return idxmap, colmap, depthmap, featuremap

    @staticmethod
    def backward(ctx, _a, grad_out_color, _b, grad_featuremap):

        # # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        means3D, normal, scales, rotations, feature_vector, sh, idxmap, featuremap = ctx.saved_tensors

        # # Restructure args as C++ method expects them
        # args = (raster_settings.bg,
        #         means3D,
        #         radii,
        #         colors_precomp,
        #         scales,
        #         rotations,
        #         raster_settings.scale_modifier,
        #         cov3Ds_precomp,
        #         raster_settings.viewmatrix,
        #         raster_settings.projmatrix,
        #         raster_settings.tanfovx,
        #         raster_settings.tanfovy,
        #         grad_out_color,
        #         sh,
        #         raster_settings.sh_degree,
        #         raster_settings.campos,
        #         geomBuffer,
        #         num_rendered,
        #         binningBuffer,
        #         imgBuffer,
        #         raster_settings.debug)
        #
        # Compute gradients for relevant tensors by invoking backward method
        # 3. 调用 C++ 实现的梯度计算函数
        # grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(
        #     *args)
        # 4. 将梯度打包为元组
        grad_means3D = torch.zeros_like(means3D)
        grad_normal = torch.zeros_like(normal)
        grad_means2D = torch.zeros_like(means3D)
        grad_feature_vector = torch.zeros_like(feature_vector)
        grad_sh = torch.zeros_like(sh)
        # grad_opacities = torch.zeros_like(scales[..., 0])
        grad_scales = torch.zeros_like(scales)
        grad_rotations = torch.zeros_like(rotations)
        args = (
            means3D,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            idxmap,
            grad_featuremap,
            raster_settings.debug
        )
        grad_feature_vector = _C.get_idxmap_backward2(*args)[0]
        grads = (
            grad_means3D,  # 3D 高斯均值的梯度
            grad_normal,
            grad_means2D,  # 2D 投影均值的梯度
            grad_feature_vector,
            grad_sh,  # 球谐函数系数的梯度
            # grad_opacities,  # 不透明度的梯度
            grad_scales,  # 缩放因子的梯度
            grad_rotations,  # 旋转参数的梯度
            None,  # raster_settings 的梯度（不需要）
        )
        return grads

class GaussianRasterizationSettings2(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool


class GaussianRasterizer2(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, 
                normal,
                means2D,
                # opacities,
                feature_vector,
                shs=None,
                scales=None,
                rotations=None):
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians2(
            means3D,  # 3D坐标
            normal,
            means2D,  # 2D坐标 = screenspace_points 的引用
            feature_vector,
            shs,  # 颜色信息
            # opacities,  # 透明度
            scales,  # 尺寸
            rotations,  # 旋转轴
            self.raster_settings,
        )