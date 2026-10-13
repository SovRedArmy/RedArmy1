"""
剪枝（论文的改进幅度剪枝）：
评分 s = |θ| / P^λ（λ=prune_weight=0.5，P 为该层参数个数），
小层（深度卷积/输出头）评分相对更高从而被少剪；
全局长阈值剪掉 target_amount 个权重，之后由 hinerv_main 微调恢复。
"""
from .utils import *


# 可被剪枝的层类型
_prunning_target_cls = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _is_pruning_target(model, name, module):
    """判断模块是否为剪枝目标：位于 bitstream_prefix 内、不在 no_prune_prefix 内、类型匹配。"""
    return name.startswith(model.bitstream_prefix) and not name.startswith(model.no_prune_prefix) and isinstance(module, _prunning_target_cls)


class PruningMask(nn.Module):
    """
    Pruning mask.
    剪枝掩码参数化：forward 时返回 mask * x（True 保留、False 剪除）。
    通过 nn.utils.parametrize 挂在模块的 weight 上，训练时掩码参与前向。
    """
    def forward(self, x):
        if not hasattr(self, 'mask'):
            self.register_buffer('mask', torch.ones_like(x, dtype=torch.bool))
        return self.mask * x


def get_sparsity(model):
    """
    Compute sparsity of the model. Note that, only the parameters that can be pruned will be counted as zeros.
    统计模型稀疏度：zeros 为被剪掉的权重数，total 为可剪枝权重总数。
    """
    model = unwrap_model(model)
    zeros = 0
    total = 0
    with torch.no_grad():
        for k, v in model.named_modules():
            if k.startswith(model.bitstream_prefix):
                if isinstance(v, PruningMask):
                    zeros += (v.mask == 0.).sum().item()
                else:
                    total += sum([p.numel() for p in v.parameters(recurse=False)])
    return zeros, total


def init_pruning(args, logger, model):
    """
    Initialize the pruning masks.
    为所有剪枝目标层的 weight 注册 PruningMask 参数化（初始全 1，不剪任何权重）。
    """
    model = unwrap_model(model)
    with torch.no_grad():
        for k, v in model.named_modules():
            if _is_pruning_target(model, k, v):
                if args.debug:
                    logger.info(f'     Init pruning: {k}.weight')
                nn.utils.parametrize.register_parametrization(v, 'weight', PruningMask())


def set_pruning(args, logger, model, ratio, weight):
    """
    Set pruning for the model.
    执行一次全局幅度剪枝：
      1) 收集所有掩码与对应权重；
      2) 计算改进评分 |θ| / (P^λ)（P=该层剩余参数个数, λ=weight, 论文取 0.5），
         已剪除位置评分为 +inf 不会被再次选中；
      3) 全局排序取第 target_amount 小的值为阈值，低于阈值的置 0；
      4) 更新各层掩码并返回新稀疏度。
    """
    model = unwrap_model(model)
    with torch.no_grad():
        parameters_masks = []
        zeros, total = get_sparsity(model)
        target_amount = int(total * ratio)  # 本次要剪掉的权重总数

        if ratio == 0. or target_amount == zeros:
            return zeros, total

        for k, v in model.named_modules():
            if _is_pruning_target(model, k, v):
                if args.debug:
                    logger.info(f'     Update pruning: {k}.weight')
                for p in v.parametrizations['weight']:
                    if isinstance(p, PruningMask):
                        parameters_masks.append((v.weight, p.mask))

        # Compute new mask
        # 改进幅度评分：小层（P 小）分母小 → 评分高 → 更不容易被剪（论文的核心改进）
        scores = nn.utils.parameters_to_vector([abs(torch.where(m, v / (m.sum() ** weight + 1e-6), torch.inf)) for v, m in parameters_masks])
        masks = nn.utils.parameters_to_vector([m for _, m in parameters_masks])
        thres = torch.sort(scores)[0][target_amount - 1]

        # Update mask
        # 低于阈值（得分最小的 target_amount 个）被剪除
        masks = masks * (scores > thres)
        pointer = 0
        for _, m in parameters_masks:
            num_param = m.numel()
            m.copy_(masks[pointer:pointer + num_param].view_as(m))
            pointer += num_param

        # Compute and return the new sparsity
        zeros, total = get_sparsity(model)
        return zeros, total