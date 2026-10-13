from utils import *
from models.layers import *


class DeterministicContext:
    """
    Deterministic context（确定性上下文：编解码时关闭 cudnn benchmark、启用确定性算法，
    保证 compress/decompress 两次运行产生的比特完全一致）
    """
    def __enter__(self):
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def __exit__(self, *args):
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


class ParamModel(nn.Module):
    """
    ParamModel（参数模型：把一组张量包装成 nn.ParameterDict，
    使得 INR 权重与各级熵模型参数可以被统一地量化、优化与 load/save）
    """
    def __init__(self, logger, params, requires_grad=True):
        super().__init__()
        # Params
        self.params = nn.ParameterDict(params)

        # Set requires_grad
        self.set_param_requires_grad(requires_grad)

    def set_param_requires_grad(self, requires_grad):
        """（统一开关所有参数的梯度）"""
        for param in self.params.values():
            if param is not None:
                param.requires_grad = requires_grad

    def load(self, params):
        """（用解码后的参数覆盖当前参数，实现“把码流写回模型”）"""
        self.params.load_state_dict(params)

    def forward(self):
        """（返回参数字典本身）"""
        return self.params