"""统计形态特征：因果构造 + 逐项验证，不通过不放行进模型。

用法见 validate.py；所有特征在 features.REGISTRY 注册。
"""
from .features import REGISTRY, Spec, build_all

__all__ = ["REGISTRY", "Spec", "build_all"]
