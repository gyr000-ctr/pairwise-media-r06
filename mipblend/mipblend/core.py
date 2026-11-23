"""mipblend 核心 API（初始未实现）。"""


class MipBlend:
    def blend(self, scale_x, scale_y):
        """返回 (level, frac)"""
        raise NotImplementedError

    def sample(self, l0, l1, f):
        """混合"""
        raise NotImplementedError
