"""colorgrade 核心 API（初始未实现）。"""


class Grade:
    def white_balance(self, rgb, gains):
        """白平衡"""
        raise NotImplementedError

    def curve(self, v, gamma):
        """曲线"""
        raise NotImplementedError
