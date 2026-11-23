# colorgrade

## 题面（可直接粘贴）

调色（白平衡 + 曲线）。约束：白平衡增益归一（灰轴不动，≤1e-6）；曲线单调且端点固定；与逐像素解析式一致（≤1e-9）；10 万像素不超过 400 ms；非法参数报错。验收：cd colorgrade && python -m pytest tests -q -k grade。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd colorgrade && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
