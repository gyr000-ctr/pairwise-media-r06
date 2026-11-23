# mipblend

## 题面（可直接粘贴）

mip 三线性混合。约束：两级权重和为 1 且级间过渡连续（无跳变）；LOD 与各向异性比一致（log2，≤1e-6）；边界钳制；10 万次不超过 100 ms；非法参数报错。验收：cd mipblend && python -m pytest tests -q -k mip。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd mipblend && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
