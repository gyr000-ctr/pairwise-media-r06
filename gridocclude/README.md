# gridocclude

## 题面（可直接粘贴）

网格视线遮挡。约束：可见集合与逐格 Bresenham 对照一致；对称性（A 见 B 则 B 见 A）；遮挡体不透明；512x512 不超过 500 ms；非法网格报错；并发安全。验收：cd gridocclude && python -m pytest tests -q -k los。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd gridocclude && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
