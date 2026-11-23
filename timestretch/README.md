# timestretch

## 题面（可直接粘贴）

OLA 时间拉伸。约束：拉伸后总时长与目标比误差 ≤1%；交叉淡化窗和为 1（≤1e-9）；不做变调（基频不变）；10 万样本不超过 400 ms；非法比例报错。验收：cd timestretch && python -m pytest tests -q -k stretch。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd timestretch && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
