# meshsmooth

## 题面（可直接粘贴）

顶点法线平滑组。约束：按平滑角分组的硬边/软边正确；法线单位化；分组内法线相同；与暴力面法线平均对照（≤1e-6）；1 万面不超过 500 ms；非法输入报错。验收：cd meshsmooth && python -m pytest tests -q -k smooth。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd meshsmooth && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
