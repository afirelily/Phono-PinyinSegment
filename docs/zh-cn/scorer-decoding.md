# 间隙评分与合法路径 MAP 解码

Phono-PinyinSegment 是一个评分器，而不是独立的贪心切分器。对于包含 `n` 个规范化字母的输入，模型为 `n - 1` 个字符间隙分别输出 logit `z_k`。一条合法切分路径 `Y` 为每个间隙指定二元状态 `y_k`：一表示音节边界，零表示音节延续。

令 `p_k = sigmoid(z_k)`，路径的 Bernoulli 对数概率为：

```text
log P(Y | X) = sum_k [y_k log p_k + (1-y_k) log(1-p_k)].
```

利用 `log p_k - log(1-p_k) = z_k`，可改写为：

```text
log P(Y | X) = sum_k log(1-p_k) + sum_k y_k z_k.
```

第一项只取决于输入和模型输出，与候选路径无关。因此仅就 MAP 解码而言：

```text
argmax_Y log P(Y | X) = argmax_Y sum_{k: y_k=1} z_k.
```

所以运行时只需在选中的边界上累加原始 logits。这是精确的 argmax 化简，而不是整条路径的校准概率。所有路径都对齐并覆盖同样的 `n - 1` 个间隙，因此全拼和简拼路径都不需要几何平均归一化或音节数惩罚。

拼音词表 Trie 枚举字符 DAG 中所有合法边 `(i, j)`。当 `j < n` 时，边贡献 `z_(j-1)`；到达输入末尾时贡献零。动态规划在每个字符偏移处保存最佳前缀路径，复杂度为 `O(E)`，其中 `E` 是 Trie 匹配边数。分数完全相同时优先选择音节更少的路径，再优先选择前部音节更长的路径，从而在通常情形不改变 MAP 结果的同时保证输出确定。

生产环境的 decoder、规范化、strict/safe 非法输入处理和逐字符修复位于 phono-core。本仓库的 `demo.py` 保留了一个紧凑的合法路径参考实现，用于验证评分模型 checkpoint。
