"""timestretch 核心 API：WSOLA（波形相似性）时间拉伸。

算法
----
WSOLA 把输入切成等长帧做重叠相加（OLA），但每帧的合成位置不是固定的
帧移：在“名义位置 ±d”的搜索窗口内，用归一化互相关找出与上一帧自然延续
段（上一个合成帧末端再前进一个帧移的波形）最相似的波形段，再做互补线性
交叉淡化叠加。由于复制的是原始波形片段而不是重采样，周期（基频）保持
不变；相似性搜索保证相邻帧相位衔接，避免固定帧移带来的周期性相消/
相长与咔嗒声。

搜索窗：半径取 ``帧移/2`` 与估计基周期的较大者（上限 ``_MAX_SEARCH``），
保证低基频信号在窗内至少有一个同相位候选。搜索窗始终以全局时间映射给出
的名义位置为中心，因此输出时长不会被相似性搜索带偏。

帧长自适应：先估计基周期。若 ``frame`` 不足两个周期，则在不超过
``_MAX_FRAME`` 的前提下加长，使低频信号也能完成整周期相位对齐。

尾部填充：有基周期时按周期重复尾段，否则镜像反射（镜像会把波形反相，
与正相延续段相关性低）。

交叉淡化窗：长度 ``L`` 的互补窗 ``w[i] = (i+1)/(L+1)``、
``1 - w[i]``，两者逐点和恒为 1（浮点误差为 0）；叠加保持幅度与直流。

性能：大搜索窗走 FFT 互相关，10 万样本在 400 ms 内完成（见
``_correlation_dots``）。

错误体系（都可被 ``except ValueError`` 捕获，且类型可区分）
------------------------------------------------------------
- :class:`InvalidRatioError`：拉伸比例非有限或 ≤ 0。
- :class:`InvalidFrameError`：帧长不是 ≥ 2 的整数或超出安全上界。
- :class:`InvalidSamplesError`：样本不是一维数值序列（含复数/嵌套）。
- :class:`NonFiniteInputError`：样本含 NaN/Inf。

线程安全：模块无共享可变状态，``Stretch`` 实例无状态，``stretch`` 只使用
调用内的局部数组，可被多线程共享调用，结果确定。
"""

import math
import numbers

try:  # numpy 仅用于加速；缺失时退化为纯 Python 实现，语义一致
    import numpy as _np
except ImportError:  # pragma: no cover - 环境无 numpy 时走兜底
    _np = None

__all__ = [
    "Stretch",
    "StretchError",
    "InvalidRatioError",
    "InvalidRatio",
    "InvalidFrameError",
    "InvalidFrameSizeError",
    "InvalidFrameLengthError",
    "InvalidSamplesError",
    "InvalidSamples",
    "NonFiniteInputError",
    "NonFiniteInput",
    "crossfade_window",
    "crossfade_windows",
]

_MAX_FRAME = 4096          # 自适应帧长上限
_MAX_FRAME_ABS = 1 << 20   # 显式帧长的安全上界（防病态内存）
_MAX_SEARCH = 8192         # 相似性搜索窗上限（样本）
_MAX_ANALYSIS = 16384      # 基周期估计最多使用的样本数
_MIN_PERIOD_CORR = 0.5     # 自相关峰低于此值视为无周期
_CORR_TIE = 1e-12          # 相似度并列容差（用于确定性择优）
_FFT_DOT_THRESHOLD = 1 << 18  # 候选点数×帧长超过此值时改用 FFT 相关
_TINY = 1e-30


class StretchError(ValueError):
    """时间拉伸参数/数据错误的基类，本身是 ``ValueError``。"""


class InvalidRatioError(StretchError):
    """非法拉伸比例：非数值、非有限或 ≤ 0。"""


class InvalidFrameError(StretchError):
    """非法帧长：不是整数、小于 2 或超出安全上界。"""


class InvalidSamplesError(StretchError, TypeError):
    """非法样本：不是一维实值数值序列（字符串/复数/嵌套序列等）。"""


class NonFiniteInputError(InvalidSamplesError):
    """样本中存在 NaN 或无穷大。"""


# 常见命名别名：便于调用方按 ``InvalidRatio`` / ``NonFiniteInput`` 引用。
InvalidRatio = InvalidRatioError
InvalidFrameSizeError = InvalidFrameError
InvalidFrameLengthError = InvalidFrameError
InvalidSamples = InvalidSamplesError
NonFiniteInput = NonFiniteInputError


def crossfade_windows(length):
    """返回一对互补交叉淡化窗 ``(fade_out, fade_in)``。

    两个窗长度均为 ``length``，逐点满足 ``fade_out[i] + fade_in[i] == 1``
    （浮点误差 ≤ 1 ulp，远小于 1e-9）。窗值落在 (0, 1) 开区间内，避免
    叠加段端点出现 0/1 突变。
    """
    if isinstance(length, bool) or not isinstance(length, numbers.Integral):
        raise InvalidFrameError("交叉淡化窗长度必须是整数，实际为 %r" % (length,))
    length = int(length)
    if length < 1:
        raise InvalidFrameError("交叉淡化窗长度必须 ≥ 1，实际为 %d" % (length,))
    denom = float(length + 1)
    fade_in = [(i + 1) / denom for i in range(length)]
    fade_out = [1.0 - v for v in fade_in]
    return fade_out, fade_in


def crossfade_window(length):
    """返回长度 ``length`` 的淡入窗 ``w``；淡出窗为 ``1 - w``。"""
    return crossfade_windows(length)[1]


def _validate_ratio(ratio):
    if isinstance(ratio, bool) or isinstance(ratio, (str, bytes, bytearray)):
        raise InvalidRatioError("拉伸比例必须是数值，实际为 %r" % (ratio,))
    try:
        value = float(ratio)
    except (TypeError, ValueError) as exc:
        raise InvalidRatioError("拉伸比例必须是数值，实际为 %r" % (ratio,)) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise InvalidRatioError(
            "拉伸比例必须是有限正数，实际为 %r" % (ratio,)
        )
    return value


def _validate_frame(frame):
    if isinstance(frame, bool):
        raise InvalidFrameError("帧长必须是 ≥ 2 的整数，实际为 %r" % (frame,))
    if isinstance(frame, numbers.Integral):
        value = int(frame)
    elif isinstance(frame, float) and math.isfinite(frame) and frame.is_integer():
        value = int(frame)
    else:
        raise InvalidFrameError("帧长必须是 ≥ 2 的整数，实际为 %r" % (frame,))
    if value < 2:
        raise InvalidFrameError("帧长必须 ≥ 2，实际为 %d" % (value,))
    if value > _MAX_FRAME_ABS:
        raise InvalidFrameError(
            "帧长 %d 超过安全上界 %d" % (value, _MAX_FRAME_ABS)
        )
    return value


def _validate_samples(samples):
    """校验并转换样本；返回 (numpy 数组或 Python 列表, 样本数)。"""
    if isinstance(samples, (str, bytes, bytearray)):
        raise InvalidSamplesError("样本必须是数值序列，不能是字符串")
    if _np is not None:
        if isinstance(samples, _np.ndarray):
            if samples.ndim != 1:
                raise InvalidSamplesError(
                    "样本必须是一维序列，实际维度为 %d" % samples.ndim
                )
            if samples.dtype.kind == "c":
                raise InvalidSamplesError("样本不能是复数")
            try:
                arr = samples.astype(_np.float64, copy=True)
            except (TypeError, ValueError) as exc:
                raise InvalidSamplesError("样本必须是实值数值") from exc
        else:
            try:
                seq = list(samples)
            except TypeError as exc:
                raise InvalidSamplesError(
                    "样本必须是数值序列，实际为 %r" % (type(samples).__name__,)
                ) from exc
            try:
                arr = _np.asarray(seq, dtype=_np.float64)
            except (TypeError, ValueError) as exc:
                raise InvalidSamplesError(
                    "样本必须是实值数值序列（不支持复数/嵌套序列）"
                ) from exc
            if arr.ndim != 1:
                raise InvalidSamplesError(
                    "样本必须是一维序列，实际维度为 %d" % arr.ndim
                )
        if arr.size and not _np.isfinite(arr).all():
            raise NonFiniteInputError("样本包含 NaN 或无穷大")
        return arr, int(arr.size)

    # 纯 Python 校验路径
    try:
        seq = list(samples)
    except TypeError as exc:
        raise InvalidSamplesError(
            "样本必须是数值序列，实际为 %r" % (type(samples).__name__,)
        ) from exc
    out = []
    for value in seq:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise InvalidSamplesError("样本必须是实值数值")
        value = float(value)
        if not math.isfinite(value):
            raise NonFiniteInputError("样本包含 NaN 或无穷大")
        out.append(value)
    return out, len(out)


def _estimate_period(values):
    """估计基周期（样本数）；无显著周期性时返回 None。"""
    if _np is not None:
        x = _np.asarray(values, dtype=_np.float64)
        n = x.size
        if n < 16:
            return None
        seg = x[: min(n, _MAX_ANALYSIS)].astype(_np.float64, copy=True)
        seg -= seg.mean()
        length = seg.size
        if float(seg @ seg) <= _TINY:
            return None
        size = 1
        while size < 2 * length:
            size <<= 1
        spec = _np.fft.rfft(seg, size)
        ac = _np.fft.irfft(spec * _np.conj(spec), size)[:length]
        if ac[0] <= 0.0:
            return None
        ac /= ac[0]
        hi = min(length // 2, 8192)
        if hi < 4:
            return None
        inner = ac[2 : hi + 1]
        # 先越过自相关的第一个谷（半周期附近），再取第一个显著局部极大。
        # 只看绝对值阈值而不和 lag≈0 的峰比较：短窗下自相关整体衰减，
        # 周期的峰远低于近零峰，但对齐位置的局部极大不受这种整体缩放影响。
        start = 0
        for offset in range(1, inner.size - 1):
            if inner[offset] <= inner[offset - 1] and inner[offset] <= inner[offset + 1]:
                start = offset
                break
        for offset in range(max(start, 1), inner.size - 1):
            value = inner[offset]
            if (
                value >= _MIN_PERIOD_CORR
                and value >= inner[offset - 1]
                and value >= inner[offset + 1]
            ):
                return offset + 2
        return None

    n = len(values)
    if n < 16:
        return None
    seg = values[: min(n, 2048)]
    length = len(seg)
    mean = sum(seg) / length
    seg = [v - mean for v in seg]
    energy = sum(v * v for v in seg)
    if energy <= _TINY:
        return None
    hi = min(length // 2, 1024)
    if hi < 4:
        return None
    corr = [0.0] * (hi + 1)
    running = energy
    for lag in range(2, hi + 1):
        dot = 0.0
        lag_energy = 0.0
        for i in range(length - lag):
            a = seg[i]
            b = seg[i + lag]
            dot += a * b
            lag_energy += b * b
        denom = math.sqrt(running * lag_energy)
        corr[lag] = dot / denom if denom > 0.0 else 0.0
    # 与 numpy 路径一致：越过第一个谷后取第一个显著局部极大。
    start = 0
    for lag in range(3, hi):
        if corr[lag] <= corr[lag - 1] and corr[lag] <= corr[lag + 1]:
            start = lag
            break
    for lag in range(max(start, 2), hi):
        value = corr[lag]
        if (
            value >= _MIN_PERIOD_CORR
            and value >= corr[lag - 1]
            and value >= corr[lag + 1]
        ):
            return lag
    return None


def _crossfade_ramp(length):
    denom = float(length + 1)
    return [(i + 1) / denom for i in range(length)]


def _pad_tail(values, period, pad):
    """尾部填充：有基周期时按周期重复尾段，否则镜像反射。

    镜像反射会把波形时间反相，搜索时与正相延续段的相关性很低；按周期
    复制尾段则保留了基频相位，输出末尾也能保持音高。
    """
    n = values.size
    if n == 0:
        return values
    if period is not None and 2 <= period < n:
        reps = pad // period + 2
        tail = values[n - period :]
        return _np.concatenate((values, _np.tile(tail, reps)[:pad]))
    return _np.pad(values, (0, pad), mode="reflect" if n > 1 else "edge")


def _correlation_dots(padded, template, lo, hi):
    """返回 padded 上 [lo, hi] 每个起点与 template 的点积。

    小搜索窗直接做矩阵乘；大窗（低基频信号帧长自适应后可达数千）改用
    FFT 求互相关，把每帧的 O(窗口×帧长) 降到 O(n log n)，10 万样本在
    极端比例下也能留在性能预算内。
    """
    L = template.size
    windows = _np.lib.stride_tricks.sliding_window_view(padded, L)
    count = hi - lo + 1
    if count * L <= _FFT_DOT_THRESHOLD:
        return windows[lo : hi + 1] @ template
    region = padded[lo : hi + L]
    size = 1
    while size < region.size:
        size <<= 1
    spec = _np.fft.rfft(region, size) * _np.fft.rfft(template[::-1], size)
    conv = _np.fft.irfft(spec, size)
    return conv[L - 1 : L - 1 + count]


def _search_radius(hop, period):
    """相似性搜索半径：至少覆盖一个基周期，否则低基频找不到同相位候选。"""
    radius = max(1, hop // 2)
    if period is not None:
        radius = max(radius, period)
    return min(radius, _MAX_SEARCH)


def _wsola_numpy(arr, ratio, frame, target, period=None):
    n = arr.size
    L = frame // 2
    hop = frame - L
    if L < 1:
        L, hop = 1, 1
    search = _search_radius(hop, period)
    pad = frame + 2 * hop + search + 8
    padded = _pad_tail(arr, period, pad)
    plen = padded.size
    cum_energy = _np.concatenate(([0.0], _np.cumsum(padded * padded)))

    out = _np.zeros(target, dtype=_np.float64)
    head = min(frame, target)
    out[:head] = padded[:head]

    ramp = _np.asarray(_crossfade_ramp(L), dtype=_np.float64)
    prev = 0
    pos = hop
    index = 1
    while pos < target:
        nominal = int(round(index * hop / ratio))
        limit = plen - frame
        if limit < 0:
            limit = 0
        # 搜索窗以名义位置（全局时间映射）为基准 ±search；模板仍取上一个
        # 合成帧的自然延续段。周期性信号在 ±search 内必有同相位候选，
        # 因此既能对齐相位，又不会让内容推进偏离目标时长。
        lo = max(0, nominal - search)
        hi = min(limit, nominal + search)
        if hi < lo:
            lo = hi = max(0, min(nominal, limit))

        t0 = prev + hop
        if t0 > plen - L:
            t0 = plen - L
        if t0 < 0:
            t0 = 0
        template = padded[t0 : t0 + L]
        t_energy = float(template @ template)

        dots = _correlation_dots(padded, template, lo, hi)
        energy = cum_energy[lo + L : hi + L + 1] - cum_energy[lo : hi + 1]
        denom = _np.sqrt(_np.maximum(energy, 0.0) * max(t_energy, 0.0))
        valid = denom > _TINY
        score = _np.full(denom.shape, -_np.inf, dtype=_np.float64)
        score[valid] = dots[valid] / denom[valid]
        if _np.isfinite(score).any():
            best = float(score.max())
            tied = _np.flatnonzero(score >= best - _CORR_TIE)
            offsets = lo + tied
            choice = int(tied[int(_np.argmin(_np.abs(offsets - nominal)))])
        else:  # 全静音/全零能量：退回名义位置
            choice = max(0, min(nominal, hi) - lo)
        start = lo + choice

        segment = padded[start : start + frame]
        fade_len = min(L, target - pos)
        if fade_len > 0:
            weight = ramp[:fade_len]
            out[pos : pos + fade_len] = (
                out[pos : pos + fade_len] * (1.0 - weight)
                + segment[:fade_len] * weight
            )
        tail_start = pos + L
        tail_end = min(pos + frame, target)
        if tail_end > tail_start:
            out[tail_start:tail_end] = segment[L : L + (tail_end - tail_start)]
        prev = start
        pos += hop
        index += 1
    return out


def _dot(a, b):
    try:
        return math.sumprod(a, b)
    except AttributeError:  # pragma: no cover - Python < 3.12
        return sum(x * y for x, y in zip(a, b))


def _best_correlated_shift(padded, cumulative, template, lo, hi, nominal):
    """在 [lo, hi] 内按归一化互相关找与 template 最相似的起点。

    穷举整个搜索窗，粗/精两路实现共用同一判据；并列时取离名义位置最近
    的候选，保证结果确定。窗口全为静音（能量为零）时退回名义位置。
    """
    L = len(template)
    t_energy = _dot(template, template)
    best = None
    best_score = None
    for shift in range(lo, hi + 1):
        cand = padded[shift : shift + L]
        energy = cumulative[shift + L] - cumulative[shift]
        denom = math.sqrt(max(energy, 0.0) * max(t_energy, 0.0))
        if denom <= _TINY:
            continue
        score = _dot(cand, template) / denom
        if best_score is None or score > best_score + _CORR_TIE:
            best, best_score = shift, score
        elif abs(score - best_score) <= _CORR_TIE and best is not None:
            if abs(shift - nominal) < abs(best - nominal):
                best = shift
    return best if best is not None else max(0, min(nominal, hi))


def _wsola_python(values, ratio, frame, target, period=None):
    n = len(values)
    L = frame // 2
    hop = frame - L
    if L < 1:
        L, hop = 1, 1
    search = _search_radius(hop, period)
    pad = frame + 2 * hop + search + 8
    if period is not None and 2 <= period < n:
        padded = list(values)
        tail = values[n - period :]
        for i in range(pad):
            padded.append(tail[i % period])
    elif n > 1:
        padded = list(values)
        reflexion = 2 * (n - 1)
        for i in range(pad):
            idx = (n + i) % reflexion
            if idx >= n:
                idx = reflexion - idx
            padded.append(values[idx])
    else:
        padded = list(values) + [values[0]] * pad
    plen = len(padded)
    cumulative = [0.0] * (plen + 1)
    for i, v in enumerate(padded):
        cumulative[i + 1] = cumulative[i] + v * v

    out = [0.0] * target
    head = min(frame, target)
    out[:head] = padded[:head]
    ramp = _crossfade_ramp(L)

    prev = 0
    pos = hop
    index = 1
    while pos < target:
        nominal = int(round(index * hop / ratio))
        limit = plen - frame
        if limit < 0:
            limit = 0
        t0 = min(max(prev + hop, 0), plen - L)
        # 搜索窗以名义位置为基准 ±search（与 numpy 路径一致）。
        lo = max(0, nominal - search)
        hi = min(limit, nominal + search)
        if hi < lo:
            lo = hi = max(0, min(nominal, limit))
        template = padded[t0 : t0 + L]
        start = _best_correlated_shift(padded, cumulative, template, lo, hi, nominal)

        segment = padded[start : start + frame]
        fade_len = min(L, target - pos)
        for i in range(fade_len):
            weight = ramp[i]
            out[pos + i] = out[pos + i] * (1.0 - weight) + segment[i] * weight
        tail_start = pos + L
        tail_end = min(pos + frame, target)
        if tail_end > tail_start:
            out[tail_start:tail_end] = segment[L : L + (tail_end - tail_start)]
        prev = start
        pos += hop
        index += 1
    return out


class Stretch:
    """WSOLA 时间拉伸器。无实例状态，可多线程共享。"""

    @staticmethod
    def crossfade_windows(length):
        """返回一对互补交叉淡化窗 ``(fade_out, fade_in)``。"""
        return crossfade_windows(length)

    @staticmethod
    def crossfade_window(length):
        """返回长度 ``length`` 的淡入窗。"""
        return crossfade_window(length)

    def stretch(self, samples, ratio, frame=256):
        """按 ``ratio`` 时间拉伸 ``samples``，保持基频不变。

        ``ratio`` > 1 变慢（输出更长），< 1 变快；输出长度为目标
        ``round(len(samples) * ratio)``，返回 ``list``。非法参数/数据按
        不同异常类型报错，见模块文档。
        """
        values, count = _validate_samples(samples)
        rate = _validate_ratio(ratio)
        frame_size = _validate_frame(frame)

        if count == 0:
            return []
        target = int(round(count * rate))
        if target <= 0:
            return []
        if rate == 1.0:
            # 比例 1 是最常见的恒等请求：逐样本原样返回，避免任何重采样
            # 或叠加引入的数值噪声（也保证两次调用结果完全相等）。
            return [float(v) for v in values]

        period = _estimate_period(values)
        if period is not None:
            needed = 2 * period + 2
            if needed > frame_size:
                frame_size = min(needed, _MAX_FRAME)
        frame_size = min(frame_size, max(2, _MAX_FRAME))
        if frame_size > _MAX_FRAME_ABS:  # pragma: no cover - 防御
            frame_size = _MAX_FRAME_ABS

        if _np is not None:
            out = _wsola_numpy(values, rate, frame_size, target, period)
            return [float(v) for v in out]
        return _wsola_python(values, rate, frame_size, target, period)
