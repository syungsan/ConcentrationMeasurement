from typing import List, Tuple

def build_segments(times: List[float], window: float, step: float, min_presence_ratio: float) -> List[Tuple[float, float]]:
    """
    times: trackが検出された時刻リスト（秒）
    window/step: 秒
    min_presence_ratio: 窓内に存在している割合（簡易に検出回数で近似）
    """
    if not times:
        return []

    times = sorted(times)
    t_min, t_max = times[0], times[-1]
    segs: List[Tuple[float, float]] = []

    # 簡易に「窓内に一定回数以上検出」なら採用する
    # 必要なら密度推定などに差し替え可
    i0 = 0
    n = len(times)

    t = t_min
    while t + window <= t_max + 1e-6:
        t_start = t
        t_end = t + window

        # 窓内の検出数カウント（two pointers）
        while i0 < n and times[i0] < t_start:
            i0 += 1
        j = i0
        cnt = 0
        while j < n and times[j] <= t_end:
            cnt += 1
            j += 1

        # 期待検出数（ざっくり）… timesの平均間隔から推定
        # ここでは単純に「窓内にそこそこ検出がある」ことを要求する
        # presence_ratio は「窓内検出数 / (window*2fps相当)」の近似などにしてもよい
        # 今回は窓内検出数が一定以上ならOKとする
        # min_presence_ratio を使い “最低限” を窓長に比例させる
        min_cnt = max(1, int(window * 2.0 * min_presence_ratio))  # 2fps相当基準
        if cnt >= min_cnt:
            segs.append((t_start, t_end))

        t += step

    return segs
