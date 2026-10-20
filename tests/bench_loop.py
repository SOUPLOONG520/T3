"""内层循环速度剖析: 比较不同索引方式与 dtype。"""
import time
import numpy as np

R_MAX = 4096
M, m, n = 225, 15, 2


def bench(R, dtype, style, iters=3000):
    Q = np.zeros((R, n, M, m), dtype=dtype)
    Qflat = [Q[:, i].reshape(-1).copy() for i in range(n)] if style == "flat" else None
    rows = np.arange(R)
    state = np.random.randint(0, M, size=R)
    actions = np.random.randint(0, m, size=(R, n))
    pi = np.random.rand(R, n)
    target = np.random.rand(R)
    t0 = time.perf_counter()
    for t in range(iters):
        if style == "basic":
            for i in range(n):
                g = np.argmax(Q[rows, i, state, :], axis=1)
                cur = Q[rows, i, state, actions[:, i]]
                Q[rows, i, state, actions[:, i]] = 0.85 * cur + 0.15 * (pi[:, i] + 0.95 * target)
                ng = np.argmax(Q[rows, i, state, :], axis=1)
        elif style == "flat":
            ar = np.arange(m)
            for i in range(n):
                base = (rows * M + state) * m
                rowv = Qflat[i][base[:, None] + ar].reshape(R, m)
                g = rowv.argmax(1)
                cur = rowv[np.arange(R), actions[:, i]]
                newv = (0.85 * cur + 0.15 * (pi[:, i] + 0.95 * target)).astype(dtype)
                Qflat[i][base[:, None] + ar] = rowv  # no-op write keeps parity
                flatidx = base + actions[:, i]
                Qflat[i][flatidx] = newv
                rowv2 = Qflat[i][base[:, None] + ar].reshape(R, m)
                ng = rowv2.argmax(1)
        elif style == "take":
            ar = np.arange(m)
            for i in range(n):
                base = (rows * M + state) * m
                rowv = np.take(Qflat[i], base[:, None] + ar).reshape(R, m)
                g = rowv.argmax(1)
                cur = rowv[np.arange(R), actions[:, i]]
                Qflat[i][base + actions[:, i]] = (0.85 * cur + 0.15 * (pi[:, i] + 0.95 * target)).astype(dtype)
                rowv2 = np.take(Qflat[i], base[:, None] + ar).reshape(R, m)
                ng = rowv2.argmax(1)
    dt = time.perf_counter() - t0
    return iters / dt


if __name__ == "__main__":
    for style in ("basic", "flat", "take"):
        for R in (32, 128, 512, 2048):
            for dt_ in (np.float32, np.float64):
                rate = bench(R, dt_, style, iters=1500)
                print(f"style={style:5s} R={R:5d} dtype={np.dtype(dt_).name:8s} "
                      f"{rate:9,.0f} periods/s  {rate * R:12,.0f} row-periods/s")