# 本机运行说明（Windows / 沙箱环境专属）

> 这份文件记录的是**当前这台机器**的运行约束。正常环境下不需要看它。
> 仓库的通用运行方式见根目录 `README.md`。

## 1. 用哪个 Python

本机 **PyPI 不可访问**（`pypi.org` 连接被重置），因此不能 `pip install`。
DSH 自带的 Python 没有 matplotlib。请使用系统 Python：

```powershell
$py='C:\Users\91203\AppData\Local\Programs\Python\Python312\python.exe'
```

该解释器已含 numpy 1.26.4 / matplotlib 3.11.0 / networkx 3.6.1。
仓库里的 `requirements.txt` 只作声明用，本机不需要执行 `pip install`。

## 2. matplotlib 配置目录

沙箱禁止写 `AppData`，所以必须重定向 matplotlib 的配置目录，否则每次
import 都会报 `WinError 5`：

```powershell
$env:MPLCONFIGDIR='E:\DeepSeek工作产出\人工智能与经济管理研究课程\replication\.mplconfig'
```

该目录已在 `.gitignore` 中。

## 3. 进程池不可用（最关键的约束）

沙箱**禁止命名管道**，因此：

```python
multiprocessing.Pipe()          # PermissionError: [WinError 5]
ProcessPoolExecutor(...)        # 同上
```

`src/parallel.py` 的多进程分支在这台机器上会直接失败。两种应对：

1. **分片 + 独立顶层命令**（本仓库复现时采用的方式）：

   ```powershell
   foreach($k in 0..5){
     Start-Process -FilePath $py -ArgumentList @(
       'run_grid.py','--nx','20','--ny','20','--sessions','3',
       '--periods','2500000','--tag','main','--shard',"$k",'--nshard','6')
   }
   & $py run_grid.py --nx 20 --ny 20 --tag main --merge   # 合并分片并出图
   ```

   注意：这里的 `Start-Process` 必须配合 harness 的后台任务机制使用，
   否则子进程会随父 PowerShell 一起被回收。

2. **线程池**（`ThreadPoolExecutor` 可用），但本项目的内层循环以 numpy
   小数组运算为主，受 GIL 限制，实测加速比 <1，不推荐。

## 4. 吞吐与批量的关系

内层循环每周期有约 30 次 numpy 调用，**固定开销由批量大小摊薄**。实测单进程吞吐：

| 每次运行的会话数 R | 吞吐（行·期/秒） |
|---|---|
| 12 | 69,000 |
| 64 | 287,000 |
| 250 | 600,000 |
| 1000 | 793,000 |

结论：**宁可多开会话，不要把会话拆碎**。R ≥ 200 时性价比最好。

## 5. 中文控制台乱码

PowerShell 5.1 下中文输出常显示为乱码，这是控制台编码问题，
**不影响脚本结果与文件内容**。脚本内部的 JSON/CSV 一律用
`encoding="utf-8"` 读写。

## 6. 硬上限提醒

单会话需要 **200 万～400 万期**才收敛（β=4e-6 时 ε 降到 1e-3 需要约 69 万期）。
代表性实验（300 会话 × 300 万期上限）单进程约 **17 分钟**；
参数网格（400 点 × 3 会话 × 250 万期上限）按 6 分片并行约 **25 分钟**。