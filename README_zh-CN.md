<div align="center">
<h1>COMP6704 Individual Assignment</h1>
<p><b>非负矩阵分解中的非凸优化</b><br/>
乘法更新 · 交替投影梯度 · HALS</p>
<p><a href="README.md">English</a> | <strong>中文</strong></p>
<p>LIU Xuyang<br/>
COMP6704 · Advanced Topics in Optimization · 2026</p>
</div>

题目：Nonconvex Optimization for Nonnegative Matrix Factorization。

比较普通乘法更新 MU、每因子一步的投影梯度 PG，以及每因子一轮坐标更新的 HALS。全部方法使用匹配初始化。

## 入口

- 代码仓库：<https://github.com/xuyang-liu16/COMP6704_Individual_Assignment>
- 主实验：`experiment.py`
- 补充分析：`analyze_results.py`

公开仓库提供数值复现代码、数据、保存结果、图和表。随附课程代码包另外包含
`main.tex` 与官方 NeurIPS 2025 样式，供重建报告使用。

## 实验内容

- 非凸性反例、梯度、KKT 条件、尺度歧义、连续投影梯度映射。
- 普通 MU、每因子一步的 PG、每因子一轮坐标更新的 HALS。
- 75 组主运行：5 种数据/秩设置 × 5 个初始化 × 3 种算法。
- 3 个零锁定对照、5 个 scikit-learn 坐标下降参考解。
- 同一批终止点的 H 行归一化敏感性、Lin 残差及初始化范围。
- 从原始曲线提取达到共同目标值的检查点，保留成功个数和条件中位数。
- 4 张图、3 张自动生成表格，以及由结果生成的正文数值宏。
- 有限差分、独立 NNLS 子问题、非负性、下降性、秩一解和零块回归检查。

`relative_pg` 是结果中的历史字段名，表示归一化目标的连续单位步长梯度映射相对范数，不是 Lin 的活动集投影梯度范数。

## 复现

保存结果的环境为 macOS / Apple M5 / Python 3.14.5。依赖版本列在 `requirements.txt`。

在独立虚拟环境中安装：

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify.py
.venv/bin/python experiment.py
.venv/bin/python analyze_results.py
.venv/bin/python make_report_assets.py
```

`experiment.py` 重新生成主实验和计时。`analyze_results.py` 使用保存的数据复跑 75 组终点，核对目标、残差、轮数和达标状态，但不覆盖主运行或计时文件；它另存 `results/diagnostics.json`。仅重建已有报告图表时，执行最后一条命令即可。

不同 Python、BLAS 或平台可能出现末位浮点、停止轮数和计时差异。补充分析按明确容差核对数值，并要求停止轮数一致；若触发断言，请先检查环境，不要跳过核验直接重建报告。没有执行全新环境安装验证。

在包含 `main.tex` 的课程代码包中，导出 PDF 可使用 Tectonic：

```bash
mkdir -p output/pdf
tectonic -X compile --untrusted --outdir output/pdf main.tex
```

也可用 TeX Live 的 `latexmk -pdf main.tex`。`main.tex` 是唯一编译入口。先生成图表和数值宏。

## 解释边界

停止阈值是 W 列单位范数规范下的相对映射 `1e-4`，上限为 2000 轮，每 10 轮检查。原协议达标数为 MU 0/25、PG 11/25、HALS 25/25。同一终点改用 H 行单位范数、并重算相应初始分母后，达标数为 0/25、2/25、7/25。这是终点敏感性分析，不是按新规则停机的实验。

Lin 残差与本文映射的零点相同，但小正坐标上的数值可能不同。SVD 值是取消非负约束后的下界；库求解器只给额外可行点。初始化范围描述固定矩阵上五个起点的差异，不代表数据抽样或数值扰动稳定性。

共同目标表使用各设置 MU 终值中位数作为事后目标，只对成功到达的运行计算轮数中位数，同时列出成功个数；它不是预注册的完整性能排名。

实验在导入 NumPy 前请求单线程，并对可识别线程池调用 `threadpoolctl`；已有环境变量不被覆盖。Apple Accelerate 实际线程数未独立验证。每个配置只有一次短计时，运行顺序固定；正文计时不能用于稳定速度排名。

## 文件说明

| 文件/目录 | 用途 |
|---|---|
| `experiment.py` | 数据、初始化、三算法、停止条件、计时与主实验 |
| `verify.py` | 数值检查与独立 SciPy NNLS 对照 |
| `analyze_results.py` | 复核 75 组终点、规范敏感性、Lin 残差、共同目标和初始化范围 |
| `make_report_assets.py` | 从 JSON 生成所有图表与关键正文数值 |
| `data/datasets.npz` | 本轮实际使用的数据和 digits 标签 |
| `results/runs.csv` / `runs.json` | 75 组主运行终值 |
| `results/curves.json` | 每十轮保存的目标、映射残差和累计时间 |
| `results/summary.json` | 五初始化中位数、范围、达标个数 |
| `results/metadata.json` | 数据哈希、环境、下界、对照实验和库参考 |
| `results/verification.json` | 数学与实现检查结果 |
| `results/diagnostics.json` | 补充分析及来源结果文件的 SHA-256 |
| `figures/` / `tables/` | 自动生成的图、LaTeX 表格与数值宏 |

数字图像数据来自 scikit-learn 内置 digits。文献入口见 PDF；源码包不附第三方论文全文。Lee 和 Seung 的 MU 论文采用 NeurIPS 官方会议年份 2000，部分文献按书籍出版年份写作 2001。

## 源码包范围

公开仓库包含中英文 README、四个 Python 文件、固定依赖、数据、保存结果和图表。
公开元数据将机器特有的动态库绝对路径替换为文件名，数值和计时保持原样。
课程代码包另附 TeX 入口、样式与报告。两者均不包含内部审核、备份或第三方论文。
