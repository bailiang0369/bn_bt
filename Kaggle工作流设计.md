# Kaggle 跨项目工作流设计

> 目标：Kaggle 只当算力，GitHub 当代码源。kernel 只放一个极小 bootstrap，运行时把 GitHub 上的多文件项目 clone 下来、装依赖、调入口。再配一个本地 launcher 编排 push / 轮询 / 拉日志 / 解析，GPU 默认关闭。
> GitHub 仓库（公开）：https://github.com/bailiang0369/bn_bt

## 0. 动机与现状痛点
- Kaggle `script` 类型 kernel 只上传 `code_file` 那一个文件。之前被迫把 `download_klines` / `run_clustering` / `eval` / CVD 全 inline 进 `morph/kaggle_entry.py`，每次改一点都要重推整个巨无霸。
- 无法复用多文件结构（`tool/`、`orderbook/` 共享代码），跨项目（形态 / 订单簿）重复造轮子。
- 正解：**代码放 GitHub，Kaggle 只跑 bootstrap**；改代码只改 GitHub，kernel 重拉最新执行。

## 1. 总体架构
```
┌──────────────┐   git clone (enable_internet=true)   ┌──────────────────────────┐
│ GitHub       │ ───────────────────────────────────► │ Kaggle kernel (运行时)    │
│ bailiang0369 │                                     │  /kaggle/working/bn_bt    │
│ /bn_bt       │                                     │  + pip install 依赖       │
└──────────────┘                                     │  + import <ENTRY>; main() │
                                                      └──────────────────────────┘
本地 kaggle_launcher.py ── 生成 kernel 目录 / push / 轮询 / logs / 解析 ──► Kaggle API
数据：Kaggle datasets（eth-btc-1m-2020, btcusdt-orderbook-features）仍作为 input 挂载，不变。
```

## 2. 仓库目录结构
```
bn_bt/
├── data/                      # 大文件，.gitignore 排除，不进 GitHub
├── tool/pl_exp.py             # 共享工具
├── morph/                     # 项目A：faiss 形态聚类（真实项目代码，非 kaggle 专用）
│   ├── kaggle_entry.py        # 入口 main()
│   ├── download_klines.py / make_dataset/ / 形态聚类经验.md / ...
├── orderbook/                 # 项目B：订单簿 LightGBM（真实项目代码）
│   ├── train.py / extend_features.py / feature_spec.md / _cov.py
│   └── kaggle_ds/dataset-metadata.json   # 仅 orderbook 用的数据集元数据（非 kernel）
├── kaggle_bootstrap.py        # 【Kaggle 胶水】唯一被推上 Kaggle 的 code_file
├── kaggle_launcher.py         # 【Kaggle 胶水】本地编排器（不进 kernel）
├── requirements.txt           # 依赖清单，bootstrap 按此 pip install -r
├── .gitignore
└── 经验文档...
```
- `morph/` `orderbook/` 是纯项目代码，去掉了误导性的 `_kaggle` 后缀——它们只是"恰好跑在 Kaggle 上"，不是 kaggle 专用。
- **kernel-metadata.json 不再散落各项目文件夹**：它由 launcher 推 kernel 时按模板生成（见 §5），项目文件夹里只放纯代码。
- 只有 `kaggle_bootstrap.py` / `kaggle_launcher.py` 这两层胶水叫 `kaggle_*`，因为它们确实只服务于 Kaggle。

## 3. kaggle_bootstrap.py（作为每个 kernel 的 code_file）
```python
import os, sys, subprocess, importlib

REPO = "https://github.com/bailiang0369/bn_bt"
ROOT = "/kaggle/working/bn_bt"

def _entry():
    # launcher 推 kernel 时在同目录写入 entry.txt 指定入口；本地跑回退默认/环境变量
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "entry.txt")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    return os.environ.get("KAGGLE_ENTRY", "morph.kaggle_entry")

def main():
    if not os.path.exists(ROOT):
        # 公开库直连 clone；私有库用 Kaggle Secret 注入 GITHUB_TOKEN 拼进 URL：
        # REPO = f"https://{os.environ['GITHUB_TOKEN']}:x-oauth-basic@github.com/bailiang0369/bn_bt"
        subprocess.run(["git", "clone", "--depth=1", REPO, ROOT], check=True)
    else:
        subprocess.run(["git", "-C", ROOT, "pull", "--depth=1"], check=True)
    sys.path.insert(0, ROOT)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    os.path.join(ROOT, "requirements.txt")], check=True)
    importlib.import_module(_entry()).main()

if __name__ == "__main__":
    main()
```
要点：
- `enable_internet=true`（clone 必需；morph 还需联网下载，orderbook 关联网只用挂载 dataset）。
- 依赖统一从仓库根 `requirements.txt` 安装（`pip install -r requirements.txt`），不在 bootstrap 硬编码——faiss-cpu 等非 Kaggle 预装的库都列在其中，**改依赖只改这一处**。
- 私有库：用 Kaggle Secrets 存 `GITHUB_TOKEN`，bootstrap 拼进 URL；当前仓库已 public，无需 token。
- **Kaggle `script` kernel 只上传 `code_file` 一个文件**，旁路文件（如 entry.txt）不会被传上去。所以入口由 launcher 在暂存时**注入到 bootstrap 副本首行**（`KAGGLE_ENTRY=<entry>` 环境变量），bootstrap 从环境变量读取——这样入口随唯一上传的 code_file 一起生效，且**只有一个 bootstrap 源文件**供所有项目复用。

## 4. 各项目如何接入
| 项目 | ENTRY | slug（沿用） | 入口文件 |
|---|---|---|---|
| morph | `morph.kaggle_entry` | `btc-vs-eth-faiss-cpu-2020` | kaggle_entry.py:main |
| orderbook | `orderbook.train` | `btc-orderbook-lgbm` | train.py:main（已确认有 main()） |

- 新增项目 = 在 `kaggle_launcher.py` 的 `PROJECTS` 字典里加一项（slug/title/entry/dataset_sources/enable_internet），无需在各项目文件夹放任何 json，也无需改 bootstrap。

## 5. kaggle_launcher.py（本地编排器，不进 kernel）
命令示例：
```
python kaggle_launcher.py --project morph --push --wait --logs --parse
```
内部流程：
1. 按 `--project` 在**临时目录**生成 kernel：`生成 kaggle_bootstrap.py`（首行注入 `KAGGLE_ENTRY=<入口>`）+ `写 kernel-metadata.json`（slug/title/dataset_sources/`enable_gpu:false`/`enable_internet` 由模板定）；
2. `kaggle kernels push -p <tmp_dir>`；
3. 轮询 `kaggle kernels status <slug>` 至 COMPLETE/ERROR（每 30s）；
4. `kaggle kernels logs <slug>` 拉取；
5. `--parse` 时从日志提取关键指标（ACC=/AUC=/选择性/best_iteration 等）打印。
约束：默认 `enable_gpu=false`，只用 CPU（faiss-cpu / lightgbm CPU）；运行由你显式触发，不会像某些 IDE 插件偷偷占 Kaggle GPU 配额。

## 6. GitHub 推送（已完成）
- remote 已加 `origin` → `https://github.com/bailiang0369/bn_bt`，分支 `main`，已推送（本环境经 Git Credential Manager 可正常 push）。
- 日常：改代码 → `git commit` → `git push`；Kaggle kernel 重跑即自动 clone 最新。

## 7. 前提与风险
- **凭证**：本环境可正常 push（已验证）。若以后仓库设为私有，Kaggle 侧需 Secret 注入 `GITHUB_TOKEN` 供 kernel clone。
- **enable_internet**：**所有项目都必须为 true**——因为 bootstrap 第一步要 `git clone` GitHub；orderbook/fusion 除 clone 外不需要额外联网（数据仍走挂载 dataset）。
- **首次 clone 开销**：每次 kernel 启动多几十秒拉仓库，可接受。
- **数据 dataset 不变**：eth-btc-1m-2020 / btcusdt-orderbook-features 仍作 input 挂载；项目代码仓库与数据 dataset 分离。
- **.gitignore**：排除 `data/`、`__pycache__/`、`.codebuddy/`，不把大 parquet 推上 GitHub。

## 8. 已落地 / 待首次试运行验证
- [x] 仓库公开、代码已推 GitHub（main）
- [x] 目录重组为 `morph/` `orderbook/` + 根 `kaggle_bootstrap.py`/`kaggle_launcher.py`，移除冗余 metadata
- [x] `requirements.txt` 无版本锁（各库最新）
- [x] orderbook `train.py` 确认有 `main()`，且 `load_data()` 已 Kaggle `/kaggle/input` 自适应
- [ ] 首次 `--project morph --push --wait --logs --parse` 实跑，确认 clone/安装/入口链路通
- [ ] 首次 `--project orderbook` 实跑，确认 dataset 挂载路径与 `btcusdt-orderbook-features` 结构一致

---

## 9. 实际状态修订（2026-11-08，与上文冲突时以本节为准）

上文 §1~§5 是初版设计，后续迭代已变更多处。新窗口按本节操作即可。

### 9.1 代码链路
- `kaggle_bootstrap.py` 把仓库 clone 到 **`/tmp/bn_bt`**（不是 `/kaggle/working/bn_bt`：
  后者会把整个项目当成 kernel output，每次下载几十 MB）。
- **改代码必须先 `git commit` + `git push`，再推 kernel** —— kernel 里跑的是 GitHub 上的版本，
  本地未提交的改动不会生效（这个坑在本窗口踩过一次，白跑一轮）。
- `requirements.txt` 另加了 `zstandard`（原始订单簿 `.parquet.zst` 解压用）；
  torch 仍不装，用 Kaggle 镜像自带的 CPU 版。

### 9.2 launcher 命令
```powershell
python kaggle_launcher.py --project <项目名>                     # 默认 push + 轮询 + 拉日志
python kaggle_launcher.py --project <项目名> --no-wait --no-logs # 只推，稍后自己查
python kaggle_launcher.py --project <项目名> --no-wait --no-logs --set K=V --set K2=V2
```
- 参数是**反向开关**：没有 `--push` / `--wait` / `--logs`，只有 `--no-*` 用来关掉默认行为。
- `--set K=V` 可多次，临时覆盖 env，不必改 `PROJECTS` 配置就能跑 smoke（例如
  `--set MAX_HOURS=3 --set EXP=S0`）。
- `--parse` 从日志里过滤关键指标行。
- `stage()` 会注入 `KAGGLE_ENTRY` + 项目 env + secrets 到 bootstrap 副本首行，
  因为 script kernel 只上传 `code_file` 这一个文件。

### 9.3 密钥（绝不进仓库）
- 本地文件 `.codebuddy/settings.json`，已被 `.gitignore` 忽略。
- 项目配置里写 `secrets=["CHD_API_KEY"]`，`stage()` 从上述文件读出并注入 kernel env；
  **不要把密钥写进 `PROJECTS` 的 env**（那是公开仓库里的源码）。
- 当前存有 `CHD_API_KEY`（CryptoHFTData R2，取原始订单簿小时文件）。

### 9.4 数据下载通道（重要）
- `morph/ob_kline.py:fetch()` 取 GitHub Release 资产：
  - `browser_download_url`（Release 页面直链）**当前一律 504**（本地也 504，与 UA 无关）；
  - **asset API** `/repos/{owner}/{repo}/releases/assets/{id}` + `Accept: application/octet-stream`
    → 206 正常。已改为优先 API、直链兜底，并保留 Range 断点续传与 12 次重试（3s 起退避）。
  - 实测 393.6MB 整包约 13 秒；`OB_MONTHS=auto` 可改走月度分片（17~27MB/个）。
- `kaggle kernels logs` 返回的是 JSON 数组，用 `kaggle_launcher._decode_logs()` 解成纯文本。
- kernel status 显示 `COMPLETE` **不代表业务成功**：日志里可能已经 `SystemExit`，
  要看有没有 `DONE` 以及 traceback。

### 9.5 轮询
- 不要用一次长 `Start-Sleep`（工具会因无输出 idle timeout 被中断）。
- 用带输出的短循环，例如每分钟打印一次状态、命中 `COMPLETE|ERROR` 就 break：
```powershell
$S='<slug>'; for($i=0;$i -lt 15;$i++){ $o=(kaggle kernels status $S);
  Write-Output "[$i] $o"; if($o -match 'COMPLETE|ERROR'){break}; Start-Sleep -Seconds 60 }
```

### 9.6 算力与约定
- **GPU 默认关闭**（`enable_gpu:false`）。未经用户明确要求不要开 GPU、不要装 `faiss-gpu`。
- 日志里中文可能乱码，脚本内加 `sys.stdout.reconfigure(encoding="utf-8")`。
- Kaggle kernel 之间不共享 `/kaggle/working` 产物；要跨 kernel 复用数据得另外发布 Dataset。
