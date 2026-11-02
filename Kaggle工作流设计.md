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
- 入口由 launcher 写入同目录的 `entry.txt` 决定（默认 `morph.kaggle_entry`），bootstrap 不再硬编码各项目差异，所以**只有一个 bootstrap 文件**供所有项目复用。

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
1. 按 `--project` 在**临时目录**生成 kernel：`拷贝 kaggle_bootstrap.py` + `写 entry.txt`（入口模块）+ `写 kernel-metadata.json`（slug/title/dataset_sources/`enable_gpu:false`/`enable_internet` 由模板定）；
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
- **enable_internet**：morph 需要（clone + 下载）；orderbook 关联网只用挂载 dataset 即可。
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
