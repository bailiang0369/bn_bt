# Kaggle 跨项目工作流设计（设计稿，确认后实现）

> 目标：Kaggle 只当算力，GitHub 当代码源。kernel 改成"极小 bootstrap"，运行时把 GitHub 上的多文件项目 clone 下来、装依赖、调入口。再配一个本地 launcher 编排 push / 轮询 / 拉日志 / 解析，GPU 默认关闭。
> GitHub 仓库（已建）：https://github.com/bailiang0369/bn_bt

## 0. 动机与现状痛点
- Kaggle `script` 类型 kernel 只上传 `code_file` 那一个文件。当前为把逻辑跑起来，被迫把 `download_klines` / `run_clustering` / `eval` / CVD 全 inline 进 `morph_kaggle/kaggle_entry.py`，每次改一点都要重推整个巨无霸。
- 无法复用多文件结构（`tool/`、`orderbook_kaggle/` 共享代码），跨项目（形态 / 订单簿）重复造轮子。
- 正解：**代码放 GitHub，Kaggle 只跑 bootstrap**；改代码只改 GitHub，kernel 重拉最新执行。

## 1. 总体架构
```
┌──────────────┐   git clone (enable_internet=true)   ┌──────────────────────────┐
│ GitHub       │ ───────────────────────────────────► │ Kaggle kernel (运行时)    │
│ bailiang0369 │                                     │  /kaggle/working/bn_bt    │
│ /bn_bt       │                                     │  + pip install 依赖       │
└──────────────┘                                     │  + import <ENTRY>; main() │
                                                      └──────────────────────────┘
本地 kaggle_launcher.py ── push / 轮询 status / 拉 logs / 解析 ──► Kaggle API
数据：Kaggle datasets（eth-btc-1m-2020, btcusdt-orderbook-features）仍作为 input 挂载，不变。
```

## 2. 仓库目录结构（GitHub 侧 = 当前本地仓库，基本不动）
```
bn_bt/
├── data/                      # 大文件，.gitignore 排除，不进 GitHub
├── tool/pl_exp.py             # 共享工具
├── morph_kaggle/              # 项目A：faiss 形态聚类（含 kaggle_entry.py 入口）
│   └── kernel-metadata.json   # slug=btc-vs-eth-faiss-cpu-2020
├── orderbook_kaggle/          # 项目B：订单簿 LightGBM（含 train.py 入口）
│   └── kernel-metadata.json   # slug=btc-orderbook-lgbm
├── kaggle_bootstrap.py        # 【新增】kernel 的 code_file，极小，与项目解耦
├── kaggle_launcher.py         # 【新增】本地编排器（不进 kernel）
├── requirements.txt           # 【新增】依赖清单，bootstrap 按此 pip install -r
├── .gitignore                 # 已排除 data/ __pycache__/ .codebuddy/
└── 经验文档...
```
- 每个项目自带 `kernel-metadata.json`（slug/title/ENTRY 不同），由 launcher 按模板生成/校验。
- `kaggle_bootstrap.py` 只负责"拉项目 + 装依赖 + 按 ENTRY 调入口"。

## 3. kaggle_bootstrap.py（作为每个 kernel 的 code_file）
```python
import os, sys, subprocess, importlib
REPO  = "https://github.com/bailiang0369/bn_bt"
ROOT  = "/kaggle/working/bn_bt"
ENTRY = os.environ.get("KAGGLE_ENTRY", "morph_kaggle.kaggle_entry")

if not os.path.exists(ROOT):
    # 公开库可直接 clone；私有库用 Kaggle Secret 注入的 GITHUB_TOKEN：
    # REPO = f"https://{os.environ['GITHUB_TOKEN']}:x-oauth-basic@github.com/bailiang0369/bn_bt"
    subprocess.run(["git", "clone", "--depth=1", REPO, ROOT], check=True)

sys.path.insert(0, ROOT)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(ROOT, "requirements.txt")], check=True)
importlib.import_module(ENTRY).main()
```
要点：
- `enable_internet=true`（clone 必需）。
- 依赖统一从仓库根 `requirements.txt` 安装（`pip install -r requirements.txt`），不在 bootstrap 硬编码——faiss-cpu 等非 Kaggle 预装的库都列在其中，**改依赖只改这一处**。
- 私有库：用 Kaggle Secrets 存 `GITHUB_TOKEN`，bootstrap 拼进 URL；公开库无需 token。
- ENTRY 决定跑哪个项目，可由 kernel-metadata 的环境变量或每个项目一个小变体设定。

## 4. 各项目如何接入
| 项目 | ENTRY | slug（建议） | 入口文件 |
|---|---|---|---|
| morph_kaggle | `morph_kaggle.kaggle_entry` | `btc-vs-eth-faiss-cpu-2020`（沿用） | kaggle_entry.py:main |
| orderbook_kaggle | `orderbook_kaggle.train` | `btc-orderbook-lgbm` | train.py（需确保可 `import train; train.main()`） |

- 新增项目 = 复制一份 kernel-metadata 改 slug/title/ENTRY，无需改 bootstrap。
- orderbook 的 train.py 当前是否有 `main()` 可供 `import train; train.main()` 调用，需确认（若执行逻辑只在 `if __name__=="__main__"` 下，则需抽成 main() 或让 bootstrap 直接执行模块）。

## 5. kaggle_launcher.py（本地编排器，不进 kernel）
命令示例：
```
python kaggle_launcher.py --project morph --push --wait --logs --parse
```
内部流程（复用我们已手敲的轮询/解析逻辑，固化）：
1. 按 `--project` 取对应 kernel-metadata 模板，写入/校验 `enable_gpu:false`、`dataset_sources` 等；
2. `kaggle kernels push -p <project_dir>`；
3. 轮询 `kaggle kernels status <slug>` 至 COMPLETE/ERROR（每 30s）；
4. `kaggle kernels logs <slug>` 拉取；
5. 解析关键指标（覆盖率/准确率）打印。
约束：默认 `enable_gpu=false`，绝不自动开 GPU；运行由你显式触发，不会像某些 IDE 插件偷偷占 Kaggle GPU 配额。

## 6. GitHub 推送步骤（实现阶段，需你最终确认）
1. `git remote add origin https://github.com/bailiang0369/bn_bt`
2. `git push -u origin master`（本环境需有推送凭证：`gh auth login` 或配置 PAT / 凭证助手；否则我无法推送，需你提供或你自己推）
3. 之后日常：改代码 → `git commit` → `git push`；Kaggle kernel 重跑即自动 clone 最新。
- 注：本地分支为 `master`，GitHub 默认可能是 `main`，如冲突可 `git branch -M main` 再推。

## 7. 前提与风险
- **凭证**：本环境能否向该 GitHub push 待验证（gh 登录 / PAT）。私有库还需在 Kaggle 侧用 Secret 注入 token 供 kernel clone。
- **enable_internet=true**：clone 必需，已在 bootstrap 设计里体现。
- **首次 clone 开销**：每次 kernel 启动多几十秒拉仓库，可接受。
- **数据 dataset 不变**：eth-btc-1m-2020 / btcusdt-orderbook-features 仍作 input 挂载；项目代码仓库与数据 dataset 分离。
- **.gitignore**：已排除 `data/`、`__pycache__/`、`.codebuddy/`，确保不把大 parquet 推上 GitHub。

## 8. 待确认后实现
- [ ] 仓库是否公开（决定 kernel clone 要不要 token）
- [ ] 本环境能否 push（先 `git ls-remote` / `gh auth status` 验证）
- [ ] launcher 是否现在一并实现，还是先只做 bootstrap + 手动 push
- [ ] orderbook train.py 入口守卫确认（能否 `train.main()`）
