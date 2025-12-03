# 复现笔记（结构化）

这份笔记力求“开箱即用”：镜像怎么构、数据怎么摆、脚本怎么跑，以及 RL 在本项目里的作用与注意事项。

## 数据准备 / 训练 / 评测

### 镜像与容器
- 构建：只 COPY 依赖清单，代码/数据运行时挂载。默认 CUDA 11.8 + torch 2.2.2/0.17.2，可用 build-arg 升级到 CUDA 12.4 + torch 2.4。文件：`docker/train.dockerfile`
  ```sh
  # CUDA 11.8
  docker build -f docker/train.dockerfile --network=host --progress=plain -t recogdrive-image:train .
  # CUDA 12.4 示例
  docker build -f docker/train.dockerfile \
    --build-arg CUDA_TAG=12.4.1-cudnn-devel-ubuntu22.04 \
    --build-arg PYTORCH_CUDA=cu124 \
    --build-arg TORCH_VERSION=2.4.1 \
    --build-arg TORCHVISION_VERSION=0.19.1 \
    --network=host --progress=plain -t recogdrive-image:train .
  ```
- 运行容器（示例路径按需替换）：
  ```sh
  docker run --name recogdrive-train -itd --gpus all --network host --shm-size=16g \
    -v /home/dzp/projects/recogdrive:/workspace/recogdrive \
    -v /data/navsim:/data/navsim \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    recogdrive-image:train
  docker exec -it recogdrive-train bash
  ```
- 容器内激活源码（依赖已装好，离线可禁用索引）：  
  `PIP_NO_INDEX=1 pip install -e . --no-deps`  
  如需重装 InternVL 依赖（联网时）：`pip install -r internvl_chat/internvl_chat.txt`

### 数据与权重路径约定
- **NAVSIM/OPENSCENE**：官方数据整理为  
  `navsim_logs/{trainval,navtrain,navtest,mini,private_test_e2e,…}`、`sensor_blobs/{同名 split}`、`maps`。建议宿主机放 `datasets/navsim`，容器挂到 `/workspace/recogdrive/datasets/navsim`（或 `/data/navsim`）。
  ```sh
  export NAVSIM_DEVKIT_ROOT=/workspace/recogdrive
  export OPENSCENE_DATA_ROOT=/workspace/recogdrive/datasets/navsim
  export NUPLAN_MAPS_ROOT=/workspace/recogdrive/datasets/navsim/maps
  export NAVSIM_EXP_ROOT=/workspace/recogdrive/exp
  export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
  ```
- **预训练 JSONL**：README 列出的 12+ 数据集（Navsim_Traj、DriveLM、LingoQA…）统一放 `datasets/pretrain_jsonl`，修改 `internvl_chat/shell/data_info/recogdrive_pretrain.json`：`annotation` 指向对应 JSONL，`root` 指向图片/传感数据根（NAVSIM 可用 `OPENSCENE_DATA_ROOT`，其它数据集指向各自解压目录）。
- **权重**：  
  - InternVL3 2B/8B（HuggingFace）；训练脚本的 `agent.vlm_path` 填本地权重目录或 HF 缓存。  
  - 官方 ReCogDrive VLM/IL/RL checkpoint（HuggingFace collection）；评测/增量训练的 `agent.checkpoint_path`、`agent.vlm_path` 指向这些目录。  
  - 隐状态缓存、度量缓存占用大（1–2 TB / 数百 GB），建议放 `/workspace/recogdrive/exp/...` 之类挂载卷。

### Stage 1：VLM 预训练（代码入口）
- 生成 NavSim QA/Traj：`scripts/generate_dataset/generate_internvl_dataset.sh` 或 `scripts/generate_dataset/generate_internvl_dataset_pipeline.sh`（pipeline 需先部署 vLLM/SGLang）。
- SFT：`internvl_chat/shell/internvl3.0/2nd_finetune/*.sh`，例如  
  ```sh
  cd internvl_chat
  sh shell/internvl3.0/2nd_finetune/internvl3_8b_dynamic_res_2nd_finetune_recogdrive_pretrain.sh
  ```
  产物：`ReCogDrive-VLM-*`，后续 `agent.vlm_path` 指向该目录。

### Stage 2：Diffusion Planner IL（代码入口）
- 缓存 hidden state（推荐提速）：`scripts/cache_dataset/run_caching_recogdrive_hidden_state.sh`（评估可用 `run_caching_recogdrive_hidden_state_eval.sh`）。
- 训练（2B 例）：`scripts/training/run_recogdrive_train_multi_node_2b.sh`（EMA 版 `_ema_2b.sh`，8B/单机可自行改 torchrun 参数）。
  需改：`NAVSIM_*`、`agent.vlm_path`、`cache_path`、`trainer.params.*`、`torchrun --nnodes/--nproc_per_node`。

### Stage 3：Diffusion Planner RL（DiffGRPO，代码入口）
- 度量缓存：`scripts/cache_dataset/run_metric_caching_train.sh`（navtrain）、`scripts/cache_dataset/run_metric_caching.sh`（navtest）；确保 NumPy ≥1.26.4。
- 训练（2B 例）：`scripts/training/run_recogdrive_train_multi_node_rl_2b.sh`  
  需改：`NAVSIM_*`、`agent.vlm_path`、`agent.metric_cache_path`、`agent.reference_policy_checkpoint`、`cache_path`。
  核心逻辑：`navsim/planning/script/run_training_recogdrive_rl.py`，agent 定义：`navsim/agents/recogdrive/recogdrive_agent.py`。

### 评测（PDM Score，代码入口）
```sh
CHECKPOINT=/workspace/recogdrive/weights/recogdrive.ckpt \
sh scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh
# 若不使用缓存：运行 *_no_hidden_state.sh 并将 agent.cache_hidden_state=False
```
必改项：`CHECKPOINT`、`agent.vlm_path`、`cache_path`、`NAVSIM_*`；输出日志在脚本内 `eval_*.txt`。

## 参考与经验总结

### RL 的作用与建议
- 定位：Stage 3 通过 RL（DiffGRPO）在 IL 基础上微调安全/舒适度（EP、DAC 等）；主入口 `navsim/planning/script/run_training_recogdrive_rl.py`，agent `navsim/agents/recogdrive/recogdrive_agent.py`。
- 常见收益/陷阱：奖励只用 PDMS + 低 BC 权重易 reward hacking（尾帧倒车，EC/EPDMS 掉）；采样 `deterministic` 设置影响波动。
- 实践要点：提高 BC 权重或改用 EPDMS 奖励；评测统一 `filter=False`；metric cache 必须 NumPy ≥1.26.4（或用官方 HF cache），navtrain 样本应为 85,109。
- 何时弱化 RL：资源紧或对平滑度要求不高时，IL 已能提供稳定基线；RL 更像“加分项”而非替代 IL。

### Issue 关键结论（复现/调参需注意）
- #10 Stage2/Stage3 分数偏低：RL metric cache 曾被 NumPy 1.23.* `linalg.inv` bug 污染；请用 NumPy ≥1.26.4（或 `OPENBLAS_CORETYPE=Haswell`）重建 cache，或用 HF `owl10/ReCogDrive_Metric_Cache`（改 metadata 路径）；navtrain 样本数应为 85,109。
- #23 RL 末点倒车、EPDMS 低：PDMS 奖励 + 低 BC 权重会 reward hacking；提高 BC 权重或改用 EPDMS 奖励，评测用 `filter=False`。
- #41 导航指令 one-hot 维度：NAVSIM 的 `high_command_one_hot` 实际 4 维（左/直/右/unknown，navsim issue #66），size=4 正常。

### 更多文档
- 安装：`docs/Installation.md`
- 训练/评估：`docs/Train_Eval.md`
- 讨论：#23 https://github.com/xiaomi-research/recogdrive/issues/23 ，#10 https://github.com/xiaomi-research/recogdrive/issues/10 ，#41 https://github.com/xiaomi-research/recogdrive/issues/41
