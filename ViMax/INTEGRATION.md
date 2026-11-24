# ViMax 基线集成说明

此目录是 ViMax 源码快照，作为普通文件夹纳入 CineWorldSim，而非 Git submodule。上游项目为 [HKUDS/ViMax](https://github.com/HKUDS/ViMax)，许可证见本目录 `LICENSE`。

- 导入来源：服务器 `/data1/AgentG/hhb/ViMax` 的 `experiment/pipeline-20261124` 分支。
- 导入提交：`7acf108`（包含 benchmark Pipeline、图像/视频后端适配、实验评估和测试）。
- 该分支所基于的上游提交：`fd4b72e`。
- 导入方式：从 `7acf108` 执行 `git archive`，因此保留版本化源码和 benchmark JSON，不含独立仓库的 `.git`、生成输出、缓存或本地 `.env`。

在 CineWorldSim 根目录运行 `scripts/run_joint_cameramaster_vimax_case.sh` 时，本目录是默认 ViMax 代码与 benchmark 根。已有独立 ViMax 运行结果仍可通过 `VIMAX_ROOT` 和 `VIMAX_OUTPUT_DIR` 显式引用。ViMax 自身使用 `Vimax` Conda 环境；相关实验协议见 `docs/pipeline_experiment_inventory.md` 和 `docs/vimax_pipeline_evaluation.md`。
