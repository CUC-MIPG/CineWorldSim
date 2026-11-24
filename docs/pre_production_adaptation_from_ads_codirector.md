# 基于 Co-Director 的 Pre-Production 适配方案（DirectorAgent / ShotAgent）

## 1. 目标与范围
- 目标：基于 `ads_codirector` 的机制，将其 **Pre-Production** 能力适配到当前项目的 `DirectorAgent` 和 `ShotAgent`。
- 范围：仅覆盖 Pre-Production 阶段，不改动 Production/Post-Production 工具实现。
- 产出：可执行的改造蓝图（模块映射、状态流、数据结构、提示词体系、验证回路、集成步骤）。

---

## 2. Co-Director 工作机制梳理

## 2.1 论文机制（2604.24842v1）
从论文第 3.1/3.2/3.3 节可归纳出三层机制：
- 分层多智能体：`Orchestrator -> Pre-Production -> Production -> Post-Production`
- 局部优化（self-refinement）：对中间产物（重点是 storyline、keyframe）做迭代反馈修正
- 全局优化（MAB）：在 creative strategy / narrative mode / aesthetic archetype 三维动作空间做探索-利用

论文中对 Pre-Production 的定义：
- `Creative Brief (B)`：用户约束扩展与语境化
- `Storyline (L)`：场景级叙事脚本
- `Visual Assets (V)`：缺失角色/视觉资产补齐
- `Storyboard (S)`：将 L 扩成可执行分镜（语义、镜头、时长、实体存在标记、音频指令）

## 2.2 代码机制（ads_codirector）
关键编排在：
- `.../ads_codirector/agent.py`

核心链路：
1. `user_assets_agent`：多模态资产摄取 + 语义角色标注（product/logo/character）
2. `parameters_agent`：将用户 brief 结构化
3. MAB 初始化 + 迭代（`mab_loop_agent`）
4. 每轮迭代内：
   - Creative direction 生成并扁平化注入
   - Pre-Production（creative brief -> storyline loop -> casting -> storyboard -> voiceover）
   - Production / Post-Production / Final verifier
5. 迭代日志落盘并更新 bandit

Pre-Production 的关键实现特征：
- 有明确的状态键规范（`common_utils.py`）
- Storyline 采用 loop + evaluator + checker 的本地闭环（阈值/重试上限）
- Storyboard 之前会注入 asset inventory，限制 filename 幻觉
- Storyboard 后有全局资产关联验证器（按 storyline 纠正 scene 的 assets）

---

## 3. 与当前 CineWorldSim 的差距

当前项目（`CineWorldSim`）已有：
- `DirectorAgent`：故事/角色/场景生成 + 参考图注入与补图
- `ShotAgent`：按 scene 递进生成 shot（含一定规则修复）
- `orchestrator`：LangGraph 分阶段执行

主要缺口：
- 缺少显式 `Creative Brief` 中间层
- 缺少 `Storyline` 局部评分-反馈-重写闭环
- 缺少严格的 `Asset Inventory` / `semantic_role` 约束
- 缺少 Storyboard 资产相关性 verifier（scene 级纠偏）
- 缺少 Creative Direction（MAB 输出）在 pre-production 提示词中的系统注入机制

---

## 4. 适配总设计（只改 DirectorAgent + ShotAgent）

## 4.1 模块映射（Co-Director -> CineWorldSim）
- `Creative Brief (B)` -> `DirectorAgent.generate_creative_brief()`
- `Storyline (L)` + refinement -> `DirectorAgent.generate_storyline_with_refinement()`
- `Visual Assets (V)` -> `DirectorAgent.ensure_character_images()/ensure_scene_images()` + 资产语义标注
- `Storyboard (S)`（scene 级）-> `DirectorAgent.build_storyboard_from_storyline()`
- `Storyboard -> Shot 展开` -> `ShotAgent.loop_storyboard_generation()`（增强为策略注入 + 约束生成）

## 4.2 状态键建议（对齐 Co-Director 思路）
在 `PipelineState` 新增（或统一）：
- `structured_constraints`
- `annotated_reference_visuals`
- `creative_configuration`
- `creative_direction`
- `creative_brief`
- `storyline`
- `refinement_history`
- `asset_inventory`

---

## 5. DirectorAgent 改造建议

## 5.1 新增阶段 A：结构化约束与资产注释入口
建议新增：
- `DirectorAgent.ingest_and_annotate_assets(reference_assets_json)`
  - 输出 `annotated_reference_visuals`
  - 每个资产至少包含：
    - `file_name`
    - `caption`
    - `semantic_role`（`product|logo|character|scene|unknown`）

## 5.2 新增阶段 B：Creative Brief 生成
建议新增：
- `DirectorAgent.generate_creative_brief(storyline_input, structured_constraints, annotated_reference_visuals, creative_direction)`

输出：
- 段落化 brief（可作为后续 Storyline 的单一上游）

## 5.3 新增阶段 C：Storyline 生成 + 自反思闭环
建议新增：
- `DirectorAgent.generate_storyline()`
- `DirectorAgent.evaluate_storyline()`
- `DirectorAgent.refine_storyline_loop(max_attempts, score_threshold)`

推荐评分维度（沿用 Co-Director）：
- hook_quality
- narrative_arc
- product_integration
- engagement
- prompt_adherence

停止条件：
- `score >= threshold` 或 `attempt >= max_attempts`

## 5.4 新增阶段 D：Scene Storyboard 生成（含资产约束）
将当前 scene 生成提示词改为：
- 显式输入 `asset_inventory`
- 禁止输出 inventory 外 filename
- 保留 scene 级：
  - topic / location / action
  - first-frame 描述
  - video motion 描述
  - duration 约束

## 5.5 新增阶段 E：Storyboard 资产相关性验证
建议新增 verifier：
- 输入：`storyline + storyboard + available_assets`
- 输出：
  - `missing_assets`
  - `invalid_filenames`
  - `assets_to_remove`
- 作用：在进入 ShotAgent 之前完成 scene 级资产纠偏

---

## 6. ShotAgent 改造建议

## 6.1 输入升级
在 `ShotAgent.loop_storyboard_generation()` 增加可选输入：
- `creative_direction`
- `asset_inventory`
- `annotated_reference_visuals`
- `scene_duration_target`

## 6.2 输出升级
每个 shot 除现有字段外建议增加：
- `duration_sec`
- `asset_bindings`（该 shot 绑定的资产 filename 列表）
- `continuity_notes`（角色一致性/服装/光线/机位连续性约束）

## 6.3 生成约束策略
- Scene Start/Mid/End 逻辑保留
- 强制与 scene-level storyline 对齐
- 对出现人物的 shot，必须绑定相应 character 资产（若存在）
- 为后续关键帧和视频代理提供“显式镜头运动语义”

---

## 7. Prompt Library 改造建议

当前 `prompt_library/default/prompt_query_template.py` 建议拆分新增：
- `Creative_Brief_Prompt`
- `Storyline_Generation_Prompt`
- `Storyline_Refinement_Prompt`
- `Storyline_Evaluator_Prompt`
- `Storyboard_With_Asset_Inventory_Prompt`
- `Storyboard_Asset_Verifier_Prompt`
- `Shot_Generation_With_Creative_Direction_Prompt`

并保留现有：
- `Scene_Creation_Prompt`
- `Shot_Creation_Prompt`
- `Story_design_Prompt`

---

## 8. 推荐执行流（替换现有 Pre-Production 子链）

在 `orchestrator` 的 `director -> shot` 之间增强为：
1. ingest assets（可选）
2. parse constraints（可选）
3. creative brief
4. storyline generation
5. storyline evaluator/refiner loop
6. scene storyboard generation
7. storyboard asset verifier
8. shot generation（ShotAgent）

---

## 9. 关键数据结构建议

建议新增 Pydantic（可放 `agents/schemas_preproduction.py`）：
- `CreativeBrief`
- `StorylineScene`, `Storyline`
- `StorylineEvaluation`（含 `breakdown + score + actionable_feedback`）
- `AnnotatedAsset`
- `ScenePlan`（scene 级 first-frame/video prompt + duration + assets）
- `PreProductionBundle`（B/L/S + logs）

---

## 10. 最小落地版本（MVP）与迭代顺序

Phase 1（先可用）：
- Director 新增 `storyline refinement loop`
- Storyboard 生成加 `asset_inventory` 强约束
- ShotAgent 接受 `creative_direction` 并输出 `duration_sec`

Phase 2（提升一致性）：
- 增加 storyboard asset verifier
- 增加 semantic-role 标注并贯通到 shot 绑定

Phase 3（向 Co-Director 靠拢）：
- 将 creative configuration 显式参数化（即使先不接 MAB）
- 引入可插拔全局优化器（后续可接 UCB/epsilon-greedy）

---

## 11. 验收标准
- 叙事一致性：storyline 评分达阈值后再下游
- 资产一致性：scene/shot 不出现 inventory 外 filename
- 角色一致性：同角色跨 shot 的 appearance/wardrobe 稳定
- 时长一致性：scene 与 shot 时长约束可闭合
- 可追溯性：保留 refinement history 与 verifier 纠偏记录

---

## 12. 直接可执行的实现清单（面向当前仓库）
- `agents/director_agent.py`
  - 新增 creative brief / storyline evaluator / refinement loop / storyboard verifier hook
- `agents/shot_agent.py`
  - 新增 creative_direction 注入、shot duration 与 asset_bindings 输出
- `agents/orchestrator.py`
  - 在 `director` 阶段内串起 pre-production 子步骤
- `prompt_library/default/prompt_query_template.py`
  - 新增 pre-production 专用模板键
- （可选）`agents/schemas.py`
  - 增补 pre-production 结构化类型

该设计保持你当前项目结构不变，仅通过“增强 Director/Shot + 增加 pre-production 状态与模板”完成适配，迁移成本最低。
