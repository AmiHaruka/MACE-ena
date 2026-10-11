# EnvHead v4

近场 ENV 的元素、坐标、电荷和排列逐层参与 MACE QM 表示，远场电势/电场提供固定维度条件化。正常预测只执行一次主干和一次电荷头：

\[
E=U_\theta(h^{cond})+E_{ele},\qquad E_{ele}=\sum_iq_i\phi_i^{all}.
\]

`energy` 为最终 QM 能量，`energy_ele` 为其中的显式交互。`compare_environment` 计算同参数无ENV的 `energy_vac`，返回完整变化 `energy_response=energy−energy_vac`。无ENV普通能量自然对应真空。ENV无参考能或自能，QM力从完整能量求导。

细致方案见 [docs/README.md](docs/README.md)，另见 [架构图](docs/figures/architecture_v4.svg) 和 [验收报告](validation-v4.md)。

## 训练

从 `mace/modules/envhead` 执行：

```bash
conda activate deepfield
bash runs/config/run_suite.sh --dry-run v4
bash runs/config/run_suite.sh v4
```

只运行四套：Fukui/NQE E/F/q、Fukui/NQE E/F、projection E/F/q、projection E/F。默认 OFF23small、cuEq float32、λ5Å、C16、partial、多层读出；60epochs、batch900、OOM−30、Adam .01、Plateau .5/5；权重10/100/10或10/100/0。输出 `runs/amino_v4/`，保存 `best.pt`、`last.pt`。

服务器默认 `Foundation/MACE-OFF23_small.model` 和 `runs/config/amino_v2.extxyz`。可覆盖：

```bash
FOUNDATION=/path/to/MACE-OFF23_small.model TRAIN_FILE=/path/to/train.extxyz \
VALID_FILE=/path/to/valid.extxyz bash runs/config/run_suite.sh v4
RESUME=1 bash runs/config/run_suite.sh v4
```

新CLI训练默认v4，恢复按checkpoint版本执行。旧v3 suite的 `all` 保持18套，旧脚本明确选择v3。新格式 `envhead-v4`，无权重转换。历史设计见 [v3说明](docs/README-v3.md)，配置见 [运行表](runs/config/README.md)。

## Tensor / ASE

```python
from mace.modules.envhead.runtime import EnvRuntime
from mace.modules.envhead.train.checkpoint import load_checkpoint
from mace.modules.envhead.calculator import EnvHeadCalculator

model = load_checkpoint('/path/to/best.pt', device='cuda')['model']
runtime = EnvRuntime(model)
out = runtime(Z_Q, R_Q, R_ENV, b_ENV, Z_ENV, total_charge=Q)
# energy, energy_ele, charges, forces
comparison = runtime.compare_environment(Z_Q, R_Q, R_ENV, b_ENV, Z_ENV,
                                         total_charge=Q)
# 四个energy字段及 charges / charges_vacuum / delta_charges
atoms.calc = EnvHeadCalculator(model, device='cuda')
comparison = atoms.calc.compare_environment(atoms)  # 保留普通缓存
```

数据沿用extxyz/HDF5：`REF_energy`、`REF_forces`、`total_charge`、可选 `REF_charges`、`mm_positions`、`mm_charges`、`mm_atomic_numbers`。单位eV、Å、e；来源group_id划分，无gas/PC配对要求。当前非周期、单能量头、单重态。

## 文件与 Git

模型源码纳入版本管理。`runs/`（包含 `runs/config/`）、`docs/`、EnvHead 测试目录、
`Foundation/`、历史设计说明和验证产物均作为本地文件忽略；取消追踪不会删除本地文件。

2026-10-11 清理将旧 v1/v2 说明、配置和已失效的测试，以及退出调度的 ESP 配置，
移入本地 `archive/2026-10-11-cleanup/`。该目录被忽略，`manifest.json` 记录原路径与文件哈希。
当前设计入口为 `docs/README.md`；活动 v3/v4 运行配置和 checkpoint 加载逻辑保留。
