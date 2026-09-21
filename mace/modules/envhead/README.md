# EnvHead：直接 delta 与多极反馈

新版预测 `E = E_base + delta_energy`。真空和 QM/PC 数据训练同一个 head，零场
修正可以非零。基础参数冻结；head 使用固定次数的多极反馈，无电子态最小化。

[架构与物理约定](ARCHITECTURE.md) · [验收记录](docs/validation.md)

## 模块职责

| 位置 | 职责与 MACE 接点 |
|---|---|
| `adapters/` | 原生 MACE、ScaleShiftMACE、MACELES 的完整能量与 pre-product 特征接点；其他类型使用模板显式注册 |
| `modules/model.py` | 冻结基础势、批量 head 装配、总能量和 QM/MM 力；基础坐标图保持可微 |
| `modules/head.py` | Q 条件化、可选 MMC、固定多极更新、可选电荷 route（fukui/eem/sqe）的守恒更新和 delta 读出 |
| `modules/fusion.py` | additive/product/message；复用 MACE interaction/product blocks |
| `modules/electrostatics.py` | 解析 Gaussian 单极/偶极/四极核、外源分块、四极转换 |
| `modules/config.py` | 新版网络配置；不接收旧变分/求解器参数 |
| `data/` | 原生 extxyz/HDF5 读取与额外 MM/电子组字段；MM 不进入基础原子图 |
| `train/` | 原生损失/训练组件的扩展接线、分组、分项指标、新版保存恢复；不修改原生 train.py |
| `runtime.py` | 独立张量调用，返回总能量与双侧力，可选极化率 |
| `script/gaussian.py` | 独立数据提取脚本；不参与训练导入，新架构不改变转换逻辑 |

## 训练

在已安装当前 mace-ena 的 deepfield 环境中：

```bash
python -m mace.modules.envhead.train \
  --foundation Foundation/DeepField_LR.model \
  --train-file eegmfcc.extxyz \
  --output-dir runs/envhead-delta-product \
  --fusion product --density-max-l 2 \
  --channels 32 --num-interactions 2 --correlation 2 \
  --gaussian-width 1.0 --mm-chunk-size 1024 \
  --epochs 20 --batch-size 32 --learning-rate 1e-3 \
  --energy-weight 1 --forces-weight 1 \
  --valid-fraction 0.1 --seed 42 --device cuda --num-workers 8
```

`fusion` 为 `additive/product/message`，默认 product。`density-max-l=2` 表示电荷、
偶极和四极；设为 1 可做电荷＋偶极对照。MMC 默认关闭：接点缺少目标阶数时报错，
显式加 `--mmc` 才构造缺失初始几何通道。MMC 不限制正常 head 内的阶数耦合。

`--charge-route` 选择 q 通道的守恒更新方式，默认 `fukui`（softmax 残差投影，
与旧版逐位一致，不新增参数）。`eem` 用 bordered KKT 变分均衡解确定 q：电负性
χ 经 tanh 界读出、硬度 η = softplus + 0.5 eV 地板，每个电荷组一组约束行；
`sqe` 在 fukui 投影之上叠加分裂荷修正：键图按 r < 1.3×共价半径和从坐标派生
（H/C/N/O/S 表，跨组键和未知元素报错），键硬度 ζ = softplus + 0.5 eV 地板，
解正定的劈裂荷正规方程，反对称构造使组电荷和精确不变。两条求解路线的方程都
以**未中心化 MM 电势 φ 驱动**（EEM: χ+φ+Cq=−Sλ；SQE: 键驱动含 φ 的端点差，
即 D-SQE 的电压项）——解出的密度因此携带显式极化响应 ∂q/∂φ ≈ −C⁻¹，常数
势平移分别被乘子（1 属于组指示张成）和 Bᵀ1=0 吸收，是纯规范；极化率探针场
同样进入求解。两条路线都用零初始化标量门插值
`q = q_uniform + g·(q_solve − q_uniform)`：g=0 时与 fukui 逐位一致（零初始
化恒等式），能量仍为 learned + p·v，求解核不加自静电能项。
每层以当层特征重解（初始层 + 每层融合后共 num_interactions+1 次）；每个
forward 只装配一次 q-q 库仑块（含解析自项对角）并在各层间复用。求解矩阵
的 χ/η/ζ 常量见 `modules/head.py`；保存的 config 携带 route，旧版 head 文件
（config 无该键）按 fukui 严格加载。

路线检查：`pytest tests/extensions/envhead/test_charge_route.py` 覆盖三条路线
的零初始化恒等式、组守恒、EEM 均衡化与 SQE 驻点方程、float32、checkpoint
roundtrip（含旧版无 route 键的 head）及 cuEq CPU naive 对照。CUDA cuEq 沙箱
内不可用，需在装有匹配 cuEquivariance 的机器上执行：
`OMP_NUM_THREADS=1 python -m pytest tests/extensions/envhead/test_charge_route.py -k cuda -v`。

`--head-config` JSON 可覆盖网络 CLI 设置，最终配置会记录并打印。`--dtype` 默认跟随
foundation；支持 float64 降为 float32，拒绝 float32 foundation 升为 float64。
训练与验证共用 batch-size，worker 使用 spawn，不继承父进程 CUDA 状态。

继续同一新版训练使用原命令加 `--resume`，epochs 为目标总 epoch 数。旧配置、旧 head
和旧训练 checkpoint 不兼容，无迁移入口；新训练拒绝已有非空输出目录。旧运行产物保留。

## 数据与损失

现有转换文件可继续使用：标准 `REF_energy`、`REF_forces`、`total_charge`，附加
`mm_positions [M,3]`、`mm_charges [M,1]`、`mm_count [B,1]`。真空 MM 数组为空。
支持可选 MM 力标签、固定电子组约束；本轮不监督 dipole 或 polarizability。

数据链路为 `extxyz/HDF5 → 原生 Configuration/EnvAtomicData → DataLoader → EnvMACE → loss`。
总能量和 QM 力沿用 MACE weighted energy/forces loss，能量误差按原子数归一；支持
MM 力损失。不要求配对 batch，不预先把文件改写成 delta 标签。真空/PC 的 E/F、各损失
分项和多极最大幅度分别记录。配置没有 group_id 时随机切分样本，不保证同帧隔离；有
完整 group_id 时按组切分。需要严格独立评估可显式提供 train/valid 文件。

## 推理与电子输出

```python
from mace.modules.envhead.train.checkpoint import load_head
from mace.modules.envhead.runtime import EnvRuntime

# foundation 为原始可信基础模型对象；新版 head 文件使用严格格式检查。
model = load_head("runs/envhead-delta-product/head.pt", foundation)
runtime = EnvRuntime(model)
result = runtime(atomic_numbers, qm_positions, mm_positions, mm_charges,
                 total_charge=Q, compute_polarizability=False)
```

主输出：`energy/forces/mm_forces`、`base_energy/delta_energy/learned_energy/external_energy`。
电子输出：`density_coefficients`、`charges`、`atomic_dipoles`、`atomic_quadrupoles`、
`density_dipole`。多极系数为 `[N,9]`（l=2）或 `[N,4]`（l=1），偶极使用 Cartesian xyz。
四极为 STF 二阶矩，单位 e Å²，不采用 `3rr-r²I` 定义。`charge_error` 是逐图最大组误差。

`compute_polarizability=True` 额外输出 `energy_dipole [B,3]`、`polarizability [B,3,3]`，
对当前 PC 环境附近的均匀探测场求导；默认原点是固定零向量，可通过 `reference_origin`
指定。alpha 单位 e² Å²/eV；这是能量导数，和密度多极分别验证，不保证强制相等或正定。
`potential_shift` 仅用于检查电势零点约定。额外导数默认关闭。

反馈网络使用跨构型批处理；Gaussian 径向核使用支持混合导数的解析 backward。
这些工程优化保持配置与权重格式不变，已有新版 direct-delta head 可以继续使用。
真实 DeepField 小批次的优化前后对照见 [验收记录](docs/validation.md#等价计算加速)。

模型支持 eager 坐标力、训练所需混合导数及按需自动微分 QM Hessian（依赖完整 foundation 的二阶可微性）。支持有限 ML 分子、固定 MM 电荷、单能量头、
单进程；暂不支持自旋响应（total_spin 必须为 1）、PME、MM 互极化、共价边界、stress、
TorchScript、实际 MD 插件或下述 cuEq 路径之外的加速后端。ASE calculator 也提供力差分的数值 Hessian。引擎保留 MM 内部项与交叉 LJ，
不能再叠加已包含的 ML–MM 静电。模型不输出未经定义的逐原子总能量。
### ASE calculator with background point charges

```python
import numpy as np
from ase import Atoms
from mace.modules.envhead.calculator import EnvHeadCalculator

atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.25, 0.92, 0]])
atoms.info["total_charge"] = 0
atoms.info["multiplicity"] = 1
atoms.info["mm_positions"] = np.array([[3.0, 0.0, 0.0]])
atoms.info["mm_charges"] = np.array([0.4])
atoms.calc = EnvHeadCalculator(
    foundation="Foundation/MACE-OFF23_medium.model", head="runs/example/head.pt",
    device="cuda",
)
energy = atoms.get_potential_energy()  # eV: complete base + delta
qm_forces = atoms.get_forces()       # eV/Angstrom, QM atom order
mm_forces = atoms.calc.get_mm_forces(atoms)  # MM source order
charges = atoms.get_charges()       # head density monopoles, e

# Standard inherited ASE calculator methods also accept Atoms directly:
energy = atoms.calc.get_potential_energy(atoms)
qm_forces = atoms.calc.get_forces(atoms)
hessian = atoms.calc.get_hessian(atoms)  # analytic, eV/Angstrom**2
# Explicit finite-difference reference:
hessian_fd = atoms.calc.get_hessian(atoms, method="numerical", step=1e-3)
# [3*N, N, 3], eV/Angstrom**2; use .reshape(3*N, 3*N) for a matrix.
```

Alternatively pass `EnvHeadCalculator(model=assembled_envmace, device="cpu")`.
The calculator owns a copy and preserves its precision. File loading uses the
existing strict head/foundation compatibility checks. Updating either MM array
in `atoms.info`, even in place, invalidates the ASE cache. Both arrays may be
omitted for vacuum. `charge_group` and `charge_group_totals` in `atoms.info`
are forwarded to the runtime. Only nonperiodic, fixed-source embedding is
supported; stress and per-atom total energies are not exposed.

MM forces are the embedding contribution only. Downstream engines retain MM
internal interactions and cross LJ; do not add a second QM-MM electrostatic
energy already covered by this calculator. The density multipoles are model
outputs, not validated population-analysis charges.

`get_hessian` defaults to `method="analytic"`, using MACE's autograd Hessian
helper on the complete SR + LR + EnvHead energy. Both methods hold MM positions
and charges fixed and preserve the caller's coordinates and calculator cache.
The output shape is `[3N, N, 3]`, in eV/Å², without mass weighting.
`EnvRuntime(..., compute_hessian=True)` and `model(..., compute_hessian=True)`
return the tensor under `hessian`. This is an inference property, detached from
the graph; Hessian-loss training is not supported. Ordinary E/F calls do not
compute Hessians. Analytic mode rejects `step`; numerical mode requires 6N force
evaluations with default displacement 1e-4 Å for float64 and 1e-3 Å for float32.
Unsupported second derivatives or nonfinite results raise an error, without
switching backend or silently falling back to numerical differences.
The installed LES real-space kernel currently gives nonfinite native Hessians
at zero-distance self terms (`les/module/make_kernels.py`, `torch.norm`).
MACELES analytic Hessians remain blocked for that implementation; this extension
does not modify LES or replace its energy. The regression explicitly records
that native failure as an expected failure until LES supplies finite derivatives.
ASE `Vibrations(atoms, ...)` can also calculate the numerical Hessian through
the standard forces interface. This is the QM block with fixed sources, not a
full QM+MM vibrational Hessian. No mass weighting is applied by `get_hessian`.

Focused checks: `pytest tests/extensions/envhead/test_hessian.py
tests/extensions/envhead/test_calculator.py`. The nonzero-head float64 water
example has a maximum force-difference error of 1.3e-6 eV/Å² at step 1e-4 Å;
float32 gives about 1.2e-3 eV/Å² at step 1e-3 Å. These are small-system numerical
checks, not physical-accuracy evidence. CUDA tests cover all three fusion modes
and both dtypes against e3nn, with fixed vacuum/PC inputs. LES cases explicitly
check the native foundation before reporting the known derivative limitation.

### ASE 驱动的 ML/MM

`mlmm.EnvQMMMCalculator` 沿用 `ref/frag-method-on-ASE-main` 的减法式组合，
由 ASE 积分、OpenMM Context 计算经典能量/力、现有 EnvHeadCalculator 计算嵌入。
不需要 OpenMM-Torch 插件。固定的是 MM 电荷，MM 坐标正常参与运动。

```python
from openmm import XmlSerializer
from mace.modules.envhead.mlmm import EnvQMMMCalculator
from mace.modules.envhead.calculator import EnvHeadCalculator

# atoms 是全体系；与 system.xml 必须具有相同的粒子顺序。
with open("system.xml") as handle:
    system = XmlSerializer.deserialize(handle.read())
env = EnvHeadCalculator(foundation="Foundation/model.model",
                        head="runs/example/head.pt", device="cuda")
atoms.calc = EnvQMMMCalculator(
    atoms, system, qm_indices=[0, 1, 2], env_calculator=env,
    total_charge=0, platform="CPU",  # 经典侧平台独立于模型 device
)
energy = atoms.get_potential_energy()  # eV，全体系
forces = atoms.get_forces()           # eV/Å，全体系，包括 MM 反作用力
# 可直接交给 ASE VelocityVerlet / Langevin；结束后 atoms.calc.close()。
```

计算 `MM_full(q_QM=0) - MM_QM(q_QM=0) + EnvMACE`。两个经典 Context
都使用原始全体系索引；减项将所有涉及 MM 的能量项置零，因此无需重新参数化 QM 区。
保留 MM 内部作用和交叉 LJ，移除 QM 内部经典项与交叉经典静电。
QM/MM 模型力按原始索引回填，不平移 QM 坐标、不扣除 QM 合力。
原始 System 不被修改；背景电荷从原始 NonbondedForce 提取，无需写进 atoms.info。

默认使用非周期 `NoCutoff`；周期 RF 用法见下节。要求非共价边界、无约束和虚拟位点。
支持标准 NonbondedForce、HarmonicBondForce、HarmonicAngleForce、
PeriodicTorsionForce、RBTorsionForce 和 CMMotionRemover；其余力类型、参数 offsets、
跨区成键项/非键例外明确拒绝。输入身份/顺序、分区和电荷固定，改变它们应重建 calculator。
这里没有实现 OpenMM 内部积分、PME 嵌入或 link atoms，也不提供全体系 Hessian。
针对实际体系需检查力幅度和步长；小体系力差分/短 NVE 只证明接线和数值一致性。

### 周期 ML/MM：正交盒 reaction field

保留 ASE 积分方式，给组合 calculator 显式指定 `rf_cutoff`（Å）：

```python
atoms.set_cell([24., 24., 24.])
atoms.pbc = True
atoms.calc = EnvQMMMCalculator(
    atoms, system, qm_indices=[0, 1, 2], env_calculator=env,
    total_charge=0, platform="CPU", rf_cutoff=8., rf_dielectric=78.4,
)
```

范围：固定的轴对齐正交盒、全三维 PBC、单个完整成键 ML 分子。
cutoff 必须严格小于最短盒长的一半，展开后的 ML 分子各方向跨度小于半盒。
根据 HarmonicBondForce 的连接关系展开分子，检查环路一致性；分子自身保持有限体系，
foundation（包括 LES）仍接收非周期几何，不引入额外的 ML 周期自相互作用。
改变盒子需要重建 calculator，本轮不支持 NPT、三斜盒、跨共价边界。

实现依据为 `ref/amp_bms/source/modules/Modules.py::CoulombQMMM` 的 m=4,n=6
reaction-field 多项式及 `source/md/TorchForce.py::min_image_block` 的最小映像。
MM 普通静电由相同 RF 多项式计算，经典 NonbondedForce 仅保留 LJ 和原有
MM 非键例外（其缩放库仑形式仍按原力场）。LJ 在 0.8rc–rc 使用 OpenMM switching，
两个经典减法系统均关闭 dispersion tail correction；传入的 PME System 在副本中
明确改为上述 RF/LJ 方案，原 System 不修改。它不再代表原先的 PME Hamiltonian。

EnvHead 外源核为 Gaussian 核加 RF 多项式；额外乘以外层 0.8rc–rc 的 C4
切换函数 `S=(1-x)^5*(1+5x+15x²+35x³+70x⁴)`，其中
`x=clip((r²-(0.8rc)²)/(rc²-(0.8rc)²),0,1)`。MM RF 也使用该切换。
这一步是为 Gaussian 四极能量/力及二阶坐标导数设计的平滑延伸，未直接复用 AMP
的偶极/四极 B 系数；所有 EnvHead 多极系数由同一标量核的导数得到。
周期映像按原原子索引回填力，固定 MM 电荷不随 cutoff 缩放，因此不漏掉电荷切换链式力。

低层接口通过 `EnvRuntime(..., mm_cell=cell, rf_cutoff=8., rf_dielectric=78.4)`
或 QM Atoms 的同名 info 字段传入环境盒子；QM Atoms 本身保持非周期且必须预先展开。
组合 calculator 自动完成展开和传递。未指定 RF 时旧 checkpoint 的非周期前向不变。
RF 会改变训练时的外源核：已有 head 可加载，但其周期物理精度仍需 cutoff/介电常数
敏感性和 QM 参考检验，不能把数值接线测试视为精度证明。周期测试见
`tests/extensions/envhead/test_periodic_mlmm.py`。

性能说明：开关核在 cutoff 外严格为零，外源电势按"先过滤邻居对、再一次性求值"
实现（AMP-BMS 邻居表思想的无状态版本），结果与全对计算在浮点重排误差内
一致（`test_source_features_filters_beyond_rf_cutoff` 以逐对参考公式验证到
1e-12）；非周期/训练路径不经过滤，逐字保持原实现。

### cuEquivariance（可选，CUDA）

训练命令增加 `--enable-cueq`；ASE 使用：

```python
calc = EnvHeadCalculator(
    foundation="Foundation/MACE-OFF23_small.model",
    head="runs/example/head.pt",
    device="cuda", enable_cueq=True,
)
```

独立 tensor runtime 可先调用 `backends.enable_cueq(model.to("cuda"))`。
目前接入 native `MACE` / `ScaleShiftMACE`（包括 MACE-OFF）及 `MACELES`；用户自定义
foundation adapter 仍明确报不支持。默认关闭；缺少 CUDA 或 cuEq 依赖时明确报错。

- Foundation 复用当前 MACE 的 `convert_e3nn_cueq` 权重转换。
- MACELES 深拷贝完整模型，仅替换 `node_embedding`、`interactions`、`products`。
  原 `les`、全部 LES readouts、普通能量 readouts、缩放系数和其余模块保持原样；
  不重新调用 LES 构造函数。临时 cuEq 主干按实际末层 irreps 构造，无法匹配的
  模块布局或未完整转移的参数明确报错。LES 长程模块继续在所选设备上执行自身代码。
- Head 的 Linear、TensorProduct、interaction 和 product basis 使用 MACE 的
  `wrapper_ops`，覆盖 `additive/product/message`。带 bias 的密度读出保留 e3nn；
  Gaussian 多极静电、普通 MLP 和消息聚合仍为 PyTorch。
- 接点统一保留 `mul_ir` 布局和 `O3_e3nn` 基底。head 的原始 CG 权重按原生转换器
  的拼接顺序映射，使用可微的 `torch.func.functional_call` 执行；优化器始终更新
  原始 head 参数，力损失的混合导数可穿过映射。
- `conv_fusion=False`：当前原生 fused-convolution descriptor 未按 `mul_ir` 转换，
  实际 CUDA 对照曾出现能量偏差。此处使用支持显式布局的 cuEq ChannelWiseTensorProduct
  CUDA kernel，gather/scatter 单独执行；多体收缩仍使用 cuEq CUDA kernel。
- 保存的 head 和训练 checkpoint 仍为原始 e3nn 格式。
  后端执行缓存不写入 checkpoint；恢复权重或移动设备后自动重建。恢复训练时也可
  显式选择是否增加 `--enable-cueq`。启动转换保留 RNG 状态，沿用模型 dtype。
- 执行缓存会额外保存转换后的模块；不保证每个体系都降低峰值显存。

代码与参考：[backends.py](backends.py)、
[MACE 原生转换器](../../cli/convert_e3nn_cueq.py)、
[NVIDIA MACE 算子说明](https://nvidia.github.io/cuEquivariance/tutorials/pytorch/MACE.html)。

数值验证区分两条路径：CPU 测试显式使用 cuEq 的官方 `naive` 实现，核验转换、能量、
QM/MM 力、head 梯度和保存恢复；GPU 测试使用实际 CUDA kernels。工具沙箱内无法访问
CUDA；沙箱外已确认 RTX 3060 6 GB 正常可用。不能据 CPU 测试宣称 GPU 加速倍数。
沙箱外实际 CUDA 验收：`-k cuda` 得到 **7 passed**，其中 6 项覆盖三种融合模式的
float32/float64 能量、双侧力和力损失梯度，另 1 项检查 CPU 模型启用 CUDA 后端时的错误提示。
在 mace-ena 根目录、安装了匹配 CUDA 的 cuEquivariance 环境中运行：

```bash
OMP_NUM_THREADS=1 python -m pytest tests/extensions/envhead/test_cueq.py -k cuda -v
OMP_NUM_THREADS=1 python -m pytest tests/extensions/envhead/test_les_cueq.py -v
```

该测试覆盖 float32/float64 和三种融合模式的能量、双侧力及力损失参数梯度。
极化率的额外导数不在此次 cuEq 验收范围内；解析 Hessian 的独立对照见 `test_hessian.py`。
LES 测试另覆盖原模块保留、pre-product 接点、`base_les_energy`、checkpoint 恢复、
两个本地 LES foundation 的真空/PC 对照及 ASE calculator 调用；本地文件缺失时
仅跳过相应真实模型用例。`--dtype` 不指定时跟随 foundation，不隐式转成 float32。
恢复之前使用 float32 训练的 run 时，如果 foundation 文件是 float64，继续显式指定
`--dtype float32`，以匹配原 checkpoint 的精度。

LES 实际 GPU 对照（float32，三原子水、非零 head 权重，分别测试真空及一个 PC）：

| Foundation | 最大总能量差 / eV | 最大 LES 能量差 / eV | 最大 QM 力差 / (eV/Å) | 最大 MM 力差 / (eV/Å) |
|---|---:|---:|---:|---:|
| DeepField_LR | 0 | 0 | 3.58e-7 | 1.63e-9 |
| MACELES-OFF_small | 0 | 0 | 3.28e-7 | 3.73e-9 |

两个文件沿用其自身精度和 LES 配置，与各自未转换的前向比较；这验证计算等价性，
不涉及训练标签精度或不同 LES 库版本之间的可迁移性。

另以本地 `ref/Foundation/MACE-OFF23_small.model`、三原子水、一个固定 PC 和非零
head 权重做过 CPU naive 对照：能量差为 0，QM 力最大差为 1.22e-15 eV/Å，
MM 力最大差为 3.47e-18 eV/Å。该单构型检查验证本地 OFF 权重转换，不代表 GPU
性能、数据集精度或 MD 稳定性。
