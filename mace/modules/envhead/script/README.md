# 训练数据提取脚本

`gaussian.py` 提取 Gaussian 单点力计算；`orca.py` 提取 ORCA 单点 EnGrad。
脚本只生成训练文件，不加载 foundation，也不参与训练循环。
推荐直接通过文件路径执行，extxyz 仅需 NumPy/ASE；HDF5 额外需要 h5py。

## ORCA / Amino

从 ModernMLIP 根目录执行：

```bash
python mace-ena/mace/modules/envhead/script/orca.py dataset/Amino \
  --pc-dir dataset/Unit2/Traj2 \
  --output dataset/amino.extxyz --workers 8 --skip-invalid
```

将输出后缀改为 `.h5` 可写出 MACE Configuration 格式 HDF5。
`--skip-invalid` 将无法解析的文件及原因写入 `.audit.jsonl`，其余继续；
不指定则遇到第一个错误停止。已有输出或审计文件不会覆盖。
解析并行、写出串行，待写队列有界。

Amino 文件 `Unit-L_frame_0343_pc.out` 对应：
`Unit2/Traj2/Unit-L/orca_qmmm/frame_0343.pc`。
同目录的 `frame_0343_qm.xyz` 存在时会核对元素顺序和坐标。
普通输出也支持日志旁的被引用 `.pc`，或带 Unit 前缀的 `.pc`。
不跨目录按帧号猜测文件。PC 缺失或数量不符会拒绝导出。

输出字段：坐标 Å，`REF_energy` eV、`REF_forces` eV/Å、
`total_charge`、`multiplicity`、`mm_positions` Å、`mm_charges` e。
真空携带空 MM 数组。训练继续使用现有 extxyz/HDF5 读取入口。
ORCA `CARTESIAN GRADIENT` 是梯度，力取负号并从 Hartree/Bohr 换算。
使用 `FINAL SINGLE POINT ENERGY`，包含该计算最终的色散修正。

[ORCA 6.1 手册](https://www.faccts.de/docs/orca/6.1/manual/_static/pdf/orca.pdf)
规定外部 `%pointcharges` 默认不包含 PC–PC 自能，因此本脚本不再扣除自能。
`DoEQ` 会改变此约定，本版明确拒绝；多步计算、优化及连续溶剂也不在本版支持范围。
不从 Mulliken 电荷猜测背景电荷，不虚构 MM 力标签。

当前已检查 Amino 的 26 份输出（13 类残基各 gas/pc），并完成小样本
extxyz 回读、HDF5 写出及力符号/PC 匹配回归。未运行全量转换。
