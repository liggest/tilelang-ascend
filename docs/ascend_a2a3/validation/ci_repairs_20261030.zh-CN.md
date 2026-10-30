# 2026-10-30 CI 修复的针对性验证

本次针对性验收的代码基线为 `8e421db75a41a8a74a94134adf079e3d9c420624`。
**修后完整 daily/manual CI 按用户明确要求暂不重跑，阶段 3 仍待验收。**
[699 完整 CI 基线](full_ci_20261030.zh-CN.md)原始失败结果保持，不把聚焦验证换算为修后全量通过。
[JSON 摘要](ci_repairs_20261030.json)记录来源、各窗口实际 native 和本地证据 hash；原始大日志不纳入提交。

## 当前修复与构建

固定 TA 来源为 `463c234ce3087205c092e931207ca87b4cd5494c`。

| 主开发线提交 | 改动 |
|---|---|
| `b8924c1b` | gather 从现代 JIT 包装读取原 factory 名；kernel、20 case、golden 不变 |
| `e4106a75` | 更新 23 项 source contract 的实际 offset 和源码 hash；完整旧记录保存在 history，其他字段和严格断言保持 |
| `4cfa41c0` | printf 测试期望 StringImm 的真实字符串，仅去除误加的 raw 前缀 |
| `8e421db7` | 整数 FloorDiv/Mod 统一调用 A2/A3 共享模板 helper，AscendC/PTO 共用 `integer_math.h` |

当前 helper 为 `inline constexpr`，保实际 C++ 类型提升、MIN `% -1` 的安全返回、单次实参求值和 Select 惰性。
无 codegen 证明器、动态 helper 发射或每函数状态；非整数保原发射。
vector/packed 保现有 whole-value 接口，不承诺新增逐 lane 语义。
原 Dilation 算法、case、seed、golden、同步和分配未改；shared/A5 路径不改。
B0E 中间实现及其设备通过记录只保留为历史，不作为本简化版本的验收。

| 已完成范围 | 结果 |
|---|---|
| gather / source contract / printf 原 host/source 家族 | 分别 221 / 152 / 199 PASS，原 native `376afd…` |
| 简化版实际模板 host 数学、ABI/UAC、单次/惰性及 packed 接口 | 26 PASS，不混入旧版 40 项矩阵 |
| 原 codegen source 家族 | 42 PASS |
| 最新 constexpr header 的关键 host 检查 | 双端 2 PASS，并编译负 floor 和模板 extent 的 static_assert |
| 双端真实 native CPP 与各自真实 common 的 CANN SDK 编译 | 2/2 退出 0、生成 ELF；未加载 ELF、未执行 NPU |
| 主仓正常 `USE_ASCEND=ON` 构建 | 两 codegen 与两 rt_mod 共 4 TU，退出 0，未复制隔离库 |
| 主仓新 native 的双端 Select / 负除法 host smoke | 2 PASS，实际根库加载身份已核，无 NPU 初始化 |

主仓新 native SHA256 为 `def65d72f5f8cf4099287195ce310c3e91fbd5d520c32623cedfe1220aff83bb`；
Cython 为 `7dc8a8055433ff91dfaed26024d25e0ebe5fe6e19f8503e0942862408cd31c01`，CANN 为原 9.1.0-beta.1。
共享 header SHA256 为 `817434ea47ee87e237b2b9b4e8117fee59672aaf14791a04513614182f0a92b4`，与双 SDK 实际消费版本一致。
各计数保各自范围，不相加充当独立 NPU case 数或完整 CI 覆盖数。

## 原 CLI 的针对性设备验证

**gather**：`4cfa41c0882933ca16da4e8099e1b7e04df45e86` / 原 native
`376afdbe0739cad1bf13d00862a70b94c96a2332ae4e347ea289b3b650b3fdbc`，
物理 3 / logical 0 完整原 20 case 全 PASS，原退出 0 且原成功谓词通过。
这是 floor 简化之前的独立窗口，不称为新 native 上重跑了 gather 矩阵。

**Dilation 简化版**：上述 `8e421db7` / 新 native `def65d72…`，物理 3 / logical 0，
执行原 `python -u test_dilation_2d.py`，无 level/case/seed/golden 覆盖，保持默认 `all`。
L0 8、L1 30、repeat 3、L2 11、Boundary 6 全 PASS，原退出 0 且打印 `Test Passed!`。
L2 是原拒绝输入检查、repeat 是原逐位一致性检查，不把各层次混称新增数值 case。
原失败 L1 case 28 保持 NHWC `(1, 2, 5, 7)` / float32 / 3×3 / SAME / seed 127，
原 `1.2e-4` gate 下 matched ratio 1.0000、max abs 0。

测试及 kernel 与 699 和本次代码基线逐字节相同；共享 header、双 common、native、Cython、SDK 及控制器 helper
冻结 hash、真实进程映射和 inode 均已核对。自有进程组自然回收、没有清理信号，物理 3 后健康、空闲、无进程。
本结果属于简化版的新窗口，未复用 B0E 的设备通过记录。

原始大产物保留在本地 `refactor_main/reports/` 的 `p3_1_ci_host_repair_candidates/`、
`p3_1_gather_device_recheck_4cfa41c0/`、`p3_1_floor_template_simplification/` 及对应新 Dilation 窗口。
四项问题的聚焦验证与原 699 全量失败现场分别保存；修后完整 daily/manual 仍未运行。
