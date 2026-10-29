# TA 463c234c 增量来源资料

本目录保存固定 TA `463c234ce3087205c092e931207ca87b4cd5494c` 的原始 operator registry 和相对 `0deb8b93da11b7f0becbcded732d9633b7ca4e34` 的差异，仅作阶段3.1来源审查。145项原登记包含新增active项及原simplefusion stale项；原登记状态不代表当前移植或设备通过。

当前不激活TA workflows、不创建额外CI执行入口，也不将这份源资料当作现有143项CI逻辑的替换。完整23新增/6修改、目标路径及各owner状态见 `../upstream_463c234c_ports.json`；旧394项的0deb来源记录保持独立。后续统一接线应保持原命令/参数/low_priority与pytest、bench协议，仅适配迁移路径和现代安装。阶段3仍待验收。

原始 unified diff 保存在 `operator_test_manifest.diff.gz`，可用 `gzip -dc` 查看；压缩保留原上下文空格，来源和压缩字节SHA均登记在上级增量台账。
