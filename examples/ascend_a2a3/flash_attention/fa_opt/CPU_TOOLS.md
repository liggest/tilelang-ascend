# CPU preprocessing and plotting tools

These two tools retain the complete original TA implementations from
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. They run on the CPU and do not
compile or launch a kernel. Source hashes, adaptations and validation are in
`maint/ascend_a2a3/ecosystem_cpu_tool_ports.json`.

## Expand lock syntax

From the repository root:

```bash
python examples/ascend_a2a3/flash_attention/fa_opt/expert_v3/tools/preprocess.py \
  /path/to/reviewed-kernel.tl -o /tmp/generated-kernel.py
```

Omit `-o` to write to stdout. The original `LOCK`, `LOCKED`, `ACQ`, `REL`,
`INIT_LOCKS` and `DESTROY_LOCKS` expansion, indexed events, comments and other
Python lines are retained. Generated `set_flag` and `wait_flag` calls use the
three-argument public compatibility form in `tilelang.ascend.a2a3.language`.
The frontend converts these calls to the shared event ABI.

The preprocessor passes macro declarations and imports through unchanged.
Use a reviewed A2/A3 input with the private language namespace and ordinary
`@T.macro`; the tool does not create an alternate macro implementation or
silently rewrite old `hygienic=False` declarations. The separate attention
kernel and `.tl` migration remain under their own review. A successful
preprocessing operation does not demonstrate that the resulting kernel
compiles or runs.

## Render a performance CSV

Use the CSV format emitted by the original attention benchmark's `run.py`:

```bash
MPLBACKEND=Agg python examples/ascend_a2a3/flash_attention/fa_opt/plot.py \
  /path/to/comparison_report.csv -o /tmp/comparison.png
```

The tool reads `S`, `TL_Time(us)` and `AC_Time(us)`, converts the times to
milliseconds, and retains the original curves, labels, grid and 200 dpi PNG.
It requires an existing `pandas` and `matplotlib` installation; no dependency
installation or SDK changes are performed by the tool.

Importing either tool does not parse CLI arguments, read a performance CSV or
write a plot. With no arguments, the plot preserves the historical literal
`.log/run_cmp_performance_${RUN_ID}/comparison_report_${RUN_ID}.csv` input and
`time_tl-ascend_vs_ascendc.png` output. `${RUN_ID}` in that default is literal;
pass an actual CSV path explicitly. Missing files or required columns still
fail rather than produce a success marker. The CLI is for postprocessing
existing measurements, not collecting new device performance samples.
