# A2/A3 host tool sources

`parse_exception_dump.py` is the complete, byte-identical relocation of
`tilelang/tools/ascend_exception_dump_bin.py` at TA commit
`0deb8b93da11b7f0becbcded732d9633b7ca4e34`. The source and destination hashes
are recorded in `maint/ascend_a2a3/testing_ports.json`.

Use the backend-local import:

```python
from tilelang.ascend.a2a3.tools.parse_exception_dump import parse_exception_dump
```

The parser retains the original CANN `ASCEND_HOME_PATH` tool lookup, lexically
latest dump selection, wait interval, stale-output cleanup, subprocess protocol,
dtype fallback and errors. It does not compile or launch a kernel.

The adopted exception tests retain their original `ci_skip` markers. Hardware
exception injection remains an explicit device test; host parser and source
registration checks do not inject an exception.
