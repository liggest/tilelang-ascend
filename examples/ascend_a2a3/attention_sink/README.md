# A2/A3 attention-sink examples

The forward-varlen and backward-BHSD examples retain their original standalone
scripts and companion tests. Configure CANN and Torch-NPU for an A2/A3 device,
then run the migrated entry points:

```bash
python examples/ascend_a2a3/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py
python examples/ascend_a2a3/attention_sink/example_gqa_sink_fwd_varlen/test_gqa_sink_fwd_varlen.py --level all
python examples/ascend_a2a3/attention_sink/example_gqa_sink_bwd_bhsd/test_gqa_sink_bwd_bhsd.py --level all
```

The test CLIs retain their original level selection and pass/failure criteria.
