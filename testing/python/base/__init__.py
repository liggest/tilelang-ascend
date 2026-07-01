"""
Base test infrastructure for TileLang-Ascend API tests.

Usage:
    from base import BinaryOpSpec, register_binary_op_tests
    from base import UnaryOpSpec, register_unary_op_tests
    from base import TOLERANCE, DTYPE_MAP, assert_close_npu, make_test_data
"""

from base.common import (
    TOLERANCE,
    DTYPE_MAP,
    DEFAULT_PASS_CONFIGS,
    assert_close_npu,
    make_test_data,
    skip_if_missing,
)

from base.binary_op import (
    BinaryOpSpec,
    BinaryOpTestClasses,
    register_binary_op_tests,
    make_binary_kernel,
    make_scalar_kernel,
    make_inplace_kernel,
    run_binary_op,
)

from base.unary_op import (
    UnaryOpSpec,
    UnaryOpTestClasses,
    register_unary_op_tests,
    make_unary_kernel,
    run_unary_op,
)
