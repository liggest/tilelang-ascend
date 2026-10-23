// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "builtin.h"
#include "op/builtin_registry.h"

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace tirx;

// ascend_pto
TIR_DEFINE_TL_BUILTIN(ascend_scalar_op)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_unary_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_binary_op)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ascend_binary_ops)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace a2a3
} // namespace tl
} // namespace tvm
