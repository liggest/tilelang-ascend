// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_BUFFER_SHAPE_CONFIG_H_
#define TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_BUFFER_SHAPE_CONFIG_H_

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl::a2a3 {

// These two entries are copied from TA 0deb8b93 operation_config.h. Keeping
// them independent avoids pulling unrelated, not-yet-migrated op registrations
// into the shape-only passes. Operation consumers will share these definitions.
inline const std::unordered_set<std::string> kScopesToFlatten = {
    "shared",        "shared.ub",     "shared.l1",
    "wmma.matrix_a", "wmma.matrix_b", "wmma.accumulator"};

inline const std::unordered_map<std::string, int> kScopeForAlignment = {
    {"shared.ub", 32 * 8}};

} // namespace tvm::tl::a2a3

#endif // TVM_TL_ASCEND_A2A3_TRANSFORM_COMMON_BUFFER_SHAPE_CONFIG_H_
