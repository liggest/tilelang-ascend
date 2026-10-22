// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_host.cc
 * \brief Preserve the A2/A3 host tiling attributes during TIRX migration.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tirx/transform.h>

#include <utility>

#include "arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace tirx;

namespace {

PrimFunc ProcessHost(PrimFunc func) {
  arith::Analyzer analyzer;
  arith::IRMutatorWithAnalyzer mutator(&analyzer);
  Stmt body = func->body;
  func.CopyOnWrite()->body = mutator(std::move(body));

  // The TA pass always wrote these attributes, even though its current
  // tiling collector never fills either container. Its AscendC and PTO
  // codegens read them, so preserve the contract while porting the pipeline.
  func = WithAttr(std::move(func), "tiling_map",
                  ffi::Map<Var, PrimExpr>{});
  return WithAttr(std::move(func), "var_sequence", ffi::Array<Var>{});
}

}  // namespace

tvm::transform::Pass A2A3HostLegalize() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return ProcessHost(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0, "tl.A2A3HostLegalize", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.A2A3HostLegalize", A2A3HostLegalize);
}

}  // namespace a2a3
}  // namespace tl
}  // namespace tvm
