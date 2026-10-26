// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
/*!
 * \file ascend/a2a3/ir.cc
 * \brief TA's NPU launch and resource-scope frames on the neutral frontend.
 *
 * Ported from TileLang-Ascend src/ir.cc at 0deb8b93. The shared launch frame
 * owns the Python stack and binding API. Only these A2/A3 factories are new;
 * they do not replace the official A5 mixed launch or its execution regions.
 */

#include "../../ir.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/tir/ir.h>
#include <tvm/tirx/stmt.h>

#include "../../transform/common/attr.h"

namespace tvm {
namespace tl {
namespace a2a3 {

using namespace script::ir_builder::tirx;
using namespace ffi;

KernelLaunchFrame KernelLaunch(const Array<PrimExpr> &grid_size,
                               const Optional<Array<PrimExpr>> &block_size,
                               const Map<String, Any> &attrs) {
  ICHECK_EQ(grid_size.size(), 1)
      << "A2/A3 NPU kernel must have exactly one block dimension";
  bool developer_mode = block_size.defined();
  PrimExpr vector_extent = Integer(2);
  if (developer_mode) {
    ICHECK_EQ(block_size.value().size(), 1)
        << "A2/A3 NPU kernel must have exactly one thread dimension";
    vector_extent = block_size.value()[0];
  }
  const auto *vector_count = vector_extent.as<IntImmNode>();
  ICHECK(vector_count && (vector_count->value == 1 || vector_count->value == 2))
      << "A2/A3 NPU kernel threads must be the constant integer 1 or 2, got "
      << vector_extent;

  ObjectPtr<KernelLaunchFrameNode> n =
      tvm::ffi::make_object<KernelLaunchFrameNode>();
  ForFrame cid_frame =
      MakeThreadBindingFrame("cid", "blockIdx.x", grid_size[0]);
  n->frames.push_back(cid_frame);
  n->grid_vars.push_back(cid_frame->vars[0]);
  n->grid_extents.push_back(grid_size[0]);

  ForFrame vid_frame = MakeThreadBindingFrame("vid", "cthread", vector_extent);
  n->frames.push_back(vid_frame);
  if (!developer_mode) {
    n->grid_vars.push_back(vid_frame->vars[0]);
    n->grid_extents.push_back(vector_extent);
  }

  // Preserve the shared launch frame's accessor contract. vid is the NPU
  // thread accessor's x binding, while the actual launch remains cthread.
  // The other two axes are neutral placeholders: MaterializeKernelLaunch
  // drops them, and diagnoses accidental use as SIMT threads on this target.
  ForFrame thread_frame = MakeLaunchThreadFrame();
  n->frames.push_back(thread_frame);
  n->thread_vars = {vid_frame->vars[0], thread_frame->vars[1],
                    thread_frame->vars[2]};
  n->thread_extents = Array<PrimExpr>{vector_extent, Integer(1), Integer(1)};

  if (developer_mode) {
    script::ir_builder::IRBuilder builder =
        script::ir_builder::IRBuilder::Current();
    Optional<PrimFuncFrame> prim_func = builder->FindFrame<PrimFuncFrame>();
    ICHECK(prim_func.defined()) << "A2/A3 KernelLaunch must be inside PrimFunc";
    prim_func.value()->attrs.Set(
        "npu_cv_ratio",
        tirx::StringImm(vector_count->value == 1 ? "cv_1_1" : "cv_1_2"));
  }

  SBlockFrame main_block =
      script::ir_builder::tirx::Block(::tvm::tl::DeviceMainBlockName);
  main_block->reads = Array<tirx::BufferRegion>();
  main_block->writes = Array<tirx::BufferRegion>();
  main_block->annotations = attrs;
  n->frames.push_back(main_block);
  return KernelLaunchFrame(n);
}

class ScopeFrameNode : public TIRFrameNode {
public:
  Array<TIRFrame> frames;

  static void RegisterReflection() {
    reflection::ObjectDef<ScopeFrameNode>().def_ro("frames",
                                                   &ScopeFrameNode::frames);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.A2A3ScopeFrame", ScopeFrameNode,
                                    TIRFrameNode);

  TVM_DLL void EnterWithScope() final {
    for (const TIRFrame &frame : frames) {
      frame->EnterWithScope();
    }
  }

  TVM_DLL void ExitWithScope() final {
    for (auto frame = frames.rbegin(); frame != frames.rend(); ++frame) {
      (*frame)->ExitWithScope();
    }
  }
};

class ScopeFrame : public TIRFrame {
public:
  explicit ScopeFrame(ObjectPtr<ScopeFrameNode> data) : TIRFrame(UnsafeInit{}) {
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScopeFrame, TIRFrame,
                                                ScopeFrameNode);
};

ScopeFrame Scope(const String &scope_name) {
  ObjectPtr<ScopeFrameNode> n = tvm::ffi::make_object<ScopeFrameNode>();
  // TA treats V as the vector resource and all other names as Cube.
  n->frames.push_back(
      Attr(Integer(0), "resource_scope", Integer(scope_name == "V" ? 1 : 0)));
  return ScopeFrame(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ScopeFrameNode::RegisterReflection();
  reflection::GlobalDef()
      .def("tl.A2A3KernelLaunch", KernelLaunch)
      .def("tl.A2A3Scope", Scope);
}

} // namespace a2a3
} // namespace tl
} // namespace tvm
