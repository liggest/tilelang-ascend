// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#ifndef TVM_TL_ASCEND_A2A3_CODEGEN_FLOOR_ARITHMETIC_H_
#define TVM_TL_ASCEND_A2A3_CODEGEN_FLOOR_ARITHMETIC_H_

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>

#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>

#include "arith/ir_visitor_with_analyzer.h"

namespace tvm::codegen::a2a3 {

// Only emission changes: collect facts on the final IR without rewriting it or
// changing A2/A3's pipeline. A shared node must be safe at every occurrence.
class FloorArithmetic {
public:
  void Analyze(const tirx::Stmt &body) {
    ProofCollector collector;
    collector(body);
    direct_ = std::move(collector.direct);
  }

  bool CanUseTrunc(const PrimExpr &expr) const {
    // Leave the inherited vector/packed/other-type emission outside this
    // signed scalar compatibility fix. It does not add lane semantics.
    if (!expr.dtype().is_scalar() || !expr.dtype().is_int() ||
        expr.dtype().bits() < 8) {
      return true;
    }
    auto it = direct_.find(expr);
    return it != direct_.end() && it->second;
  }

  std::string EmitHelper(
      bool is_mod, std::ostream &declarations, const char *qualifier,
      const std::function<std::string(const std::string &)> &reserve_name) {
    std::string &name = is_mod ? mod_name_ : div_name_;
    if (!name.empty()) {
      return name;
    }
    name = reserve_name(is_mod ? "a2a3_floor_mod" : "a2a3_floor_div");
    declarations << "template <typename A, typename B>\n"
                 << qualifier << " inline auto " << name
                 << "(A a, B b) -> decltype(a / b) {\n"
                 << "  using R = decltype(a / b);\n"
                 << "  R lhs = a;\n  R rhs = b;\n";
    // Deduction preserves the existing source's usual arithmetic conversions,
    // including int32 IR shape variables emitted in an int64 kernel ABI.
    if (is_mod) {
      declarations << "  if constexpr (R(-1) < R(0)) {\n"
                   << "    if (rhs == R(-1)) return R(0);\n  }\n";
    } else {
      declarations << "  R q = lhs / rhs;\n";
    }
    declarations << "  R r = lhs % rhs;\n"
                 << "  if constexpr (R(-1) < R(0)) {\n"
                 << "    if (r != 0 && ((r < 0) != (rhs < 0))) "
                 << (is_mod ? "return r + rhs;" : "return q - 1;")
                 << "\n  }\n  return " << (is_mod ? "r" : "q") << ";\n}\n\n";
    return name;
  }

private:
  using ProofMap = std::unordered_map<PrimExpr, bool, ffi::ObjectPtrHash,
                                      ffi::ObjectPtrEqual>;

  class ProofCollector : public arith::IRVisitorWithAnalyzer {
  public:
    ProofMap direct;

    void VisitStmt_(const tirx::BindNode *op) final {
      VisitExpr(op->value);
      // Mutable loads/calls are not permanent equations for later statements.
      if (tirx::SideEffect(op->value) <= tirx::CallEffectKind::kPure) {
        analyzer_.Bind(op->var, op->value);
      }
    }

    void VisitExpr_(const tirx::LetNode *op) final {
      VisitExpr(op->value);
      if (tirx::SideEffect(op->value) <= tirx::CallEffectKind::kPure) {
        analyzer_.Bind(op->var, op->value);
      }
      VisitExpr(op->body);
    }

    void VisitExpr_(const tirx::FloorDivNode *op) final {
      Record(ffi::GetRef<PrimExpr>(op), op->a, op->b, false);
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const tirx::FloorModNode *op) final {
      Record(ffi::GetRef<PrimExpr>(op), op->a, op->b, true);
      StmtExprVisitor::VisitExpr_(op);
    }

  private:
    void Record(const PrimExpr &expr, const PrimExpr &a, const PrimExpr &b,
                bool is_mod) {
      bool safe = false;
      if (expr.dtype().is_scalar() && expr.dtype().is_int() &&
          tirx::SideEffect(a) <= tirx::CallEffectKind::kPure &&
          tirx::SideEffect(b) <= tirx::CallEffectKind::kPure) {
        bool positive = analyzer_.CanProve(a >= 0) && analyzer_.CanProve(b > 0);
        bool negative = analyzer_.CanProve(a <= 0) && analyzer_.CanProve(b < 0);
        bool exact = analyzer_.CanProve(b != 0) &&
                     analyzer_.CanProve(tvm::truncmod(a, b) == 0);
        safe = positive || negative || exact;
        // Even exact/same-sign mod cannot evaluate MIN % -1 in C++. Guard the
        // overflow domain before considering the direct-emission fast path.
        if (is_mod && !analyzer_.CanProve(b > 0) &&
            !analyzer_.CanProve(b < -1) && !analyzer_.CanProve(a >= 0)) {
          safe = false;
        }
      }
      auto [it, inserted] = direct.emplace(expr, safe);
      if (!inserted) {
        it->second = it->second && safe;
      }
    }
  };

  ProofMap direct_;
  std::string div_name_;
  std::string mod_name_;
};

} // namespace tvm::codegen::a2a3
#endif // TVM_TL_ASCEND_A2A3_CODEGEN_FLOOR_ARITHMETIC_H_
