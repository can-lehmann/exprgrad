# Copyright 2021 Can Joshua Lehmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:/www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pretty printing for the intermediate representation

import std/[tables, strutils]
import ir

proc `[]`(regs: seq[string], reg: RegId): string =
  let index = int(reg) - 1
  if index >= 0 and index < regs.len:
    result = regs[index]
  else:
    result = "<no_reg>"

proc format_expr(instr: Instr, regs: var seq[string]): string =
  case instr.kind:
    of InstrIndex: result = $instr.index_lit
    of InstrScalar: result = $instr.scalar_lit
    of InstrBoolean: result = $instr.boolean_lit
    else:
      for it, arg in instr.args:
        if it != 0:
          result &= ", "
        result &= regs[arg]
      result = $instr.kind & "(" & result & ")"
  
  if instr.res != RegId(0):
    regs[instr.res] = result

proc format_expr(index: LinearIndex, regs: var seq[string]): string =
  for instr in index.setup:
    discard instr.format_expr(regs)
  
  if index.constant != 0:
    result = $index.constant
  
  for reg, factor in index.factors:
    if result.len > 0:
      result &= " + "
    if factor != 1:
      result &= $factor & " * "
    result &= regs[reg]
  
  if result.len == 0:
    result = "0"

proc format_expr(op: TensorOp, regs: var seq[string]): string =
  for dim in op.dims:
    if result.len > 0:
      result &= ", "
    result &= dim.format_expr(regs)
  
  if op.is_raw:
    result = "{" & result & "}"
  else:
    result = "[" & result & "]"
  result = $op.tensor & result

proc format_expr(expr: Expr, regs: var seq[string]): string =
  for instr in expr.instrs:
    discard instr.format_expr(regs)
  result = regs[expr.res]

proc `$`*(kernel: Kernel): string =
  if kernel.generator.kind != GenNone:
    result = $kernel.generator.kind
  else:
    var regs = new_seq[string](kernel.regs.len)
    for loop in kernel.loops:
      regs[loop.iter] = $loop.iter
    for read in kernel.reads:
      regs[read.data] = read.format_expr(regs)
    result = kernel.write.format_expr(regs) & " ++= " & kernel.expr.format_expr(regs)

proc `$`*(target: Target): string =
  result = target.name & " " & $target.output & ":"
  for it, kernel in target.kernels:
    result &= "\n  " & $kernel

proc `$`*(program: Program): string =
  var it = 0
  for tensor_it, tensor in program.tensors:
    if tensor.kind != TensorResult:
      if it != 0:
        result.add('\n')
      result &= $TensorId(tensor_it + 1) & " = "
      result &= ($tensor.kind)[len("Tensor")..^1].to_lower_ascii()
      case tensor.kind:
        of TensorParam: result &= "(" & $tensor.shape & ")"
        of TensorInput: result &= "(" & tensor.name & ")"
        else: discard
      it += 1
  for name, target in program.targets:
    if it != 0:
      result.add('\n')
    result &= $target
    it += 1

