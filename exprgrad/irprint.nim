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

proc make_indent(level: int, width: int = 2): string =
  for it in 0..<(level * width):
    result.add(' ')

proc stringify(instrs: seq[Instr], regs: var seq[string], level: int): string

proc count_usages(instrs: seq[Instr], usages: var seq[int]) =
  for it in countdown(instrs.len - 1, 0):
    let instr = instrs[it]
    if instr.res == RegId(0) or usages[instr.res] > 0:
      for arg in instr.args:
        usages[arg] += 1
      instr.body.count_usages(usages)

proc count_usages(index: LinearIndex, usages: var seq[int]) =
  for reg, factor in index.factors:
    usages[reg] += 1
  index.setup.count_usages(usages)

proc count_usages(op: TensorOp, usages: var seq[int]) =
  for dim in op.dims:
    dim.count_usages(usages)

proc count_usages(kernel: Kernel): seq[int] =
  result = new_seq[int](kernel.regs.len)
  if kernel.write.data != RegId(0):
    kernel.write.count_usages(result)
    result[kernel.write.data] += 1
  kernel.expr.instrs.count_usages(result)
  for read in kernel.reads:
    if result[read.data] > 0:
      read.count_usages(result)
  kernel.setup.count_usages(result)

proc find_inline_regs(instrs: seq[Instr],
                      inline: var seq[bool],
                      decl_levels: var seq[int],
                      level: int) =
  for instr in instrs:
    case instr.kind:
      of InstrLoop: inline[instr.loop_iter] = false
      of InstrThreads:
        inline[instr.threads_begin] = false
        inline[instr.threads_end] = false
      of InstrGpu:
        for index in instr.gpu_indices:
          inline[index.local] = false
          inline[index.group] = false
      of InstrRead: inline[instr.res] = false
      of InstrIndex, InstrBoolean, InstrScalar:
        inline[instr.res] = true
        decl_levels[instr.res] = -1
      else:
        if instr.res != RegId(0):
          decl_levels[instr.res] = level
    for arg in instr.args:
      if inline[arg] and decl_levels[arg] != level and decl_levels[arg] != -1:
        inline[arg] = false
    instr.body.find_inline_regs(inline, decl_levels, level + 1)

proc find_inline_regs(kernel: Kernel): seq[bool] =
  let usages = kernel.count_usages()
  result = new_seq[bool](kernel.regs.len)
  for reg in 0..<kernel.regs.len:
    if usages[reg] == 1:
      result[reg] = true
  for loop in kernel.loops:
    result[loop.iter] = false
  var decl_levels = new_seq[int](kernel.regs.len)
  kernel.setup.find_inline_regs(result, decl_levels, 0)
  kernel.expr.instrs.find_inline_regs(result, decl_levels, 1)

proc stringify(instr: Instr, regs: var seq[string], level: int): string =
  case instr.kind:
    of InstrBoolean: result &= $instr.boolean_lit
    of InstrIndex: result &= $instr.index_lit
    of InstrScalar: result &= $instr.scalar_lit
    of InstrLoop:
      result &= "loop " & $instr.loop_iter & " in " & regs[instr.args[0]]
      result &= " to " & regs[instr.args[1]]
      if instr.loop_step != 1:
        result &= " step " & $instr.loop_step
      result &= ":\n" & instr.body.stringify(regs, level + 1)
    of InstrThreads:
      result &= "threads (" & $instr.threads_begin & ", " & $instr.threads_end & ") in "
      result &= regs[instr.args[0]] & " to " & regs[instr.args[1]] & ":\n"
      result &= instr.body.stringify(regs, level + 1)
    of InstrGpu:
      result &= "gpu"
      for it, index in instr.gpu_indices:
        if it == 0:
          result &= " "
        else:
          result &= "\n" & make_indent(level + 2)
        result &= "(local " & $index.local & ", group " & $index.group & ", size " & $index.size & ")"
        result &= " in " & regs[instr.args[it * 2]] & " to " & regs[instr.args[it * 2 + 1]]
      result &= ":\n"
      result &= instr.body.stringify(regs, level + 1)
    of InstrIf:
      result &= "if " & regs[instr.args[0]] & ":\n"
      result &= instr.body.stringify(regs, level + 1)
    of InstrShape:
      result &= $instr.tensor & ".shape[" & $instr.dim & "]"
    of InstrSharedCache:
      result &= "shared_cache[" & $instr.cache_size & "]"
    of InstrAdd, InstrSub, InstrMul, InstrDiv, InstrIndexDiv, InstrMod,
       InstrEq, InstrLt, InstrLe, InstrAnd, InstrOr:
      const OPERATORS = [
        InstrAdd: "+", InstrSub: "-", InstrMul: "*", InstrDiv: "/",
        InstrIndexDiv: "div", InstrMod: "mod", InstrWrap: "wrap",
        InstrNegate: "", InstrSin: "", InstrCos: "",
        InstrExp: "", InstrPow: "", InstrSqrt: "",
        InstrLog: "", InstrLog10: "", InstrLog2: "", InstrLn: "",
        InstrEq: "==", InstrLt: "<", InstrLe: "<=",
        InstrAnd: "and", InstrOr: "or"
      ]
      result &= "(" & regs[instr.args[0]] & " " & OPERATORS[instr.kind] & " " & regs[instr.args[1]] & ")"
    else:
      result &= ($instr.kind)[len("Instr")..^1].to_lower_ascii()
      if instr.tensor != TensorId(0):
        result &= "[" & $instr.tensor & "]"
      result &= "("
      for it, arg in instr.args:
        if it != 0:
          result &= ", "
        result &= regs[arg]
      result &= ")"
  
  if instr.res != RegId(0):
    if regs[instr.res].len == 0:
      regs[instr.res] = result
      result = ""
    else:
      result = make_indent(level) & regs[instr.res] & " = " & result
  else:
    result = make_indent(level) & result

proc stringify(instrs: seq[Instr], regs: var seq[string], level: int): string =
  for it, instr in instrs:
    let line = instr.stringify(regs, level)
    if line.len > 0:
      if result.len > 0:
        result &= "\n"
      result &= line

proc stringify(index: LinearIndex, regs: var seq[string]): string =
  if index.setup.len > 0:
    discard index.setup.stringify(regs, 0) # TODO
  for reg, factor in index.factors:
    if result.len != 0:
      result &= " + "
    if factor == 1:
      result &= regs[reg]
    else:
      result &= $factor & " * " & regs[reg]
  if result.len == 0:
    result = "0"

proc stringify(cache: LocalCache, regs: var seq[string]): string =
  result = "cache " & regs[cache.reg] & " region "
  for it, dim in cache.dims:
    if it != 0:
      result &= ", "
    result &= dim.offset.stringify(regs)
    result &= " + [" & $dim.interval.min & ", " & $dim.interval.max & "]"

proc stringify(op: TensorOp, kind: TensorOpKind, regs: var seq[string], level: int): string =
  result &= make_indent(level)
  if kind == OpRead:
    result &= regs[op.data] & " = "
  result &= $op.tensor
  var dims = ""
  for it, dim in op.dims:
    if it != 0:
      dims &= ", "
    dims &= dim.stringify(regs)
  if op.is_raw:
    result &= "{" & dims & "}"
  else:
    result &= "[" & dims & "]"
  if kind == OpWrite:
    result &= " += " & regs[op.data]
  if op.cache.exists:
    result &= " " & op.cache.stringify(regs)

proc stringify(kernel: Kernel, level: int): string =
  var regs = new_seq[string](kernel.regs.len)
  for it, can_inline in kernel.find_inline_regs():
    if not can_inline:
      regs[it] = $RegId(it + 1)
  result &= make_indent(level) & "kernel:\n"
  result &= kernel.setup.stringify(regs, level + 1)
  if kernel.reads.len > 0 or
     kernel.write.data != RegId(0) or
     kernel.expr.instrs.len > 0:
    result &= "\n" & make_indent(level + 1) & "loops"
    if kernel.loops.len > 0:
      result &= " "
      for it, loop in kernel.loops:
        if it != 0:
          result &= ",\n" & make_indent(level + 1) & make_indent(6, 1)
        result &= $loop.iter
        if loop.has_bounds:
          result &= " in " & loop.start.stringify(regs) & " to " & loop.stop.stringify(regs)
          if loop.step != 1:
            result &= " step " & $loop.step
        if loop.schedule.tile:
          result &= " tile " & $loop.schedule.tile_size
    result &= ":"
    for read in kernel.reads:
      result &= "\n" & read.stringify(OpRead, regs, level + 2)
    if kernel.expr.instrs.len > 0:
      result &= "\n" & kernel.expr.instrs.stringify(regs, level + 2)
    if kernel.write.data != RegId(0):
      result &= "\n" & kernel.write.stringify(OpWrite, regs, level + 2)

proc stringify(target: Target, level: int): string =
  result = make_indent(level) & target.name & " " & $target.output & ":"
  for it, kernel in target.kernels:
    result &= "\n" & kernel.stringify(level + 1)

proc `$`*(kernel: Kernel): string = result = kernel.stringify(0)
proc `$`*(target: Target): string = result = target.stringify(0)

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

