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

# Generate C source code from the IR

import std/[strutils, tables, sets]
import ir

proc make_indent(level: int, width: int = 2): string =
  for it in 0..<(level * width):
    result.add(' ')

type Context = ref object
  program: Program
  kernel: Kernel
  regs: seq[string]
  indent: int

const NAMES = [
  InstrAdd: "+",
  InstrSub: "-",
  InstrMul: "*",
  InstrDiv: "/",
  InstrNegate: "-",
  InstrSin: "sin",
  InstrCos: "cos",
  InstrExp: "exp",
  InstrPow: "pow",
  InstrSqrt: "sqrt",
  InstrLog: "",
  InstrLog10: "log10",
  InstrLog2: "log2",
  InstrLn: "log",
  InstrEq: "==",
  InstrLt: "<",
  InstrLe: "<="
]

proc to_c(scalar_type: ScalarType): string =
  const TYPES = [Scalar32: "float", Scalar64: "double"]
  result = TYPES[scalar_type]

proc nim_int_to_c(): string = "long"

proc to_c(typ: Type, ctx: Context): string =
  case typ.kind:
    of TypeBoolean: result = "char"
    of TypeIndex: result = nim_int_to_c()
    of TypeScalar: result = ctx.program.scalar_type.to_c()

proc to_c(instrs: seq[Instr], ctx: Context): string =
  for instr in instrs:
    var expr = ""
    case instr.kind:
      of InstrIndex: expr = $instr.index_lit
      of InstrScalar: expr = $instr.scalar_lit
      of InstrBoolean: expr = $ord(instr.boolean_lit)
      of InstrAdd, InstrSub, InstrMul, InstrDiv,
         InstrEq, InstrLt, InstrLe:
        let
          op = NAMES[instr.kind]
          a = ctx.regs[instr.args[0]]
          b = ctx.regs[instr.args[1]]
        expr = "(" & a & " " & op & " " & b & ")"
      of InstrNegate:
        expr = "(" & NAMES[instr.kind] & ctx.regs[instr.args[0]] & ")"
      of InstrSin, InstrCos, InstrExp, InstrPow, InstrSqrt,
         InstrLog10, InstrLog2, InstrLn:
        expr = NAMES[instr.kind] & "("
        for it, arg in instr.args:
          if it != 0:
            expr &= ", "
          expr &= ctx.regs[arg]
        expr &= ")"
      of InstrSelect:
        let
          cond = ctx.regs[instr.args[0]]
          a = ctx.regs[instr.args[1]]
          b = ctx.regs[instr.args[2]]
        expr = "(" & cond & " ? " & a & " : " & b & ")"
      of InstrToScalar, InstrToIndex:
        var target_type = ""
        case instr.kind:
          of InstrToScalar: target_type = ctx.program.scalar_type.to_c()
          of InstrToIndex: target_type = nim_int_to_c()
          else: assert false
        expr = "((" & target_type & ")" & ctx.regs[instr.args[0]] & ")"
      of InstrShape, InstrLen, InstrShapeLen, InstrEpoch:
        case instr.kind:
          of InstrLen:
            expr = "builtin_len(" & $instr.tensor & ")"
          of InstrShapeLen:
            expr = "builtin_shape_len(" & $instr.tensor & ")"
          of InstrShape:
            expr = "builtin_shape(" & $instr.tensor & ", " & $instr.dim & ")"
          of InstrEpoch:
            expr = "builtin_epoch()"
          else: discard
      of InstrRead:
        let index = ctx.regs[instr.args[0]]
        expr = $instr.tensor & "[" & index & "]"
      of InstrWrite:
        let
          index = ctx.regs[instr.args[0]]
          value = ctx.regs[instr.args[1]]
        expr = $instr.tensor & "[" & index & "] += " & value
      of InstrLog, InstrExtern, InstrLoop, InstrThreads:
        raise GeneratorError(msg: "Unable to generate c source for " & $instr.kind)
    
    var stmt = ""
    if instr.res == RegId(0):
      stmt = expr
    elif ctx.regs[instr.res].len > 0:
      let typ = ctx.kernel.regs[instr.res].typ.to_c(ctx)
      stmt = typ & " " & ctx.regs[instr.res] & " = " & expr
    else:
      ctx.regs[instr.res] = expr
      continue
    
    if result.len > 0:
      result &= "\n"
    result &= make_indent(ctx.indent) & stmt & ";"

proc to_c(loop: Loop, body: string, ctx: Context): string =
  let iter = ctx.regs[loop.iter]
  result = loop.start.setup.to_c(ctx)
  if result.len > 0:
    result.add('\n')
  result &= loop.stop.setup.to_c(ctx)
  if result.len > 0:
    result.add('\n')
  result &= make_indent(ctx.indent)
  result &= "for(" & nim_int_to_c() & " " & iter & " = "
  result &= ctx.regs[loop.start.setup[^1].res] & "; "
  result &= iter & " < " & ctx.regs[loop.stop.setup[^1].res] & "; "
  result &= "++" & iter
  result &= ") {\n" & body & "\n" & make_indent(ctx.indent) & "}"

proc inline_registers(instrs: seq[Instr], usages: var seq[int]) =
  for instr in instrs:
    for arg in instr.args:
      usages[arg] += 1

proc inline_registers(expr: Expr, usages: var seq[int]) =
  expr.instrs.inline_registers(usages)
  if expr.res != RegId(0):
    usages[expr.res] += 1

proc inline_registers(index: LinearIndex, usages: var seq[int]) =
  index.setup.inline_registers(usages)
  usages[index.setup[^1].res] += 1

proc inline_registers(loop: Loop, usages: var seq[int]) =
  loop.start.inline_registers(usages)
  loop.stop.inline_registers(usages)
  usages[loop.iter] += 2

template with_indent(ctx: Context, body: untyped) =
  block:
    ctx.indent += 1
    defer: ctx.indent -= 1
    body

proc to_c(kernel: Kernel, ctx: Context): string =
  ctx.with_indent:
    ctx.kernel = kernel
    ctx.regs = new_seq[string](kernel.regs.len)
    ctx.indent += kernel.loops.len
    
    var usages = new_seq[int](kernel.regs.len)
    for loop in kernel.loops:
      loop.inline_registers(usages)
    kernel.expr.inline_registers(usages)
    
    for it, usage_count in usages:
      if usage_count > 1:
        ctx.regs[it] = $RegId(it + 1)
        if kernel.regs[it].name.len > 0:
          ctx.regs[it] &= "_" & kernel.regs[it].name
    
    result = kernel.expr.instrs.to_c(ctx)
    for it in countdown(kernel.loops.len - 1, 0):
      ctx.indent -= 1
      result = kernel.loops[it].to_c(result, ctx)
  result = make_indent(ctx.indent) & "{\n" & result
  result &= "\n" & make_indent(ctx.indent) & "}"

proc to_c*(program: Program): string =
  let ctx = Context(program: program)
  
  block builtins:
    result = make_indent(ctx.indent) & program.scalar_type.to_c() & "* "
    result &= "builtin_tensor(void* model, " & nim_int_to_c() & " id);"
    let tensor_arg = program.scalar_type.to_c() & "* " & "tensor"
    result &= "\n" & make_indent(ctx.indent) & "long builtin_shape(" & tensor_arg & ", long dim);"
    result &= "\n" & make_indent(ctx.indent) & "long builtin_len(" & tensor_arg & ");"
    result &= "\n" & make_indent(ctx.indent) & "long builtin_shape_len(" & tensor_arg & ");"
    result &= "\n" & make_indent(ctx.indent) & "long builtin_epoch();"
  
  for name, target in program.targets:
    result &= "\n" & make_indent(ctx.indent) & "void target_" & name & "(void* model) {"
    ctx.with_indent:
      for tensor in target.tensors:
        result &= "\n" & make_indent(ctx.indent)
        result &= program.scalar_type.to_c() & "* " & $tensor & " = "
        result &= "builtin_tensor(model, " & $int(tensor) & ");"
        result &= " // " & $program.tensors[tensor].kind
      for it, kernel in target.kernels:
        result &= "\n" & make_indent(ctx.indent) & "// " & $KernelId(it + 1)
        result &= "\n" & kernel.to_c(ctx)
    result &= "\n" & make_indent(ctx.indent) & "}"

