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

# Generate OpenCL kernels from the IR

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
  InstrIndexDiv: "/",
  InstrMod: "%",
  InstrWrap: "",
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
  InstrLe: "<=",
  InstrAnd: "&&",
  InstrOr: "||"
]

proc to_cl(scalar_type: ScalarType): string =
  const TYPES = [Scalar32: "float", Scalar64: "double"]
  result = TYPES[scalar_type]

proc nim_int_to_cl(): string = "long"

proc to_cl(typ: Type, ctx: Context): string =
  case typ.kind:
    of TypeBoolean: result = "char"
    of TypeIndex: result = nim_int_to_cl()
    of TypeScalar: result = ctx.program.scalar_type.to_cl()
    of TypeArray: raise GeneratorError(msg: $typ.kind & " is not supported by the c generator.")

template with_indent(ctx: Context, body: untyped): untyped =
  block:
    ctx.indent += 1
    defer: ctx.indent -= 1
    body

proc to_cl(instrs: seq[Instr], ctx: Context): string =
  for instr in instrs:
    var expr = ""
    case instr.kind:
      of InstrIndex: expr = $instr.index_lit
      of InstrScalar: expr = $instr.scalar_lit
      of InstrBoolean: expr = $ord(instr.boolean_lit)
      of InstrAdd, InstrSub, InstrMul, InstrDiv,
         InstrIndexDiv, InstrMod,
         InstrEq, InstrLt, InstrLe,
         InstrAnd, InstrOr:
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
          of InstrToScalar: target_type = ctx.program.scalar_type.to_cl()
          of InstrToIndex: target_type = nim_int_to_cl()
          else: assert false
        expr = "((" & target_type & ")" & ctx.regs[instr.args[0]] & ")"
      of InstrRead:
        let index = ctx.regs[instr.args[0]]
        expr = $instr.tensor & "[" & index & "]"
      of InstrWrite, InstrOverwrite:
        let
          index = ctx.regs[instr.args[0]]
          value = ctx.regs[instr.args[1]]
          op = case instr.kind:
            of InstrWrite: "+="
            of InstrOverwrite: "="
            else: ""
        expr = $instr.tensor & "[" & index & "] " & op & " " & value
        expr &= "/* " & $instr.tensor & " " & $ctx.program.tensors[instr.tensor].shape & " */"
      of InstrLoop:
        let
          body = ctx.with_indent(instr.body.to_cl(ctx))
          iter = ctx.regs[instr.loop_iter]
        if result.len > 0:
          result &= "\n"
        result &= make_indent(ctx.indent)
        result &= "for(" & nim_int_to_cl() & " " & iter & " = "
        result &= ctx.regs[instr.args[0]] & "; "
        result &= iter & " < " & ctx.regs[instr.args[1]] & "; "
        case instr.loop_step:
          of 1: result &= "++" & iter
          of -1: result &= "--" & iter
          else: result &= iter & " += " & $instr.loop_step
        result &= ") {\n" & body & "\n" & make_indent(ctx.indent) & "}"
        result &= " // " & $instr.loop_fuse_next
        continue
      of InstrIf:
        let
          body = ctx.with_indent(instr.body.to_cl(ctx))
          cond = ctx.regs[instr.args[0]]
        if result.len > 0:
          result &= "\n"
        result &= make_indent(ctx.indent)
        result &= "if (" & cond & ") {\n"
        result &= body & "\n" & make_indent(ctx.indent) & "}"
        continue
      of InstrSharedCache:
        if result.len > 0:
          result &= "\n"
        result &= make_indent(ctx.indent)
        result &= "__local " & ctx.program.scalar_type.to_cl() & " "
        result &= ctx.regs[instr.res] & "[" & $instr.cache_size & "]" & ";"
        continue
      of InstrCacheWrite:
        let
          cache = ctx.regs[instr.args[0]]
          index = ctx.regs[instr.args[1]]
          value = ctx.regs[instr.args[2]]
        expr = cache & "[" & index & "] = " & value
      of InstrBarrier:
        expr = "barrier(CLK_LOCAL_MEM_FENCE)"
      of InstrArrayRead:
        let
          arr = ctx.regs[instr.args[0]]
          index = ctx.regs[instr.args[1]]
        expr = arr & "[" & index & "]"
      of InstrShape, InstrLen, InstrShapeLen, InstrEpoch:
        raise GeneratorError(msg: $instr.kind & " may not appear in OpenCL kernel. This is an internal compiler error, please report this issue.")
      of InstrLog, InstrThreads,  InstrArray, InstrArrayLen, InstrWrap, InstrGpu:
        raise GeneratorError(msg: "Unable to generate OpenCL source for " & $instr.kind)
    
    var stmt = ""
    if instr.res == RegId(0):
      stmt = expr
    elif ctx.regs[instr.res].len > 0:
      let typ = ctx.kernel.regs[instr.res].typ.to_cl(ctx)
      stmt = typ & " " & ctx.regs[instr.res] & " = " & expr
    else:
      ctx.regs[instr.res] = expr
      continue
    
    if result.len > 0:
      result &= "\n"
    result &= make_indent(ctx.indent) & stmt & ";"

proc to_cl(indices: seq[GpuIndex], ctx: Context): string =
  for it, index in indices:
    template get_index(reg: RegId, name: string) =
      if ctx.regs[reg].len > 0:
        if result.len > 0:
          result &= "\n"
        result &= make_indent(ctx.indent) & nim_int_to_cl() & " " & ctx.regs[reg] & " = " & name & "(" & $it & ");"
    
    get_index(index.local, "get_local_id")
    get_index(index.group, "get_group_id")

proc inline_registers(instrs: seq[Instr], usages: var seq[int]) =
  for instr in instrs:
    for arg in instr.args:
      if usages[arg] != -1:
        usages[arg] += 1
    if instr.body.len > 0:
      instr.body.inline_registers(usages)
    case instr.kind:
      of InstrLoop: usages[instr.loop_iter] += 2
      of InstrSharedCache: usages[instr.res] += 2
      of InstrIndex, InstrScalar, InstrBoolean:
        usages[instr.res] = -1
      else: discard

proc to_cl*(instrs: seq[Instr],
            closure: ParallelClosure,
            indices: seq[GpuIndex],
            kernel: Kernel,
            program: Program): string =
  let ctx = Context(kernel: kernel, program: program)
  ctx.with_indent:
    ctx.regs = new_seq[string](kernel.regs.len)
    
    var usages = new_seq[int](kernel.regs.len)
    instrs.inline_registers(usages)
    
    for reg in closure.regs:
      usages[reg] = 2
    
    for index in indices:
      if usages[index.local] > 0:
        usages[index.local] = 2
      if usages[index.group] > 0:
        usages[index.group] = 2
    
    for it, usage_count in usages:
      if usage_count > 1:
        ctx.regs[it] = $RegId(it + 1)
        if kernel.regs[it].name.len > 0:
          ctx.regs[it] &= "_" & kernel.regs[it].name
    
    result = indices.to_cl(ctx)
    if result.len > 0:
      result &= "\n"
    result &= instrs.to_cl(ctx)
  
  var args: seq[string] = @[]
  for tensor in closure.tensors:
    args.add("__global " & program.scalar_type.to_cl() & "* " & $tensor)
  
  for reg in closure.regs:
    args.add(kernel.regs[reg].typ.to_cl(ctx) & " " & $reg)
  
  result = make_indent(ctx.indent) & "__kernel void cl_kernel(" & args.join(", ") & ") {\n" & result
  result &= "\n" & make_indent(ctx.indent) & "}"

