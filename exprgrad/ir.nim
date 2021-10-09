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

# Intermediate representation for exprgrad

import std/[macros, tables, sets, hashes]

type
  CompilerError* = ref object of CatchableError
  ParserError* = ref object of CompilerError
  TypeError* = ref object of CompilerError
  GradientError* = ref object of CompilerError
  GeneratorError* = ref object of CompilerError
  JitError* = ref object of CompilerError
  StageError* = ref object of CompilerError
  ShapeError* = ref object of CompilerError

const TARGET_SUPPORTS_THREADS* = compile_option("threads")

type
  KernelId* = distinct int
  RegId* = distinct int
  TensorId* = distinct int
  LoopId* = distinct int
  
  TypeKind* = enum TypeScalar, TypeIndex, TypeBoolean
  Type* = object
    kind*: TypeKind
    count*: int
  
  InstrKind* = enum
    InstrIndex, InstrScalar, InstrBoolean,
    InstrAdd, InstrSub, InstrMul, InstrDiv,
    InstrNegate, InstrSin, InstrCos,
    InstrExp, InstrPow, InstrSqrt,
    InstrLog, InstrLog10, InstrLog2, InstrLn,
    InstrEq, InstrLt, InstrLe, InstrSelect,
    InstrToScalar, InstrToIndex,
    InstrShape, InstrLen, InstrShapeLen,
    InstrRead, InstrWrite,
    InstrExtern, InstrEpoch,
    InstrLoop, InstrThreads
  
  Instr* = object
    args*: seq[RegId]
    res*: RegId
    tensor*: TensorId
    body*: seq[Instr]
    case kind*: InstrKind:
      of InstrIndex: index_lit*: int
      of InstrScalar: scalar_lit*: float64
      of InstrBoolean: boolean_lit*: bool
      of InstrExtern: extern*: string
      of InstrShape: dim*: int
      of InstrLoop: loop_iter*: RegId
      of InstrThreads:
        threads_closure*: seq[RegId]
        threads_tensors*: seq[TensorId]
        threads_begin*: RegId
        threads_end*: RegId
      else: discard
  
  Register* = object
    name*: string
    typ*: Type
  
  Expr* = object
    instrs*: seq[Instr]
    res*: RegId
  
  LinearIndex* = object
    setup*: seq[Instr]
    factors*: Table[RegId, int]
    constant*: int
  
  LoopMode* = enum LoopNone, LoopIndependent, LoopParallel
  
  Loop* = object
    iter*: RegId
    mode*: LoopMode
    has_bounds*: bool
    start*: LinearIndex
    stop*: LinearIndex
  
  TensorOpKind* = enum OpRead, OpWrite
  TensorOp* = object
    tensor*: TensorId
    is_raw*: bool
    dims*: seq[LinearIndex]
    data*: RegId
  
  ShapeConstrKind* = enum
    ShapeNone, ShapeDims, ShapeLinear, ShapeCopy
  
  ShapeConstraint* = object
    dest*: TensorId
    case kind*: ShapeConstrKind:
      of ShapeNone: discard
      of ShapeDims: dims*: seq[LinearIndex]
      of ShapeLinear:
        reads*: Table[TensorId, seq[seq[LinearIndex]]]
        write*: seq[LinearIndex]
      of ShapeCopy:
        src*: TensorId
  
  GenKind* = enum
    GenNone, GenBackwards, GenGradient, GenReshape
  
  Generator* = object
    tensor*: TensorId
    case kind*: GenKind:
      of GenReshape: reshape*: seq[int]
      else: discard
  
  Kernel* = ref object
    generator*: Generator
    regs*: seq[Register]
    setup*: seq[Instr]
    loops*: seq[Loop]
    reads*: seq[TensorOp]
    expr*: Expr
    write*: TensorOp
  
  CompileTarget* = enum CompileCpu, CompileThreads
  
  Target* = object
    name*: string
    output*: TensorId
    tensors*: HashSet[TensorId]
    shapes*: seq[ShapeConstraint]
    kernels*: seq[Kernel]
    compile_target*: CompileTarget
  
  TensorKind* = enum
    TensorResult, TensorInput, TensorParam, TensorCache, TensorRandom
  
  TensorDef* = object
    shape*: seq[int]
    name*: string
    case kind*: TensorKind:
      of TensorParam: init_range*: HSlice[float64, float64]
      of TensorRandom: random_range*: HSlice[float64, float64]
      of TensorCache: cache*: TensorId
      else: discard
  
  ScalarType* = enum
    Scalar32, Scalar64
  
  Stage* = enum
    StageTyped, # All register types are inferred
    StageGenerated, # All generators are converted to kernels
    StageFolded, # Linear indices are folded
    StageTensors, # Tensor lookups are available (Program.inputs, Program.params, Program.caches)
    StageCollected, # Required tensors are collected into target.tensors
    StageShapes, # Shape constraints are available for all kernels
    StageBounds, # All loop bounds are inferred
    StageTensorInstrs, # Tensor access operators are converted to instructions
    StageSortedShapes, # Shape constraint order is known. This stage should only be used in addition to StageConstraints, not insted of it.
    StageStaticShapes, # All static shapes are inferred
    StageIndependent, # All independent loops are identified
    StageLoops # All loops are inlined
  
  Program* = ref object
    tensors*: seq[TensorDef]
    inputs*: Table[string, TensorId]
    params*: seq[TensorId]
    caches*: seq[TensorId]
    targets*: Table[string, Target]
    stages*: set[Stage]
    scalar_type*: ScalarType

const
  SIDE_EFFECT_INSTRS* = {InstrWrite, InstrLoop, InstrThreads}

proc `<`*(a, b: LoopMode): bool = ord(a) < ord(b)
proc `<=`*(a, b: LoopMode): bool = ord(a) <= ord(b)

template define_id(Id, Obj: untyped, name, type_name: static[string]) =
  proc `==`*(a, b: Id): bool {.borrow.}
  proc hash*(id: Id): Hash {.borrow.}
  
  proc `[]`*[T](objs: seq[T], id: Id): T = objs[int(id) - 1]
  proc `[]`*[T](objs: var seq[T], id: Id): var T = objs[int(id) - 1]
  proc `[]=`*[T](objs: var seq[T], id: Id, obj: T) = objs[int(id) - 1] = obj
  
  proc `$`*(id: Id): string =
    if int(id) == 0:
      result = "no_" & name
    else:
      result = name & $(int(id) - 1)
  
  proc alloc*(objs: var seq[Obj], obj: Obj): Id =
    result = Id(objs.len + 1)
    objs.add(obj)
  
  proc alloc*(objs: var seq[Obj]): Id =
    result = Id(objs.len + 1)
    objs.add(Obj())
  
  proc new_lit*(id: Id): NimNode =
    new_call(bind_sym(type_name), new_lit(int(id)))

define_id(KernelId, Kernel, "kernel", "KernelId")
define_id(RegId, Register, "reg", "RegId")
define_id(TensorId, TensorDef, "tensor", "TensorId")
define_id(LoopId, Loop, "loop", "LoopId")

proc `$`*(typ: Type): string =
  result = ($typ.kind)[len("Type")..^1] & ":" & $typ.count

proc `==`*(a, b: Instr): bool =
  if a.kind == b.kind:
    result = a.args == b.args and a.res == b.res and a.tensor == b.tensor
    if result:
      case a.kind:
        of InstrIndex: result = a.index_lit == b.index_lit
        of InstrScalar: result = a.scalar_lit == b.scalar_lit
        of InstrBoolean: result = a.boolean_lit == b.boolean_lit
        of InstrShape: result = a.dim == b.dim
        else: discard

proc hash*(instr: Instr): Hash =
  result = hash(instr.kind) !& hash(instr.args) !& hash(instr.tensor) !& hash(instr.res)
  case instr.kind:
    of InstrIndex: result = result !& hash(instr.index_lit)
    of InstrScalar: result = result !& hash(instr.scalar_lit)
    of InstrBoolean: result = result !& hash(instr.boolean_lit)
    of InstrShape: result = result !& hash(instr.dim)
    else: discard
  result = !$result

proc `==`*(a, b: ShapeConstraint): bool =
  if a.kind == b.kind and a.dest == b.dest:
    case a.kind:
      of ShapeNone: result = true
      of ShapeDims: result = a.dims == b.dims
      of ShapeLinear: result = a.reads == b.reads and a.write == b.write
      of ShapeCopy: result = a.src == b.src

proc substitute*(instrs: var seq[Instr], subs: Table[RegId, RegId]) =
  template sub(x: var RegId) =
    if x in subs:
      x = subs[x]
  
  for instr in instrs.mitems:
    for arg in instr.args.mitems:
      sub(arg)
    sub(instr.res)

proc substitute*(expr: var Expr, subs: Table[RegId, RegId]) =  
  expr.instrs.substitute(subs)
  if expr.res in subs:
    expr.res = subs[expr.res]

proc substitute*(index: var LinearIndex, subs: Table[RegId, RegId]) =  
  index.setup.substitute(subs)
  var new_factors = init_table[RegId, int]()
  for reg, factor in index.factors:
    if reg in subs:
      new_factors[subs[reg]] = factor
    else:
      new_factors[reg] = factor
  index.factors = new_factors

proc substitute*(op: var TensorOp, subs: Table[RegId, RegId]) =
  for dim in op.dims.mitems:
    dim.substitute(subs)
  if op.data in subs:
    op.data = subs[op.data]

proc clone*(kernel: Kernel): Kernel =
  result = Kernel(
    generator: kernel.generator,
    regs: kernel.regs,
    loops: kernel.loops,
    reads: kernel.reads,
    expr: kernel.expr,
    write: kernel.write
  )

proc substitute*(instrs: var seq[Instr], subs: Table[TensorId, TensorId]) =
  for instr in instrs.mitems:
    if instr.tensor != TensorId(0):
      instr.tensor = subs[instr.tensor]

proc substitute*(index: var LinearIndex, subs: Table[TensorId, TensorId]) =
  index.setup.substitute(subs)

proc substitute*(loop: var Loop, subs: Table[TensorId, TensorId]) =
  loop.start.substitute(subs)
  loop.stop.substitute(subs)

proc substitute*(op: var TensorOp, subs: Table[TensorId, TensorId]) =
  for dim in op.dims.mitems:
    dim.substitute(subs)
  op.tensor = subs[op.tensor]

proc substitute*(kernel: Kernel, subs: Table[TensorId, TensorId]) =
  kernel.setup.substitute(subs)
  for loop in kernel.loops.mitems:
    loop.substitute(subs)
  for read in kernel.reads.mitems:
    read.substitute(subs)
  kernel.expr.instrs.substitute(subs)
  kernel.write.substitute(subs)

proc substitute*(shape: var ShapeConstraint, subs: Table[TensorId, TensorId]) =
  shape.dest = subs[shape.dest]
  case shape.kind:
    of ShapeNone: discard
    of ShapeDims:
      for dim in shape.dims.mitems:
        dim.substitute(subs)
    of ShapeLinear:
      var reads = init_table[TensorId, seq[seq[LinearIndex]]]()
      for tensor, dims in shape.reads:
        reads[subs[tensor]] = dims
        for indices in reads[subs[tensor]].mitems:
          for index in indices.mitems:
            index.substitute(subs)
      shape.reads = reads
    of ShapeCopy:
      shape.src = subs[shape.src]

iterator tensor_ops*(kernel: Kernel): (TensorOpKind, TensorOp) =
  for read in kernel.reads:
    yield (OpRead, read)
  yield (OpWrite, kernel.write)

proc new_lit*[K, V](tab: Table[K, V]): NimNode =
  result = new_tree(nnkStmtListExpr)
  let tab_sym = gensym(nskVar, "tab")
  result.add(new_var_stmt(tab_sym, new_call(new_tree(nnkBracketExpr, [
    bind_sym("init_table"), get_type_inst(K), get_type_inst(V)
  ]))))
  for key, value in tab.pairs:
    result.add(new_call(bind_sym("[]="), [
      tab_sym, new_lit(key), new_lit(value)
    ]))
  result.add(tab_sym)

proc new_lit*[T](hash_set: HashSet[T]): NimNode =
  result = new_tree(nnkStmtListExpr)
  let hash_set_sym = gensym(nskVar, "hash_set")
  result.add(new_var_stmt(hash_set_sym, new_call(new_tree(nnkBracketExpr, [
    bind_sym("init_hash_set"), get_type_inst(T)
  ]))))
  for value in hash_set:
    result.add(new_call(bind_sym("incl"), [
      hash_set_sym, new_lit(value)
    ]))
  result.add(hash_set_sym)

# Stages

const ALL_STAGES*: set[Stage] = block:
  var stages: set[Stage]
  for stage in low(Stage)..high(Stage):
    stages.incl(stage)
  stages

proc assert_pass*(program: Program,
                  name: string,
                  requires: set[Stage] = {},
                  produces: set[Stage] = {},
                  preserves: set[Stage] = {}) =
  for stage in requires:
    if stage notin program.stages:
      raise StageError(msg: "Pass " & name & " requires " & $stage & ", but only stages " & $program.stages & " are available")
  program.stages = program.stages * preserves + produces

proc assert_gen*(program: Program,
                 name: string,
                 requires: set[Stage] = {}) =
  for stage in requires:
    if stage notin program.stages:
      raise StageError(msg: "Generator " & name & " requires stage " & $stage & ", but only stages " & $program.stages & " are available.")

# LinearIndex arithmetic

proc `*`*(a: LinearIndex, b: int): LinearIndex =
  if b != 0:
    result.constant = a.constant * b
    result.setup = a.setup
    for reg, factor in a.factors:
      result.factors[reg] = factor * b

proc `+`*(a, b: LinearIndex): LinearIndex =
  result = a
  result.constant += b.constant
  result.setup.add(b.setup)
  for reg, factor in b.factors:
    if reg in result.factors:
      result.factors[reg] += factor
      if result.factors[reg] == 0:
        result.factors.del(reg)
    else:
      result.factors[reg] = factor

proc `-`*(a: LinearIndex): LinearIndex = a * (-1)
proc `-`*(a, b: LinearIndex): LinearIndex = a + b * (-1)

proc `*`*(a, b: LinearIndex): LinearIndex =
  if a.factors.len == 0:
    result = b * a.constant
  elif b.factors.len == 0:
    result = a * b.constant
  else:
    raise new_exception(ValueError, "")

proc `-`*(a: LinearIndex, b: int): LinearIndex =
  result = a
  result.constant -= b

proc eval*(index: LinearIndex, values: Table[RegId, int]): int =
  result = index.constant
  for reg, factor in index.factors:
    result += factor * values[reg]
