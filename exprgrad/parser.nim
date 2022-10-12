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

# Parser for exprgrad's domain-specific language

import std/[tables, hashes, strutils, macros, sets]
import ir

type
  ExprKind* = enum
    ExprInstr, ExprIter, ExprRead
  
  ExprBuilder* = ref object
    children*: seq[ExprBuilder]
    tensor*: Fun
    res*: Table[int, RegId]
    case kind*: ExprKind:
      of ExprIter:
        iter*: string
      of ExprInstr:
        case instr*: InstrKind:
          of InstrIndex: indexLit*: int
          of InstrScalar: scalarLit*: float64
          of InstrBoolean: booleanLit*: bool
          of InstrShape: dim*: int
          else: discard
      of ExprRead:
        isRaw*: bool
      else: discard
  
  Scalar* = distinct ExprBuilder
  Index* = distinct ExprBuilder
  Boolean* = distinct ExprBuilder
  Array*[T] = distinct ExprBuilder
  
  Schedule = ref object
    tensors: Table[Fun, TensorSchedule]
    loops: Table[string, LoopSchedule]
  
  KernelBuilder = ref object
    target: Fun
    dims: seq[Index]
    isRaw: bool
    value: Scalar
    hasCustomGrad: bool
    grads: seq[KernelBuilder]
    blockCount: int
    schedules: array[CompileTarget, Schedule]
  
  ShapeConstraintBuilder* = object
    case kind: ShapeConstrKind:
      of ShapeNone, ShapeLinear, ShapeRank: discard
      of ShapeDims: dims: seq[Index]
      of ShapeCopy: copy: Fun
  
  FunKind* = enum
    FunInput, FunParam, FunResult, FunCache, FunRandom,
    FunBackwards, FunGradient, FunEffect, FunMultiple,
    FunReshape, FunTarget, FunCond, FunGradientArg
  
  Fun* = ref object
    targets: HashSet[string]
    tensor*: TensorId
    children: seq[Fun]
    name*: string
    locked*: bool
    case kind: FunKind:
      of FunInput:
        inputShape: seq[int]
      of FunParam:
        paramShape: seq[int]
        initRange: HSlice[float64, float64]
      of FunCache: cache: Fun
      of FunRandom: randomRange: HSlice[float64, float64]
      of FunResult, FunEffect:
        kernels: seq[KernelBuilder]
        shapeConstr: ShapeConstraintBuilder
        effect: Fun
      of FunReshape: reshape: seq[int]
      of FunCond:
        cond: Table[string, Fun]
        condElse: Fun
      of FunTarget:
        compileTarget: CompileTarget
      of FunBackwards, FunGradient, FunMultiple, FunGradientArg:
        discard

proc hash*(fun: Fun): Hash = hash(fun[].addr)

proc literal*(value: bool): Boolean =
  result = Boolean(ExprBuilder(kind: ExprInstr, instr: InstrBoolean, booleanLit: value))

proc literal*(value: int): Index =
  result = Index(ExprBuilder(kind: ExprInstr, instr: InstrIndex, indexLit: value))

proc literal*(value: float): Scalar =
  result = Scalar(ExprBuilder(kind: ExprInstr, instr: InstrScalar, scalarLit: value))

proc literal*(value: Index): Index = value
proc literal*(value: Scalar): Scalar = value
proc literal*(value: Boolean): Boolean = value
proc literal*[T](value: Array[T]): Array[T] = value

proc literal*[T](arr: openArray[T]): auto =
  let builder = ExprBuilder(kind: ExprInstr, instr: InstrArray)
  for value in arr.items:
    builder.children.add(ExprBuilder(literal(value)))
  result = Array[typeof(literal(arr[0]))](builder)

proc iteratorLiteral*(name: string, start: Index = nil, stop: Index = nil): Index =
  result = Index(ExprBuilder(kind: ExprIter,
    iter: name.nimIdentNormalize()
  ))
  if not ExprBuilder(start).isNil or not ExprBuilder(stop).isNil:
    ExprBuilder(result).children = @[ExprBuilder(start), ExprBuilder(stop)]

type BuildContext = object
  kernel: Kernel
  iters: Table[string, RegId]
  grads: Table[TensorId, TensorId]
  blockCount: int
  maxTensor: TensorId
  schedule: Schedule
  compileTarget: CompileTarget

proc allocBlock(ctx: var BuildContext): int =
  result = ctx.blockCount
  ctx.blockCount += 1

proc lookupTensor(ctx: var BuildContext, fun: Fun): TensorId =
  if fun.kind == FunGradientArg:
    let id = ctx.lookupTensor(fun.children[0])
    if id notin ctx.grads:
      ctx.grads[id] = TensorId(-ctx.grads.len - 1)
    result = ctx.grads[id]
  else:
    result = fun.tensor

proc build*(builder: ExprBuilder,
            instrs: var seq[Instr],
            blockId: int,
            ctx: var BuildContext): RegId

proc buildLinearIndex*(builder: ExprBuilder, ctx: var BuildContext): LinearIndex =
  let reg = builder.build(result.setup, ctx.allocBlock(), ctx)
  result.factors = toTable({reg: 1})

proc build*(builder: ExprBuilder,
            instrs: var seq[Instr],
            blockId: int,
            ctx: var BuildContext): RegId =
  if blockId notin builder.res:
    case builder.kind:
      of ExprRead:
        var dims: seq[LinearIndex] = @[]
        for dim in builder.children:
          dims.add(dim.buildLinearIndex(ctx))
        
        var schedule = DEFAULT_TENSOR_SCHEDULE
        if builder.tensor in ctx.schedule.tensors:
          schedule = ctx.schedule.tensors[builder.tensor]
        
        let res = ctx.kernel.regs.alloc()
        ctx.kernel.reads.add(TensorOp(
          tensor: ctx.lookupTensor(builder.tensor),
          isRaw: builder.isRaw,
          dims: dims,
          data: res,
          schedule: schedule
        ))
        builder.res[blockId] = res
      of ExprIter:
        if builder.iter notin ctx.iters:
          let reg = ctx.kernel.regs.alloc()
          ctx.iters[builder.iter] = reg
          var loop = Loop(iter: reg, schedule: DEFAULT_LOOP_SCHEDULE)
          if builder.iter in ctx.schedule.loops:
            loop.schedule = ctx.schedule.loops[builder.iter]
          if builder.children.len > 0:
            loop.hasBounds = true
            loop.start = builder.children[0].buildLinearIndex(ctx)
            loop.stop = builder.children[1].buildLinearIndex(ctx)
            loop.step = 1
          ctx.kernel.loops.add(loop)
        builder.res[blockId] = ctx.iters[builder.iter]
      of ExprInstr:
        var instr = Instr(kind: builder.instr)
        for child in builder.children:
          instr.args.add(child.build(instrs, blockId, ctx))
        
        if not builder.tensor.isNil:
          instr.tensor = ctx.lookupTensor(builder.tensor)
        
        case builder.instr:
          of InstrIndex: instr.indexLit = builder.indexLit
          of InstrScalar: instr.scalarLit = builder.scalarLit
          of InstrBoolean: instr.booleanLit = builder.booleanLit
          of InstrShape: instr.dim = builder.dim
          else: discard 
        
        instr.res = ctx.kernel.regs.alloc()
        builder.res[blockId] = instr.res
        instrs.add(instr)
  
  result = builder.res[blockId]

proc buildExpr(builder: ExprBuilder, ctx: var BuildContext): Expr =
  result.res = builder.build(result.instrs, ctx.allocBlock(), ctx)

proc clear(expr: ExprBuilder) =
  for child in expr.children:
    child.clear()
  expr.res = initTable[int, RegId]()

proc clear(builder: KernelBuilder) =
  ExprBuilder(builder.value).clear()
  for dim in builder.dims:
    ExprBuilder(dim).clear()

proc build(builder: KernelBuilder, ctx: var BuildContext): Kernel =
  result = Kernel()
  ctx.kernel = result
  ctx.schedule = builder.schedules[ctx.compileTarget]
  result.expr = ExprBuilder(builder.value).buildExpr(ctx)
  result.write = TensorOp(
    tensor: ctx.lookupTensor(builder.target),
    isRaw: builder.isRaw,
    data: result.expr.res
  )
  for dim in builder.dims:
    result.write.dims.add(ExprBuilder(dim).buildLinearIndex(ctx))
  if builder.hasCustomGrad:
    result.grad = KernelGradient(isCustom: true)
    var grads = initTable[TensorId, TensorId]()
    for grad in builder.grads:
      grad.clear()
      var gradCtx = BuildContext(
        grads: grads,
        compileTarget: ctx.compileTarget
      )
      result.grad.kernels.add(grad.build(gradCtx))
      grads = gradCtx.grads
    result.grad.tensors = grads

proc build(builder: KernelBuilder, compileTarget: CompileTarget): Kernel =
  builder.clear()
  var ctx = BuildContext(compileTarget: compileTarget)
  result = builder.build(ctx)

proc allocTensors(fun: Fun, program: Program) =
  if fun.tensor == TensorId(0):
    case fun.kind:
      of FunInput:
        if fun.name notin program.inputs:
          program.inputs[fun.name] = program.tensors.alloc(TensorDef(
            kind: TensorInput,
            shape: fun.inputShape,
            name: fun.name
          ))
        fun.tensor = program.inputs[fun.name]
        if program.tensors[fun.tensor].shape != fun.inputShape:
          raise ParserError(msg: "Expected shapes for input \"" & fun.name & "\" do not match.")
      of FunParam:
        fun.tensor = program.tensors.alloc(TensorDef(
          kind: TensorParam,
          shape: fun.paramShape,
          initRange: fun.initRange,
          name: fun.name
        ))
      of FunRandom:
        fun.tensor = program.tensors.alloc(TensorDef(
          kind: TensorRandom,
          randomRange: fun.randomRange,
          name: fun.name
        ))
      of FunResult, FunGradient, FunReshape:
        fun.tensor = program.tensors.alloc(TensorDef(
          kind: TensorResult,
          name: fun.name
        ))
      of FunEffect:
        fun.effect.allocTensors(program)
        fun.tensor = fun.effect.tensor
      of FunCache:
        fun.cache.allocTensors(program)
        fun.tensor = program.tensors.alloc(TensorDef(
          kind: TensorCache,
          cache: fun.cache.tensor,
          name: fun.name
        ))
      of FunCond:
        fun.tensor = TensorId(0)
        for target, child in fun.cond:
          child.allocTensors(program)
        if not fun.condElse.isNil:
          fun.condElse.allocTensors(program)
      else: discard
    
    for child in fun.children:
      child.allocTensors(program)
    
    case fun.kind:
      of FunTarget: fun.tensor = fun.children[0].tensor
      else: discard

proc flatten(fun: Fun, target: Target) =
  if target.name notin fun.targets:
    for child in fun.children:
      child.flatten(target)
    if fun.kind == FunEffect:
      fun.effect.flatten(target)
    
    fun.targets.incl(target.name)
    case fun.kind:
      of FunResult, FunEffect:
        for kernel in fun.kernels:
          target.kernels.add(kernel.build(target.compileTarget))
        case fun.shapeConstr.kind:
          of ShapeCopy:
            target.shapes.add(ShapeConstraint(kind: ShapeCopy,
              priority: PriorityUser,
              dest: fun.tensor,
              src: fun.shapeConstr.copy.tensor
            ))
          of ShapeDims:
            var constr = ShapeConstraint(kind: ShapeDims,
              priority: PriorityUser,
              dest: fun.tensor
            )
            for dim in fun.shapeConstr.dims:
              var ctx = BuildContext(kernel: Kernel())
              ExprBuilder(dim).clear()
              constr.dims.add(ExprBuilder(dim).buildLinearIndex(ctx))
            target.shapes.add(constr)
          else: discard
      of FunBackwards:
        target.kernels.add(Kernel(generator: Generator(
          kind: GenBackwards, tensor: fun.children[0].tensor
        )))
      of FunGradient:
        target.kernels.add(Kernel(
          generator: Generator(
            kind: GenGradient,
            tensor: fun.children[1].tensor
          ),
          write: TensorOp(tensor: fun.tensor)
        ))
      of FunReshape:
        target.kernels.add(Kernel(
          generator: Generator(
            kind: GenReshape,
            tensor: fun.children[0].tensor,
            reshape: fun.reshape
          ),
          write: TensorOp(tensor: fun.tensor)
        ))
      of FunCond:
        var child = Fun(nil)
        if target.name in fun.cond:
          child = fun.cond[target.name]
        elif not fun.condElse.isNil:
          child = fun.condElse
        else:
          raise ParserError(msg: "Conditional node does not have a branch for the target \"" & target.name & "\"")
        child.flatten(target)
        fun.tensor = child.tensor
      of FunRandom:
        target.shapes.add(ShapeConstraint(kind: ShapeCopy,
          priority: PriorityUser,
          dest: fun.tensor,
          src: fun.children[0].tensor
        ))
      else: discard

proc collectTargets(fun: Fun, targets: var Table[string, Fun]) =
  case fun.kind:
    of FunTarget:
      if fun.name in targets:
        if fun != targets[fun.name]:
          raise ParserError(msg: "There are multiple targets named \"" & fun.name & "\". Target names must be unique within a model. Choose a different name every target.")
        else:
          return
      targets[fun.name] = fun
    of FunCond:
      for name, child in fun.cond:
        child.collectTargets(targets)
      if not fun.condElse.isNil:
        fun.condElse.collectTargets(targets)
    else: discard
  for child in fun.children:
    child.collectTargets(targets)

proc toProgram*(graphs: openArray[Fun]): Program =
  result = Program()
  var targets = initTable[string, Fun]()
  for fun in graphs:
    fun.allocTensors(result)
    fun.collectTargets(targets)
  for name, fun in targets:
    let target = Target(
      name: name,
      output: fun.tensor,
      compileTarget: fun.compileTarget
    )
    fun.flatten(target)
    result.targets[name] = target

proc `$`(fun: Fun): string =
  if fun.isNil:
    result = "nil"
  else:
    result = "<fun>"

proc ensureInit(fun: var Fun) =
  if fun.isNil:
    fun = Fun(kind: FunResult)

proc collectChildren(expr: ExprBuilder, fun: Fun) =
  for child in expr.children:
    child.collectChildren(fun)
  if not expr.tensor.isNil:
    if expr.tensor != fun and expr.tensor notin fun.children: # TODO?
      fun.children.add(expr.tensor)

proc addKernel(fun: Fun, kernel: KernelBuilder) =
  if fun.kind notin {FunResult, FunEffect}:
    raise ParserError(msg: "Unable to add a kernel to a " & $fun.kind)
  fun.kernels.add(kernel)
  collectChildren(ExprBuilder(kernel.value), fun)

proc isName(node: NimNode): bool =
  result = node.kind == nnkIdent or node.kind == nnkSym

proc isName(node: NimNode, name: string): bool =
  result = node.isName and nimIdentNormalize(node.strVal) == nimIdentNormalize(name)

proc newDotExpr(node: NimNode, field: string): NimNode =
  result = newTree(nnkDotExpr, node, ident(field))

proc newBracketExpr(node, index: NimNode): NimNode =
  result = newTree(nnkBracketExpr, node, index)

proc newObjConstr(typ: NimNode, attrs: openArray[(string, NimNode)]): NimNode =
  result = newTree(nnkObjConstr, typ)
  for (name, value) in attrs:
    result.add(newTree(nnkExprColonExpr, ident(name), value))

type TargetInfo = object
  tensor: NimNode
  dims: seq[NimNode]
  isRaw: bool
  defineTensor: bool

proc parseTarget(node: NimNode): TargetInfo =
  case node.kind:
    of nnkInfix:
      assert node[0].eqIdent("*")
      result.defineTensor = true
      result.tensor = node[1]
      assert node[2].kind in {nnkCurly, nnkBracket}
      if node[2].kind == nnkCurly:
        result.isRaw = true
      for child in node[2]:
        result.dims.add(child)
    of nnkBracketExpr, nnkCurlyExpr:
      result.tensor = node[0]
      result.isRaw = node.kind == nnkCurlyExpr
      for it in 1..<node.len:
        result.dims.add(node[it])
    of nnkCall:
      if node[0].eqIdent("[]") or node[0].eqIdent("{}"):
        result.tensor = node[1]
        for it in 2..<node.len:
          result.dims.add(node[it])
      else:
        error("Invalid target: " & node.repr)
    else:
      error("Invalid target: " & node.repr)

type
  ScheduleDef = ref object
    tensors: Table[string, TensorSchedule]
    loops: Table[string, LoopSchedule]
  
  KernelAttrs = object
    hasCustomGrad: bool
    customGrad: seq[NimNode]
    schedules: array[CompileTarget, ScheduleDef]

proc lookupLoop(schedule: ScheduleDef, iterName: string): var LoopSchedule =
  let name = iterName.nimIdentNormalize()
  if name notin schedule.loops:
    schedule.loops[name] = DEFAULT_LOOP_SCHEDULE
  result = schedule.loops[name]

proc lookupTensor(schedule: ScheduleDef, tensorName: string): var TensorSchedule =
  let name = tensorName.nimIdentNormalize()
  if name notin schedule.tensors:
    schedule.tensors[name] = DEFAULT_TENSOR_SCHEDULE
  result = schedule.tensors[name]

const TARGET_NAMES = block:
  var names: seq[string] = @[]
  for target in ALL_COMPILE_TARGETS:
    names.add(($target)[len("Compile")..^1].toLowerAscii().nimIdentNormalize())
  names

proc parseSchedules(node: NimNode,
                     schedules: array[CompileTarget, ScheduleDef],
                     targets: set[CompileTarget]) =
  for child in node:
    if child.kind == nnkDiscardStmt:
      continue
    assert child.kind in nnkCallKinds
    assert child[0].isName
    
    template property(body: untyped) =
      for target in targets:
        let schedule {.inject.} = schedules[target]
        body
    
    let name = nimIdentNormalize(child[0].strVal)
    case name:
      of TARGET_NAMES:
        let target = parseEnum[CompileTarget]("Compile" & name)
        if target notin targets:
          raise ParserError(msg: "Unreachable schedule condition")
        child[^1].parseSchedules(schedules, {target})
      of "cache":
        property:
          schedule.lookupTensor(child[1].strVal).cache = true
      of "tile":
        property:
          schedule.lookupLoop(child[1].strVal).tile = true
      of "tilesize":
        property:
          schedule.lookupLoop(child[1].strVal).tileSize = child[2].intVal.int
      of "parallel":
        property:
          for it in 1..<child.len:
            let arg = child[it]
            assert arg.isName
            schedule.lookupLoop(arg.strVal).parallel = true
      of "sharecache":
        property:
          schedule.lookupLoop(child[1].strVal).shareCache = true
      else:
        raise ParserError(msg: "Unknown schedule property " & $name)

proc parseSchedules(node: NimNode): array[CompileTarget, ScheduleDef] =
  for target, schedule in result.mpairs:
    schedule = ScheduleDef()
  node.parseSchedules(result, ALL_COMPILE_TARGETS)

proc genKernelBuilder(target: TargetInfo, valueNode: NimNode, attrs: KernelAttrs): NimNode

proc parseAttrs(node: NimNode): KernelAttrs =
  for child in node:
    if child.kind in nnkCallKinds:
      assert child[0].isName
      case nimIdentNormalize(child[0].strVal):
        of "customgrad":
          result.hasCustomGrad = true
          for kernelNode in child[1]:
            if kernelNode.kind == nnkDiscardStmt:
              continue
            if kernelNode.kind != nnkInfix or not kernelNode[0].eqIdent("++=") or kernelNode.len < 3:
              raise ParserError(msg: "Custom gradient must be a valid kernel")
            
            var attrs = KernelAttrs()
            if kernelNode.len >= 4:
              attrs = kernelNode[3].parseAttrs()
            let target = kernelNode[1].parseTarget()
            result.customGrad.add(genKernelBuilder(target, kernelNode[2], attrs))
        of "schedule":
          result.schedules = child[^1].parseSchedules()
        else:
          raise ParserError(msg: $child[0].name & " is not a valid kernel attribute")
    elif child.kind != nnkDiscardStmt:
      raise ParserError(msg: $child.kind & " is not a valid kernel attribute")

proc newLit(schedule: ScheduleDef): NimNode =
  if schedule.isNil:
    return newObjConstr(bindSym("Schedule"), [])
  result = newNimNode(nnkStmtListExpr)
  let name = genSym()
  result.add(newLetStmt(name, newObjConstr(bindSym("Schedule"), {
    "loops": newLit(schedule.loops)
  })))
  for tensorName, tensorSchedule in schedule.tensors:
    result.add(newCall(bindSym("[]="),
      name.newDotExpr("tensors"),
      ident(tensorName),
      newLit(tensorSchedule)
    ))
  result.add(name)
  result = newTree(nnkBlockStmt, newEmptyNode(), result)

type Iterator = object
  name: string
  start: NimNode
  stop: NimNode

proc parseIterators(node: NimNode): seq[Iterator] = 
  case node.kind:
    of nnkSym, nnkIdent:
      result.add(Iterator(name: node.strVal))
    of nnkPar, nnkTupleConstr:
      for child in node:
        result.add(child.parseIterators())
    of nnkInfix:
      if not node[0].eqIdent("in") or not node[1].isName():
        raise ParserError()
      if not node[2][0].eqIdent("..<"):
        raise ParserError(msg: node[2][1].repr & " is not a valid iterator")
      result.add(Iterator(
        name: node[1].strVal,
        start: node[2][1],
        stop: node[2][2]
      ))
    else:
      raise ParserError(msg: node.repr & " is not a valid iterator")

proc genKernelBuilder(target: TargetInfo, valueNode: NimNode, attrs: KernelAttrs): NimNode =
  var dims: seq[NimNode] = @[]
  for dim in target.dims:
    dims.add(newCall(bindSym("literal"), dim))
  var
    value = valueNode
    iters: seq[Iterator] = @[]
  if value.kind == nnkInfix and value[0].eqIdent("|"):
    value = value[1]
    iters = valueNode[2].parseIterators()
  var fields = @{
    "target": target.tensor,
    "dims": newCall(bindSym("@"), newTree(nnkBracket, dims)),
    "isRaw": newLit(target.isRaw),
    "value": value,
    "schedules": newLit(attrs.schedules)
  }
  if attrs.hasCustomGrad:
    fields.add(("hasCustomGrad", newLit(true)))
    fields.add(("grads", newCall(bindSym("@"), newTree(nnkBracket, attrs.customGrad))))
  result = newNimNode(nnkStmtListExpr)
  for iter in iters:
    let call = newCall(bindSym("iteratorLiteral"), newLit(iter.name))
    if not iter.start.isNil:
      call.add(newTree(nnkExprEqExpr, ident("start"), iter.start))
    if not iter.stop.isNil:
      call.add(newTree(nnkExprEqExpr, ident("stop"), iter.stop))
    result.add(newLetStmt(ident(iter.name), call))
  result.add(newObjConstr(bindSym("KernelBuilder"), fields))
  result = newTree(nnkBlockStmt, newEmptyNode(), result)

proc genKernel(targetNode, value, attrsNode: NimNode): NimNode =
  let
    target = targetNode.parseTarget()
    attrs = attrsNode.parseAttrs()
    kernel = genKernelBuilder(target, value, attrs)
  result = newStmtList()
  if target.defineTensor:
    result.add(newVarStmt(target.tensor, newCall(bindSym("Fun"), newNilLit())))
  result.add(newCall(bindSym("ensureInit"), target.tensor))
  result.add(newCall(bindSym("addKernel"), target.tensor, kernel))

macro `++=`*(targetNode, value, attrsNode: untyped): untyped =
  result = genKernel(targetNode, value, attrsNode)

macro `++=`*(targetNode, value: untyped): untyped =
  result = genKernel(targetNode, value, newStmtList())

proc copyShape*(fun, src: Fun) =
  if fun.kind != FunResult:
    raise ParserError(msg: "Cannot set shape of " & $fun.kind)
  fun.shapeConstr = ShapeConstraintBuilder(kind: ShapeCopy, copy: src)
  if src notin fun.children:
    fun.children.add(src)

proc withShape*(fun: Fun, dims: varargs[Index, literal]) =
  if fun.kind != FunResult:
    raise ParserError(msg: "Cannot set shape of " & $fun.kind)
  fun.shapeConstr = ShapeConstraintBuilder(kind: ShapeDims, dims: @dims)
  for dim in dims:
    collectChildren(ExprBuilder(dim), fun)

macro layer*(fn: untyped): untyped =
  result = fn
  var name = fn.name
  while not name.isName():
    case name.kind:
      of nnkPostfix: name = name[1]
      else:
        raise newException(ValueError, "Unable to extract name from " & $name.kind)
  result.body.add(newAssignment(
    newTree(nnkDotExpr, ident("result"), ident("name")),
    newLit(name.strVal)
  ))

proc lock*(fun: Fun) =
  fun.locked = true

proc param*(shape: openArray[int],
            initRange: HSlice[float64, float64] = -0.1..0.1,
            name: string = ""): Fun =
  result = Fun(kind: FunParam,
    initRange: initRange,
    paramShape: newSeq[int](shape.len),
    name: name
  )
  for dim, size in shape:
    result.paramShape[dim] = size

proc input*(name: string, shape: openArray[int] = []): Fun =
  result = Fun(kind: FunInput,
    name: name,
    inputShape: newSeq[int](shape.len)
  )
  for dim, size in shape:
    result.inputShape[dim] = size

proc rand*(fun: Fun, randRange: HSlice[float64, float64]): Fun =
  result = Fun(kind: FunRandom,
    children: @[fun],
    randomRange: randRange
  )

proc backwards*(fun: Fun): Fun =
  result = Fun(kind: FunBackwards, children: @[fun])

proc params*(fun: Fun, stop: HashSet[string]): HashSet[Fun] =
  if fun.kind != FunTarget or fun.name notin stop:
    for child in fun.children:
      result = result.union(child.params(stop))
    case fun.kind:
      of FunParam: result.incl(fun)
      of FunCond:
        for target, child in fun.cond:
          result = result.union(child.params(stop))
        if not fun.condElse.isNil:
          result = result.union(fun.condElse.params(stop))
      else: discard

proc params*(fun: Fun, stop: openArray[string] = []): HashSet[Fun] =
  result = fun.params(toHashSet(stop))

proc optimize*(gradients: Fun,
               params: HashSet[Fun],
               optim: proc (param: var Fun, grad: Fun)): Fun =
  result = Fun(kind: FunMultiple)
  for param in params:
    var
      effect = Fun(kind: FunEffect, effect: param)
      grad = Fun(kind: FunGradient, children: @[gradients, param])
    optim(effect, grad)
    result.children.add(effect)

proc optimize*(gradients: Fun, optim: proc (param: var Fun, grad: Fun)): Fun =
  result = gradients.optimize(gradients.params, optim)

proc optimize*(grad, param: Fun, optim: proc (param: var Fun, grad: Fun)): Fun =
  result = Fun(kind: FunMultiple)
  var effect = Fun(kind: FunEffect, effect: param)
  optim(effect, grad)
  result.children.add(effect)

proc backprop*(loss: Fun, optim: proc (param: var Fun, grad: Fun)): Fun =
  result = loss.backwards().optimize(optim)

proc grad*(gradients, fun: Fun): Fun =
  result = Fun(kind: FunGradient, children: @[gradients, fun])

proc grad*(fun: Fun): Fun =
  result = Fun(kind: FunGradientArg, children: @[fun])

proc reshape*(fun: Fun, shape: openArray[int]): Fun =
  result = Fun(kind: FunReshape,
    name: "reshape",
    children: @[fun],
    reshape: newSeq[int](shape.len)
  )
  for dim, size in shape:
    result.reshape[dim] = size

proc cache*(cache: Fun, name: string = ""): Fun =
  result = Fun(kind: FunEffect,
    effect: Fun(kind: FunCache, cache: cache, name: name)
  )

proc target*(fun: Fun,
             name: string,
             compileTarget = CompileThreads): Fun =
  result = Fun(kind: FunTarget,
    name: name,
    children: @[fun],
    compileTarget: compileTarget
  )
  when not TARGET_SUPPORTS_THREADS:
    if compileTarget == CompileThreads:
      result.compileTarget = CompileCpu

proc cond*(branches: openArray[(string, Fun)],
           otherwise: Fun = nil): Fun =
  result = Fun(kind: FunCond,
    cond: toTable(branches),
    condElse: otherwise
  )

macro makeOpt*(opt: typed, args: varargs[untyped]): untyped =
  result = newTree(nnkCall, opt, ident("param"), ident("grad"))
  for arg in args:
    result.add(arg)
  
  result = newProc(
    params=[
      newEmptyNode(),
      newIdentDefs(ident("param"), newTree(nnkVarTy, bindSym("Fun"))),
      newIdentDefs(ident("grad"), bindSym("Fun"))
    ],
    body=newStmtList(result)
  )
