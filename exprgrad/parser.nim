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
          of InstrIndex: index_lit*: int
          of InstrScalar: scalar_lit*: float64
          of InstrBoolean: boolean_lit*: bool
          of InstrShape: dim*: int
          else: discard
      of ExprRead:
        is_raw*: bool
      else: discard
  
  Scalar* = distinct ExprBuilder
  Index* = distinct ExprBuilder
  Boolean* = distinct ExprBuilder
  Array*[T] = distinct ExprBuilder
  
  Schedule = ref object
    discard
  
  KernelBuilder = ref object
    target: Fun
    dims: seq[Index]
    is_raw: bool
    value: Scalar
    has_custom_grad: bool
    grads: seq[KernelBuilder]
    block_count: int
    schedule: Schedule
  
  ShapeConstraintBuilder* = object
    case kind: ShapeConstrKind:
      of ShapeNone, ShapeLinear: discard
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
        input_shape: seq[int]
      of FunParam:
        param_shape: seq[int]
        init_range: HSlice[float64, float64]
      of FunCache: cache: Fun
      of FunRandom: random_range: HSlice[float64, float64]
      of FunResult, FunEffect:
        kernels: seq[KernelBuilder]
        shape_constr: ShapeConstraintBuilder
        effect: Fun
      of FunReshape: reshape: seq[int]
      of FunCond:
        cond: Table[string, Fun]
        cond_else: Fun
      of FunTarget:
        compile_target: CompileTarget
      of FunBackwards, FunGradient, FunMultiple, FunGradientArg:
        discard

proc literal*(value: bool): Boolean =
  result = Boolean(ExprBuilder(kind: ExprInstr, instr: InstrBoolean, boolean_lit: value))

proc literal*(value: int): Index =
  result = Index(ExprBuilder(kind: ExprInstr, instr: InstrIndex, index_lit: value))

proc literal*(value: float): Scalar =
  result = Scalar(ExprBuilder(kind: ExprInstr, instr: InstrScalar, scalar_lit: value))

proc literal*(value: Index): Index = value
proc literal*(value: Scalar): Scalar = value
proc literal*(value: Boolean): Boolean = value
proc literal*[T](value: Array[T]): Array[T] = value

proc literal*[T](arr: openArray[T]): auto =
  let builder = ExprBuilder(kind: ExprInstr, instr: InstrArray)
  for value in arr.items:
    builder.children.add(ExprBuilder(literal(value)))
  result = Array[typeof(literal(arr[0]))](builder)

proc iterator_literal*(name: string): Index =
  result = Index(ExprBuilder(kind: ExprIter, iter: name))

type BuildContext = object
  kernel: Kernel
  iters: Table[string, RegId]
  grads: Table[TensorId, TensorId]
  block_count: int
  max_tensor: TensorId

proc alloc_block(ctx: var BuildContext): int =
  result = ctx.block_count
  ctx.block_count += 1

proc lookup_tensor(ctx: var BuildContext, fun: Fun): TensorId =
  if fun.kind == FunGradientArg:
    let id = ctx.lookup_tensor(fun.children[0])
    if id notin ctx.grads:
      ctx.grads[id] = TensorId(-ctx.grads.len - 1)
    result = ctx.grads[id]
  else:
    result = fun.tensor

proc build*(builder: ExprBuilder,
            instrs: var seq[Instr],
            block_id: int,
            ctx: var BuildContext): RegId

proc build_linear_index*(builder: ExprBuilder, ctx: var BuildContext): LinearIndex =
  let reg = builder.build(result.setup, ctx.alloc_block(), ctx)
  result.factors = to_table({reg: 1})

proc build*(builder: ExprBuilder,
            instrs: var seq[Instr],
            block_id: int,
            ctx: var BuildContext): RegId =
  if block_id notin builder.res:
    case builder.kind:
      of ExprRead:
        var dims: seq[LinearIndex] = @[]
        for dim in builder.children:
          dims.add(dim.build_linear_index(ctx))
        
        let res = ctx.kernel.regs.alloc()
        ctx.kernel.reads.add(TensorOp(
          tensor: ctx.lookup_tensor(builder.tensor),
          is_raw: builder.is_raw,
          dims: dims,
          data: res
        ))
        builder.res[block_id] = res
      of ExprIter:
        if builder.iter notin ctx.iters:
          let reg = ctx.kernel.regs.alloc()
          ctx.iters[builder.iter] = reg
          ctx.kernel.loops.add(Loop(iter: reg))
        builder.res[block_id] = ctx.iters[builder.iter]
      of ExprInstr:
        var instr = Instr(kind: builder.instr)
        for child in builder.children:
          instr.args.add(child.build(instrs, block_id, ctx))
        
        if not builder.tensor.is_nil:
          instr.tensor = ctx.lookup_tensor(builder.tensor)
        
        case builder.instr:
          of InstrIndex: instr.index_lit = builder.index_lit
          of InstrScalar: instr.scalar_lit = builder.scalar_lit
          of InstrBoolean: instr.boolean_lit = builder.boolean_lit
          of InstrShape: instr.dim = builder.dim
          else: discard 
        
        instr.res = ctx.kernel.regs.alloc()
        builder.res[block_id] = instr.res
        instrs.add(instr)
  
  result = builder.res[block_id]

proc build_expr(builder: ExprBuilder, ctx: var BuildContext): Expr =
  result.res = builder.build(result.instrs, ctx.alloc_block(), ctx)

proc clear(expr: ExprBuilder) =
  for child in expr.children:
    child.clear()
  expr.res = init_table[int, RegId]()

proc clear(builder: KernelBuilder) =
  ExprBuilder(builder.value).clear()
  for dim in builder.dims:
    ExprBuilder(dim).clear()

proc build(builder: KernelBuilder, ctx: var BuildContext): Kernel =
  result = Kernel()
  ctx.kernel = result
  result.expr = ExprBuilder(builder.value).build_expr(ctx)
  result.write = TensorOp(
    tensor: ctx.lookup_tensor(builder.target),
    is_raw: builder.is_raw,
    data: result.expr.res
  )
  for dim in builder.dims:
    result.write.dims.add(ExprBuilder(dim).build_linear_index(ctx))
  if builder.has_custom_grad:
    result.grad = KernelGradient(is_custom: true)
    var grads = init_table[TensorId, TensorId]()
    for grad in builder.grads:
      grad.clear()
      var grad_ctx = BuildContext(grads: grads)
      result.grad.kernels.add(grad.build(grad_ctx))
      grads = grad_ctx.grads
    result.grad.tensors = grads

proc build(builder: KernelBuilder): Kernel =
  builder.clear()
  var ctx = BuildContext()
  result = builder.build(ctx)

proc hash(fun: Fun): Hash = hash(fun[].addr)

proc alloc_tensors(fun: Fun, program: Program) =
  if fun.tensor == TensorId(0):
    case fun.kind:
      of FunInput:
        if fun.name notin program.inputs:
          program.inputs[fun.name] = program.tensors.alloc(TensorDef(
            kind: TensorInput,
            shape: fun.input_shape,
            name: fun.name
          ))
        fun.tensor = program.inputs[fun.name]
        if program.tensors[fun.tensor].shape != fun.input_shape:
          raise ParserError(msg: "Expected shapes for input \"" & fun.name & "\" do not match.")
      of FunParam:
        fun.tensor = program.tensors.alloc(TensorDef(
          kind: TensorParam,
          shape: fun.param_shape,
          init_range: fun.init_range,
          name: fun.name
        ))
      of FunRandom:
        fun.tensor = program.tensors.alloc(TensorDef(
          kind: TensorRandom,
          random_range: fun.random_range,
          name: fun.name
        ))
      of FunResult, FunGradient, FunReshape:
        fun.tensor = program.tensors.alloc(TensorDef(
          kind: TensorResult,
          name: fun.name
        ))
      of FunEffect:
        fun.effect.alloc_tensors(program)
        fun.tensor = fun.effect.tensor
      of FunCache:
        fun.cache.alloc_tensors(program)
        fun.tensor = program.tensors.alloc(TensorDef(
          kind: TensorCache,
          cache: fun.cache.tensor,
          name: fun.name
        ))
      of FunCond:
        fun.tensor = TensorId(0)
        for target, child in fun.cond:
          child.alloc_tensors(program)
        if not fun.cond_else.is_nil:
          fun.cond_else.alloc_tensors(program)
      else: discard
    
    for child in fun.children:
      child.alloc_tensors(program)
    
    case fun.kind:
      of FunTarget: fun.tensor = fun.children[0].tensor
      else: discard

proc flatten(fun: Fun, target: var Target) =
  if target.name notin fun.targets:
    for child in fun.children:
      child.flatten(target)
    if fun.kind == FunEffect:
      fun.effect.flatten(target)
    
    fun.targets.incl(target.name)
    case fun.kind:
      of FunResult, FunEffect:
        for kernel in fun.kernels:
          target.kernels.add(kernel.build())
        case fun.shape_constr.kind:
          of ShapeCopy:
            target.shapes.add(ShapeConstraint(kind: ShapeCopy,
              dest: fun.tensor,
              src: fun.shape_constr.copy.tensor
            ))
          of ShapeDims:
            var constr = ShapeConstraint(kind: ShapeDims, dest: fun.tensor)
            for dim in fun.shape_constr.dims:
              var ctx = BuildContext(kernel: Kernel())
              ExprBuilder(dim).clear()
              constr.dims.add(ExprBuilder(dim).build_linear_index(ctx))
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
        elif not fun.cond_else.is_nil:
          child = fun.cond_else
        else:
          raise ParserError(msg: "Conditional node does not have a branch for the target \"" & target.name & "\"")
        child.flatten(target)
        fun.tensor = child.tensor
      of FunRandom:
        target.shapes.add(ShapeConstraint(kind: ShapeCopy,
          dest: fun.tensor, src: fun.children[0].tensor
        ))
      else: discard

proc collect_targets(fun: Fun, targets: var Table[string, Fun]) =
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
        child.collect_targets(targets)
      if not fun.cond_else.is_nil:
        fun.cond_else.collect_targets(targets)
    else: discard
  for child in fun.children:
    child.collect_targets(targets)

proc to_program*(graphs: openArray[Fun]): Program =
  result = Program()
  var targets = init_table[string, Fun]()
  for fun in graphs:
    fun.alloc_tensors(result)
    fun.collect_targets(targets)
  for name, fun in targets:
    var target = Target(
      name: name,
      output: fun.tensor,
      compile_target: fun.compile_target
    )
    fun.flatten(target)
    result.targets[name] = target

proc `$`(fun: Fun): string =
  if fun.is_nil:
    result = "nil"
  else:
    result = "<fun>"

proc ensure_init(fun: var Fun) =
  if fun.is_nil:
    fun = Fun(kind: FunResult)

proc collect_children(expr: ExprBuilder, fun: Fun) =
  for child in expr.children:
    child.collect_children(fun)
  if not expr.tensor.is_nil:
    if expr.tensor != fun and expr.tensor notin fun.children: # TODO?
      fun.children.add(expr.tensor)

proc add_kernel(fun: Fun, kernel: KernelBuilder) =
  if fun.kind notin {FunResult, FunEffect}:
    raise ParserError(msg: "Unable to add a kernel to a " & $fun.kind)
  fun.kernels.add(kernel)
  collect_children(ExprBuilder(kernel.value), fun)

proc is_name(node: NimNode): bool =
  result = node.kind == nnkIdent or node.kind == nnkSym

proc is_name(node: NimNode, name: string): bool =
  result = node.is_name and nim_ident_normalize(node.str_val) == nim_ident_normalize(name)

proc new_obj_constr(typ: NimNode, attrs: openArray[(string, NimNode)]): NimNode =
  result = new_tree(nnkObjConstr, typ)
  for (name, value) in attrs:
    result.add(new_tree(nnkExprColonExpr, ident(name), value))

type TargetInfo = object
  tensor: NimNode
  dims: seq[NimNode]
  is_raw: bool
  define_tensor: bool

proc parse_target(node: NimNode): TargetInfo =
  case node.kind:
    of nnkInfix:
      assert node[0].is_name("*")
      result.define_tensor = true
      result.tensor = node[1]
      assert node[2].kind in {nnkCurly, nnkBracket}
      if node[2].kind == nnkCurly:
        result.is_raw = true
      for child in node[2]:
        result.dims.add(child)
    of nnkBracketExpr, nnkCurlyExpr:
      result.tensor = node[0]
      result.is_raw = node.kind == nnkCurlyExpr
      for it in 1..<node.len:
        result.dims.add(node[it])
    else:
      error("Invalid target: " & node.repr)

type KernelAttrs = object
  has_custom_grad: bool
  custom_grad: seq[NimNode]
  schedule: Schedule

proc parse_schedule(node: NimNode): Schedule =
  result = Schedule()

proc gen_kernel_builder(target: TargetInfo, value_node: NimNode, attrs: KernelAttrs): NimNode

proc parse_attrs(node: NimNode): KernelAttrs =
  for child in node:
    if child.kind in nnkCallKinds:
      assert child[0].is_name
      case nim_ident_normalize(child[0].str_val):
        of "customgrad":
          result.has_custom_grad = true
          for kernel_node in child[1]:
            if kernel_node.kind == nnkDiscardStmt:
              continue
            if kernel_node.kind != nnkInfix or not kernel_node[0].is_name("++=") or kernel_node.len < 3:
              raise ParserError(msg: "Custom gradient must be a valid kernel")
            
            var attrs = KernelAttrs()
            if kernel_node.len >= 4:
              attrs = kernel_node[3].parse_attrs()
            let target = kernel_node[1].parse_target()
            result.custom_grad.add(gen_kernel_builder(target, kernel_node[2], attrs))
        of "schedule":
          result.schedule = node[^1].parse_schedule()
        else:
          raise ParserError(msg: $child[0].name & " is not a valid kernel attribute")
    elif child.kind != nnkDiscardStmt:
      raise ParserError(msg: $child.kind & " is not a valid kernel attribute")

proc gen_kernel_builder(target: TargetInfo, value_node: NimNode, attrs: KernelAttrs): NimNode =
  var dims: seq[NimNode] = @[]
  for dim in target.dims:
    dims.add(new_call(bind_sym("literal"), dim))
  var
    value = value_node
    iters: seq[string] = @[]
  if value.kind == nnkInfix and value[0].is_name("|"):
    value = value[1]
    let iters_node = value_node[2]
    case iters_node.kind:
      of nnkSym, nnkIdent:
        iters.add(iters_node.str_val)
      of nnkPar, nnkTupleConstr:
        for child in iters_node:
          assert child.is_name
          iters.add(child.str_val)
      else:
        raise ParserError(msg: iters_node.repr & " is not a valid iterator")
  var fields = @{
    "target": target.tensor,
    "dims": new_call(bind_sym("@"), new_tree(nnkBracket, dims)),
    "is_raw": new_lit(target.is_raw),
    "value": value
  }
  if attrs.has_custom_grad:
    fields.add(("has_custom_grad", new_lit(true)))
    fields.add(("grads", new_call(bind_sym("@"), new_tree(nnkBracket, attrs.custom_grad))))
  result = new_nim_node(nnkStmtListExpr)
  for iter in iters:
    result.add(new_let_stmt(ident(iter),
      new_call(bind_sym("iterator_literal"), new_lit(iter))
    ))
  result.add(new_obj_constr(bind_sym("KernelBuilder"), fields))
  result = new_tree(nnkBlockStmt, new_empty_node(), result)

proc gen_kernel(target_node, value, attrs_node: NimNode): NimNode =
  let
    target = target_node.parse_target()
    attrs = attrs_node.parse_attrs()
    kernel = gen_kernel_builder(target, value, attrs)
  result = new_stmt_list()
  if target.define_tensor:
    result.add(new_var_stmt(target.tensor, new_call(bind_sym("Fun"), new_nil_lit())))
  result.add(new_call(bind_sym("ensure_init"), target.tensor))
  result.add(new_call(bind_sym("add_kernel"), target.tensor, kernel))

macro `++=`*(target_node, value, attrs_node: untyped): untyped =
  result = gen_kernel(target_node, value, attrs_node)

macro `++=`*(target_node, value: untyped): untyped =
  result = gen_kernel(target_node, value, new_stmt_list())

proc copy_shape*(fun, src: Fun) =
  if fun.kind != FunResult:
    raise ParserError(msg: "Cannot set shape of " & $fun.kind)
  fun.shape_constr = ShapeConstraintBuilder(kind: ShapeCopy, copy: src)
  if src notin fun.children:
    fun.children.add(src)

proc with_shape*(fun: Fun, dims: varargs[Index, literal]) =
  if fun.kind != FunResult:
    raise ParserError(msg: "Cannot set shape of " & $fun.kind)
  fun.shape_constr = ShapeConstraintBuilder(kind: ShapeDims, dims: @dims)
  for dim in dims:
    collect_children(ExprBuilder(dim), fun)

macro layer*(fn: untyped): untyped =
  result = fn
  var name = fn.name
  while not name.is_name():
    case name.kind:
      of nnkPostfix: name = name[1]
      else:
        raise new_exception(ValueError, "Unable to extract name from " & $name.kind)
  result.body.add(new_assignment(
    new_tree(nnkDotExpr, ident("result"), ident("name")),
    new_lit(name.str_val)
  ))

proc lock*(fun: Fun) =
  fun.locked = true

proc param*(shape: openArray[int],
            init_range: HSlice[float64, float64] = -0.1..0.1): Fun =
  result = Fun(kind: FunParam,
    init_range: init_range,
    param_shape: new_seq[int](shape.len)
  )
  for dim, size in shape:
    result.param_shape[dim] = size

proc input*(name: string, shape: openArray[int] = []): Fun =
  result = Fun(kind: FunInput,
    name: name,
    input_shape: new_seq[int](shape.len)
  )
  for dim, size in shape:
    result.input_shape[dim] = size

proc rand*(fun: Fun, rand_range: HSlice[float64, float64]): Fun =
  result = Fun(kind: FunRandom,
    children: @[fun],
    random_range: rand_range
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
        if not fun.cond_else.is_nil:
          result = result.union(fun.cond_else.params(stop))
      else: discard

proc params*(fun: Fun, stop: openArray[string] = []): HashSet[Fun] =
  result = fun.params(to_hash_set(stop))

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
    reshape: new_seq[int](shape.len)
  )
  for dim, size in shape:
    result.reshape[dim] = size

proc cache*(cache: Fun, name: string = ""): Fun =
  result = Fun(kind: FunEffect,
    effect: Fun(kind: FunCache, cache: cache, name: name)
  )

proc target*(fun: Fun,
             name: string,
             compile_target = CompileThreads): Fun =
  result = Fun(kind: FunTarget,
    name: name,
    children: @[fun],
    compile_target: compile_target
  )
  when not TARGET_SUPPORTS_THREADS:
    if compile_target == CompileThreads:
      result.compile_target = CompileCpu

proc cond*(branches: openArray[(string, Fun)],
           otherwise: Fun = nil): Fun =
  result = Fun(kind: FunCond,
    cond: to_table(branches),
    cond_else: otherwise
  )

macro make_opt*(opt: typed, args: varargs[untyped]): untyped =
  result = new_tree(nnkCall, opt, ident("param"), ident("grad"))
  for arg in args:
    result.add(arg)
  
  result = new_proc(
    params=[
      new_empty_node(),
      new_ident_defs(ident("param"), new_tree(nnkVarTy, bind_sym("Fun"))),
      new_ident_defs(ident("grad"), bind_sym("Fun"))
    ],
    body=new_stmt_list(result)
  )
