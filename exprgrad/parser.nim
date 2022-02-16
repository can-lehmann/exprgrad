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
        args: Table[Fun, TensorId]
        kernels: seq[Kernel]
        shape_constr: ShapeConstraint
        effect: Fun
      of FunReshape: reshape: seq[int]
      of FunCond:
        cond: Table[string, Fun]
        cond_else: Fun
      of FunTarget:
        compile_target: CompileTarget
      of FunBackwards, FunGradient, FunMultiple, FunGradientArg:
        discard

proc hash(fun: Fun): Hash = hash(fun[].addr)

proc alloc_tensors(fun: Fun, program: Program) =
  if fun.tensor == TensorId(0):
    case fun.kind:
      of FunInput:
        fun.tensor = program.tensors.alloc(TensorDef(
          kind: TensorInput,
          shape: fun.input_shape,
          name: fun.name
        ))
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
        var subs = init_table[TensorId, TensorId]()
        for arg, local_tensor_id in fun.args:
          subs[local_tensor_id] = arg.tensor
        for kernel in fun.kernels:
          let target_kernel = kernel.clone()
          target_kernel.substitute(subs)
          target.kernels.add(target_kernel)
        if fun.shape_constr.kind != ShapeNone:
          var constr = fun.shape_constr
          constr.substitute(subs)
          target.shapes.add(constr)
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
  if fun.args.len == 0:
    fun.args[fun] = TensorId(1)

proc register_args(fun: Fun, args: openArray[(Fun, TensorId)]): Table[TensorId, TensorId] =
  for (arg, id) in args:
    if arg notin fun.args:
      fun.args[arg] = TensorId(fun.args.len + 1)
      fun.children.add(arg)
    result[id] = fun.args[arg]

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

proc literal*(value: bool): Boolean =
  result = Boolean(ExprBuilder(kind: ExprInstr, instr: InstrBoolean, boolean_lit: value))

proc literal*(value: int): Index =
  result = Index(ExprBuilder(kind: ExprInstr, instr: InstrIndex, index_lit: value))

proc literal*(value: float): Scalar =
  result = Scalar(ExprBuilder(kind: ExprInstr, instr: InstrScalar, scalar_lit: value))

proc literal*(value: Index): Index = value
proc literal*(value: Scalar): Scalar = value
proc literal*(value: Boolean): Boolean = value

proc iterator_literal*(name: string): Index =
  result = Index(ExprBuilder(kind: ExprIter, iter: name))

proc clear*(builder: ExprBuilder) =
  builder.res = init_table[int, RegId]()
  for child in builder.children:
    child.clear()

type BuildContext = object
  kernel: Kernel
  tensors: Table[Fun, TensorId]
  iters: Table[string, RegId]
  grads: Table[TensorId, TensorId]
  block_count: int

proc alloc_block(ctx: var BuildContext): int =
  result = ctx.block_count
  ctx.block_count += 1

proc lookup_tensor(ctx: var BuildContext, tensor: Fun): TensorId =
  if tensor.kind == FunGradientArg:
    let id = ctx.lookup_tensor(tensor.children[0])
    if id notin ctx.grads:
      ctx.grads[id] = TensorId(ctx.grads.len + ctx.tensors.len + 1)
    result = ctx.grads[id]
  else:
    if tensor notin ctx.tensors:
      ctx.tensors[tensor] = TensorId(ctx.grads.len + ctx.tensors.len + 1)
    result = ctx.tensors[tensor]

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

proc register_args(fun: Fun, ctx: BuildContext) =
  for arg, id in ctx.tensors:
    if arg notin fun.args:
      fun.args[arg] = id
      fun.children.add(arg)

type KernelBuilder = ref object
  target: Fun
  dims: seq[Index]
  is_raw: bool
  value: Scalar
  has_custom_grad: bool
  grads: seq[KernelBuilder]

proc with_custom_grad(builder: KernelBuilder, grads: openArray[KernelBuilder]): KernelBuilder =
  result = builder
  result.has_custom_grad = true
  for grad in grads:
    result.grads.add(grad)

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
      var grad_ctx = BuildContext(tensors: ctx.tensors, grads: grads)
      result.grad.kernels.add(grad.build(grad_ctx))
      if grad_ctx.tensors != ctx.tensors:
        raise ParserError(msg: "Gradient kernel may only use tensors (and their gradients) which the forward pass also uses.")
      grads = grad_ctx.grads
    result.grad.tensors = grads

proc add_kernel(builder: KernelBuilder) =
  if builder.target.locked:
    raise ParserError(msg: "Unable to add another kernel to a locked function")
  if builder.target.kind notin {FunResult, FunEffect}:
    raise ParserError(msg: "Unable to add a kernel to " & $builder.target.kind)
  
  var ctx = BuildContext(tensors: builder.target.args)
  builder.target.kernels.add(builder.build(ctx))
  builder.target.register_args(ctx)

proc is_name(node: NimNode): bool =
  result = node.kind == nnkIdent or node.kind == nnkSym

proc is_name(node: NimNode, name: string): bool =
  result = node.is_name and nim_ident_normalize(node.str_val) == nim_ident_normalize(name)

proc new_obj_constr(typ: NimNode, attrs: openArray[(string, NimNode)]): NimNode =
  result = new_tree(nnkObjConstr, typ)
  for (name, value) in attrs:
    result.add(new_tree(nnkExprColonExpr, ident(name), value))

proc wrap_expr(node: NimNode, locals: var HashSet[string]): NimNode =
  case node.kind:
    of nnkSym, nnkIdent:
      if node.str_val == "true" or node.str_val == "false":
        result = new_call(bind_sym("literal"), node)
      elif nim_ident_normalize(node.str_val) in locals:
        result = node
      else:
        result = new_call(bind_sym("iterator_literal"), new_lit(node.str_val))
    of nnkIntLit: result = new_call(bind_sym("literal"), node)
    of nnkFloatLit: result = new_call(bind_sym("literal"), node)
    of nnkCallKinds:
      assert node.len > 0
      if node[0].is_name("@"):
        result = new_call(bind_sym("literal"), node[1])
      else:
        result = new_nim_node(node.kind)
        result.add(node[0])
        for it in 1..<node.len:
          result.add(node[it].wrap_expr(locals))
    of nnkBracketExpr, nnkCurlyExpr:
      result = new_nim_node(node.kind)
      result.add(node[0])
      for it in 1..<node.len:
        result.add(node[it].wrap_expr(locals))
    of nnkLetSection:
      result = new_nim_node(nnkLetSection)
      for def in node:
        for it in 0..<(def.len - 2):
          assert def[it].is_name()
          locals.incl(nim_ident_normalize(def[it].str_val))
        result.add(new_tree(nnkIdentDefs, def[0..^3] & @[
          def[^2], def[^1].wrap_expr(locals)
        ]))
    else:
      result = new_nim_node(node.kind)
      for child in node:
        result.add(child.wrap_expr(locals))

macro quote_expr*(expr: untyped): untyped =
  var locals = init_hash_set[string]()
  result = expr.wrap_expr(locals)

type TargetInfo = object
  tensor: NimNode
  dims: seq[NimNode]
  is_raw: bool
  define_tensor: bool

proc parse_target(node: NimNode, locals: var HashSet[string]): TargetInfo =
  case node.kind:
    of nnkInfix:
      assert node[0].is_name("*")
      result.define_tensor = true
      result.tensor = node[1]
      assert node[2].kind in {nnkCurly, nnkBracket}
      if node[2].kind == nnkCurly:
        result.is_raw = true
      for child in node[2]:
        result.dims.add(child.wrap_expr(locals))
    of nnkBracketExpr, nnkCurlyExpr:
      result.tensor = node[0]
      result.is_raw = node.kind == nnkCurlyExpr
      for it in 1..<node.len:
        result.dims.add(node[it].wrap_expr(locals))
    else:
      error("Invalid target: " & node.repr)

proc extract_attributes(node: NimNode, attrs: var seq[NimNode]) =
  case node.kind:
    of nnkPar: node[0].extract_attributes(attrs)
    of nnkTupleConstr:
      for child in node:
        child.extract_attributes(attrs)
    of nnkCallKinds:
      attrs.add(node)
    else:
      raise ParserError(msg: node.repr & " is not a valid kernel attribute")

proc remove_attributes(node: NimNode): tuple[expr: NimNode, attrs: seq[NimNode]] =
  result.expr = node
  while result.expr.kind in nnkCallKinds and
        result.expr[0].is_name("|"):
    result.expr[2].extract_attributes(result.attrs)
    result.expr = result.expr[1]

proc gen_kernel_builder(target: TargetInfo, value: NimNode, attrs: seq[NimNode]): NimNode

proc gen_custom_grad(node: NimNode): NimNode =
  if node.kind != nnkInfix or not node[0].is_name("++=") or node.len != 3:
    raise ParserError(msg: "Custom gradient must be a valid kernel")
  
  var locals = init_hash_set[string]()
  let
    target = node[1].parse_target(locals)
    (value_node, attrs) = node[2].remove_attributes()
    value = value_node.wrap_expr(locals)
  
  result = gen_kernel_builder(target, value, attrs)

proc gen_kernel_builder(target: TargetInfo, value: NimNode, attrs: seq[NimNode]): NimNode =
  result = new_obj_constr(bind_sym("KernelBuilder"), {
    "target": target.tensor,
    "dims": new_call(bind_sym("@"), new_tree(nnkBracket, target.dims)),
    "is_raw": new_lit(target.is_raw),
    "value": value
  })
  for attr in attrs:
    assert attr[0].is_name
    case nim_ident_normalize(attr[0].str_val):
      of "customgrad":
        var grads: seq[NimNode] = @[]
        for it in 1..<attr.len:
          grads.add(attr[it].gen_custom_grad())
        result = new_call(bind_sym("with_custom_grad"), result, new_tree(nnkBracket, grads))
      else:
        result = new_call(attr[0], @[result] & attr[1..^1])

macro `++=`*(target_node, body_node: untyped): untyped =
  var locals = init_hash_set[string]()
  let
    target = target_node.parse_target(locals)
    (value_node, attrs) = body_node.remove_attributes()
    value = value_node.wrap_expr(locals)
  
  result = new_stmt_list()
  if target.define_tensor:
    result.add(new_var_stmt(target.tensor, new_call(bind_sym("Fun"), new_nil_lit())))
  result.add(new_call(bind_sym("ensure_init"), target.tensor))
  result.add(new_call(bind_sym("add_kernel"), gen_kernel_builder(target, value, attrs)))

proc use_shape(fun: Fun, shape: ShapeConstraint, args: openArray[(Fun, TensorId)]) =
  if fun.kind != FunResult:
    raise ParserError(msg: "Cannot set shape of " & $fun.kind)
  fun.shape_constr = shape
  let subs = fun.register_args(args)
  fun.shape_constr.substitute(subs)

proc copy_shape*(fun, src: Fun) =
  fun.use_shape(
    ShapeConstraint(kind: ShapeCopy, dest: TensorId(1), src: TensorId(2)),
    [(fun, TensorId(1)), (src, TensorId(2))]
  )

proc use_shape(fun: Fun, dims: openArray[Index]) =
  if fun.kind != FunResult:
    raise ParserError(msg: "Cannot set shape of " & $fun.kind)
  var
    ctx = BuildContext(tensors: fun.args)
    shape = ShapeConstraint(kind: ShapeDims)
  for dim in dims:
    shape.dims.add(ExprBuilder(dim).build_linear_index(ctx))
  fun.register_args(ctx)
  fun.shape_constr = shape

macro with_shape*(target, dim_nodes: untyped): untyped =
  if dim_nodes.kind != nnkBracket:
    raise ParserError(msg: "The second argument to with_shape must be of the format [dim0, dim1, ...]")
  
  var dims = new_nim_node(nnkBracket)
  for dim in dim_nodes:
    var locals = init_hash_set[string]()
    dims.add(dim.wrap_expr(locals))
  
  result = new_stmt_list([
    new_call(bind_sym("ensure_init"), target),
    new_call(bind_sym("use_shape"), target, dims)
  ])

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

proc target*(fun: Fun, name: string): Fun =
  result = Fun(kind: FunTarget, name: name, children: @[fun])
  when TARGET_SUPPORTS_THREADS:
    result.compile_target = CompileThreads
  else:
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
