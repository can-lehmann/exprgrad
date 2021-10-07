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
    FunReshape, FunTarget, FunCond
  
  Fun* = ref object
    targets: HashSet[string]
    tensor*: TensorId
    children: seq[Fun]
    name*: string
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
      of FunBackwards, FunGradient, FunMultiple:
        discard

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

type ParserContext = object
  kernel: Kernel
  tensor_lookup: Table[string, TensorId]
  tensors: seq[NimNode]
  indices: Table[string, RegId]

proc hash(fun: Fun): Hash = hash(fun[].addr)

proc register_tensor(ctx: var ParserContext, node: NimNode): TensorId =
  result = TensorId(ctx.tensors.len + 1)
  ctx.tensors.add(node)

proc lookup_tensor(ctx: var ParserContext, tensor: string): TensorId =
  let name = nim_ident_normalize(tensor)
  if name notin ctx.tensor_lookup:
    let id = ctx.register_tensor(ident(tensor))
    ctx.tensor_lookup[name] = id
  result = ctx.tensor_lookup[name]

proc lookup_index(ctx: var ParserContext, index: string): RegId =
  let name = nim_ident_normalize(index)
  if name notin ctx.indices:
    let iter = ctx.kernel.regs.alloc(Register(name: name))
    ctx.kernel.loops.add(Loop(iter: iter))
    ctx.indices[name] = iter
  result = ctx.indices[name]

proc is_name(node: NimNode): bool =
  result = node.kind == nnkIdent or node.kind == nnkSym

proc is_name(node: NimNode, name: string): bool =
  result = node.is_name and nim_ident_normalize(node.str_val) == nim_ident_normalize(name)

proc parse_expr(node: NimNode,
                instrs: var seq[Instr],
                ctx: var ParserContext,
                is_main_expr: bool = false): RegId

proc parse_dims*(node: NimNode, ctx: var ParserContext, start: int = 1): seq[LinearIndex] =
  for it in start..<node.len:
    var instrs: seq[Instr]
    let res = node[it].parse_expr(instrs, ctx)
    result.add(LinearIndex(
      setup: instrs,
      factors: to_table({res: 1})
    ))

proc parse_tensor_op(node: NimNode, ctx: var ParserContext): TensorOp =
  case node.kind:
    of nnkCurlyExpr, nnkBracketExpr:
      if node[0].is_name:
        result.tensor = ctx.lookup_tensor(node[0].str_val)
      else:
        result.tensor = ctx.register_tensor(node[0])
      result.is_raw = node.kind == nnkCurlyExpr
      result.dims = node.parse_dims(ctx)
    of nnkInfix:
      assert node[0].is_name("*")
      assert node[1].is_name
      result.tensor = ctx.lookup_tensor(node[1].str_val)
      result.is_raw = node[2].kind == nnkCurly
      result.dims = node[2].parse_dims(ctx, 0)
    else:
      raise ParserError(msg: "Cannot parse tensor operation from " & $node.kind)

proc parse_write(node: NimNode, value: RegId, ctx: var ParserContext): (TensorOp, bool) =
  result[0] = node.parse_tensor_op(ctx)
  result[0].data = value
  result[1] = node.kind == nnkInfix

proc is_shape_access(node: NimNode): bool =
  node.kind == nnkBracketExpr and
  node[0].kind == nnkDotExpr and
  node[0][1].is_name("shape") and
  node[0][0].is_name()

proc parse_expr(node: NimNode,
                instrs: var seq[Instr],
                ctx: var ParserContext,
                is_main_expr: bool = false): RegId =
  var instr: Instr
  case node.kind:
    of nnkIntLit..nnkUint64Lit:
      instr = Instr(kind: InstrIndex,
        index_lit: int(node.int_val)
      )
    of nnkFloatLit..nnkFloat128Lit:
      instr = Instr(kind: InstrScalar,
        scalar_lit: float64(node.float_val)
      )
    of nnkSym, nnkIdent:
      let name = nim_ident_normalize(node.str_val)
      case name:
        of "true", "false": 
          instr = Instr(kind: InstrBoolean,
            boolean_lit: name == "true"
          )
        else:
          return ctx.lookup_index(name)
    of nnkCallKinds:
      if not node[0].is_name:
        raise ParserError(msg: "Indirect calls are not supported")
      let name = nim_ident_normalize(node[0].str_val)
      case name:
        of "@":
          assert node[1].is_name()
          result = ctx.kernel.regs.alloc()
          instrs.add(Instr(kind: InstrExtern,
            extern: node[1].str_val,
            res: result
          ))
          return
        else: discard
      var args: seq[RegId] = @[]
      for it in 1..<node.len:
        args.add(node[it].parse_expr(instrs, ctx, is_main_expr=is_main_expr))
      case name:
        of "-":
          case args.len:
            of 1: instr = Instr(kind: InstrNegate, args: args)
            of 2: instr = Instr(kind: InstrSub, args: args)
            else: raise ParserError(msg: name & " expects one or two arguments, but got " & $args.len) 
        of "sq":
          if args.len != 1:
            raise ParserError(msg: name & " expects one arguments, but got " & $args.len)
          instr = Instr(kind: InstrMul, args: @[args[0], args[0]])
        of "min", "max":
          if args.len != 2:
            raise ParserError(msg: name & " expects two arguments, but got " & $args.len)
          let cond = ctx.kernel.regs.alloc()
          instrs.add(Instr(kind: InstrLt, args: args, res: cond))
          if name == "max":
            swap(args[0], args[1])
          instr = Instr(kind: InstrSelect, args: @[cond, args[0], args[1]])
        else:
          let (kind, arg_count) = case name:
            of "+": (InstrAdd, 2)
            of "*": (InstrMul, 2)
            of "/": (InstrDiv, 2)
            of "==": (InstrEq, 2)
            of "<": (InstrLt, 2)
            of "<=": (InstrLe, 2)
            of ">":
              swap(args[0], args[1])
              (InstrLt, 2)
            of ">=":
              swap(args[0], args[1])
              (InstrLe, 2)
            of "select": (InstrSelect, 3)
            of "Scalar": (InstrToScalar, 1)
            of "Index": (InstrToIndex, 1)
            of "sin": (InstrSin, 1)
            of "cos": (InstrCos, 1)
            of "exp": (InstrExp, 1)
            of "pow": (InstrPow, 2)
            of "sqrt": (InstrSqrt, 1)
            of "log": (InstrLog, 2)
            of "ln": (InstrLn, 1)
            of "log2": (InstrLog2, 1)
            of "log10": (InstrLog10, 1)
            of "epoch": (InstrEpoch, 0)
            else: (InstrIndex, -1)
          if arg_count == -1:
            raise ParserError(msg: name & " is not a valid instruction")
          if arg_count != args.len:
            raise ParserError(msg: name & " expects " & $arg_count & " arguments, but got " & $args.len)
          instr = Instr(kind: kind, args: args)
    of nnkCurlyExpr, nnkBracketExpr:
      if node.is_shape_access:
        let tensor = ctx.lookup_tensor(node[0][0].str_val)
        var dim = 0
        if node[1].kind == nnkIntLit:
          dim = int(node[1].int_val)
          if dim < 0:
            raise ParserError(msg: "Negative dimensions are not supported, use ^" & $abs(dim) & " for indexing from the end.")
        elif node[1].kind in nnkCallKinds and
             node[1].len == 2 and
             node[1][0].is_name("^"):
          assert node[1][1].kind == nnkIntLit
          dim = -int(node[1][1].int_val)
        else:
          raise ParserError(msg: node.repr & " is not a valid dimension. Only int literals or backwards indices are allowed as dimensions.")
        instr = Instr(kind: InstrShape, dim: dim, tensor: tensor)
      elif is_main_expr:
        var read = node.parse_tensor_op(ctx)
        read.data = ctx.kernel.regs.alloc()
        ctx.kernel.reads.add(read)
        return read.data
      else:
        raise ParserError(msg: "Nested tensor operations are not allowed")
    of nnkDotExpr:
      assert node[1].is_name()
      case node[1].str_val.nim_ident_normalize:
        of "len":
          assert node[0].is_name()
          instr = Instr(kind: InstrLen,
            tensor: ctx.lookup_tensor(node[0].str_val)
          )
        else: raise ParserError(msg: "Unable to parse expression from " & node.repr)
    of nnkPar:
      return node[0].parse_expr(instrs, ctx, is_main_expr)
    of nnkStmtListExpr:
      for stmt in node:
        result = stmt.parse_expr(instrs, ctx, is_main_expr)
      return
    of nnkLetSection:
      for def in node:
        assert def.len == 3
        if not def[0].is_name:
          raise ParserError(msg: "Cannot assign to " & def[0].repr)
        let
          name = def[0].str_val.nim_ident_normalize()
          value = def[2].parse_expr(instrs, ctx, is_main_expr)
        ctx.indices[name] = value
    else:
      raise ParserError(msg: "Unable to parse expression from " & $node.kind)
  
  result = ctx.kernel.regs.alloc()
  instr.res = result
  instrs.add(instr)

proc parse_attribute(node: NimNode, ctx: var ParserContext) =
  case node.kind:
    of nnkCallKinds:
      assert node[0].is_name()
      case node[0].str_val.nim_ident_normalize:
        of "loop":
          let iter = ctx.lookup_index(node[1].str_val)
          for loop in ctx.kernel.loops.mitems:
            if loop.iter == iter:
              loop.has_bounds = true
              loop.start.factors[node[2].parse_expr(loop.start.setup, ctx, true)] = 1
              loop.stop.factors[node[3].parse_expr(loop.stop.setup, ctx, true)] = 1
              break
        else:
          raise ParserError(msg: "Call to " & node[0].str_val & " is not a valid attribute")
    of nnkPar: node[0].parse_attribute(ctx)
    of nnkTupleConstr:
      for child in node:
        child.parse_attribute(ctx)
    else: raise ParserError(msg: node.repr & " is not a valid attribute")

proc parse_body(node: NimNode, ctx: var ParserContext): RegId =
  if node.kind in nnkCallKinds and node[0].is_name("|"):
    result = node[1].parse_body(ctx)
    node[2].parse_attribute(ctx)
  else:
    result = node.parse_expr(ctx.kernel.expr.instrs, ctx, true)

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

proc register_args(fun: Fun, args: openArray[(Fun, TensorId)]):
    Table[TensorId, TensorId] =
  for (arg, id) in args:
    if arg notin fun.args:
      fun.args[arg] = TensorId(fun.args.len + 1)
      fun.children.add(arg)
    result[id] = fun.args[arg]

proc add_kernel(fun: Fun, kernel: Kernel, args: openArray[(Fun, TensorId)]) =
  let subs = fun.register_args(args)
  kernel.substitute(subs)
  fun.kernels.add(kernel)

proc init_literal_instr(lit: int, res: RegId): Instr =
  result = Instr(kind: InstrIndex, res: res, index_lit: lit)

proc init_literal_instr(lit: float64, res: RegId): Instr =
  result = Instr(kind: InstrScalar, res: res, scalar_lit: lit)

proc init_literal_instr(lit: bool, res: RegId): Instr =
  result = Instr(kind: InstrBoolean, res: res, boolean_lit: lit)

proc new_lit(instr: Instr): NimNode =
  if instr.kind == InstrExtern:
    result = new_call(bind_sym("init_literal_instr"),
      ident(instr.extern), new_lit(instr.res)
    )
  else:
    template add_field(constr: NimNode, name: string, value: NimNode) =
      constr.add(new_tree(nnkExprColonExpr, ident(name), value))
    
    result = new_tree(nnkObjConstr, bind_sym("Instr"))
    result.add_field("kind", new_lit(instr.kind))
    result.add_field("args", new_lit(instr.args))
    result.add_field("res", new_lit(instr.res))
    result.add_field("tensor", new_lit(instr.tensor))
    case instr.kind:
      of InstrIndex: result.add_field("index_lit", new_lit(instr.index_lit))
      of InstrScalar: result.add_field("scalar_lit", new_lit(instr.scalar_lit))
      of InstrBoolean: result.add_field("boolean_lit", new_lit(instr.boolean_lit))
      of InstrShape: result.add_field("dim", new_lit(instr.dim))
      else: discard

proc gen_args(tensors: seq[NimNode]): NimNode =
  result = new_nim_node(nnkBracket)
  for it, node in tensors:
    result.add(new_tree(nnkTupleConstr, [
      node, new_lit(TensorId(it + 1))
    ]))

macro `++=`*(target, value: untyped): untyped =
  var ctx = ParserContext(kernel: Kernel())
  let res = value.parse_body(ctx)
  ctx.kernel.expr.res = res
  let (write, new_var) = target.parse_write(res, ctx)
  ctx.kernel.write = write
  let write_name = ctx.tensors[write.tensor]
  result = new_stmt_list()
  if new_var:
    result.add(new_var_stmt(write_name, new_call(bind_sym("Fun"), new_nil_lit())))
  result.add(new_call(bind_sym("ensure_init"), write_name))
  result.add(new_call(bind_sym("add_kernel"),
    write_name,
    new_lit(ctx.kernel),
    ctx.tensors.gen_args()
  ))

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

macro with_shape*(target, dims: untyped): untyped =
  if dims.kind != nnkBracket:
    raise ParserError(msg: "The second argument to with_shape must be of the format [dim0, dim1, ...]")
  
  var
    ctx = ParserContext(kernel: Kernel())
    shape = ShapeConstraint(kind: ShapeDims)
  for dim, child in dims:
    var instrs: seq[Instr] = @[]
    let res = child.parse_expr(instrs, ctx, true)
    shape.dims.add(LinearIndex(factors: to_table({res: 1}), setup: instrs))
  assert target.is_name()
  shape.dest = ctx.lookup_tensor(target.str_val)
  
  result = new_stmt_list([
    new_call(bind_sym("ensure_init"), target),
    new_call(bind_sym("use_shape"), target, new_lit(shape), ctx.tensors.gen_args())
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
