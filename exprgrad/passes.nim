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

# Compiler passes for exprgrad's compiler

import std/[tables, algorithm, sets, math, rationals, sequtils, strutils]
import ir, irprint

proc infer_types(instrs: seq[Instr], regs: var seq[Register]) =
  for instr in instrs:
    template ret_type(): var Type = regs[instr.res].typ
    template arg_type(index: int): Type = regs[instr.args[index]].typ
    
    case instr.kind:
      of InstrIndex: ret_type = Type(kind: TypeIndex, count: 1)
      of InstrScalar: ret_type = Type(kind: TypeScalar, count: 1)
      of InstrBoolean: ret_type = Type(kind: TypeBoolean, count: 1)
      of InstrAdd, InstrSub, InstrMul, InstrDiv,
         InstrEq, InstrLe, InstrLt:
        let (a, b) = (arg_type(0), arg_type(1))
        if a != b:
          raise TypeError(msg: "Arguments of " & $instr.kind & " must have the same type, but got " & $a & " and " & $b & " instead.")
        case instr.kind:
          of InstrEq, InstrLe, InstrLt:
            ret_type = Type(kind: TypeBoolean, count: a.count)
          else: ret_type = a
      of InstrNegate:
        if arg_type(0).kind notin {TypeScalar, TypeIndex}:
          raise TypeError(msg: "Argument to " & $instr.kind & " must be a Scalar or an Index")
        ret_type = arg_type(0)
      of InstrSelect:
        let (cond, a, b) = (arg_type(0), arg_type(1), arg_type(2))
        if a != b:
          raise TypeError(msg: "The second and the third argument of " & $instr.kind & " must have the same type")
        if cond.kind != TypeBoolean:
          raise TypeError(msg: "The first argument of " & $instr.kind & " must be a Boolean")
        if cond.count != a.count:
          raise TypeError(msg: "All arguments of " & $instr.kind & " must have the same count")
        ret_type = a
      of InstrToScalar:
        if arg_type(0).kind notin {TypeIndex}:
          raise TypeError(msg: "Unable to convert " & $arg_type(0) & " to Scalar")
        ret_type = Type(kind: TypeScalar, count: arg_type(0).count)
      of InstrToIndex:
        if arg_type(0).kind notin {TypeScalar}:
          raise TypeError(msg: "Unable to convert " & $arg_type(0) & " to Index")
        ret_type = Type(kind: TypeIndex, count: arg_type(0).count)
      of InstrSin, InstrCos, InstrExp, InstrPow, InstrSqrt,
         InstrLog, InstrLog10, InstrLog2, InstrLn:
        for it in 0..<instr.args.len:
          if arg_type(it).kind != TypeScalar:
            raise TypeError(msg: "Argument " & $it & " to " & $instr.kind & " is currently of type " & $arg_type(it) & ", but must be of type Scalar.")
        ret_type = arg_type(0)
      of InstrShape, InstrLen, InstrShapeLen:
        ret_type = Type(kind: TypeIndex, count: 1)
      of InstrRead, InstrWrite:
        if instr.tensor == TensorId(0):
          raise TypeError(msg: $instr.kind & " must have a tensor argument")
        if arg_type(0).kind != TypeIndex:
          raise TypeError(msg: "First argument to " & $instr.kind & " must be an Index")
        case instr.kind:
          of InstrRead: ret_type = Type(kind: TypeScalar, count: 1)
          of InstrWrite:
            if arg_type(1).kind != TypeScalar:
              raise TypeError(msg: "Second argument of " & $instr.kind & " must be a Scalar")
          else: discard
      of InstrExtern: raise TypeError(msg: $instr.kind & " is not valid at runtime")
      of InstrEpoch: ret_type = Type(kind: TypeIndex, count: 1)

proc infer_types(expr: Expr, regs: var seq[Register]) =
  expr.instrs.infer_types(regs)

proc infer_types(index: LinearIndex, regs: var seq[Register]) =
  index.setup.infer_types(regs)
  for reg, factor in index.factors:
    if regs[reg].typ.kind != TypeIndex:
      raise TypeError(msg: "LinearIndex factor have the type Index")

proc infer_types(tensor_op: TensorOp, regs: var seq[Register]) =
  for dim in tensor_op.dims:
    dim.infer_types(regs)
  if tensor_op.is_raw and tensor_op.dims.len != 1:
    raise TypeError(msg: "A raw tensor operation must have exactly one index")

proc infer_types*(kernel: Kernel) =
  if kernel.generator.kind == GenNone:
    kernel.setup.infer_types(kernel.regs)
    for loop in kernel.loops:
      loop.start.infer_types(kernel.regs)
      loop.stop.infer_types(kernel.regs)
      kernel.regs[loop.iter].typ = Type(kind: TypeIndex, count: 1)
    for read in kernel.reads:
      read.infer_types(kernel.regs)
      kernel.regs[read.data].typ = Type(kind: TypeScalar, count: 1)
    kernel.expr.infer_types(kernel.regs)
    kernel.write.infer_types(kernel.regs)
    if kernel.write.data != RegId(0) and
       kernel.regs[kernel.write.data].typ.kind != TypeScalar:
      raise TypeError(msg: "Kernel must write a Scalar to the output tensor")

proc infer_types*(program: Program) =
  program.assert_pass("infer_types",
    produces={StageTyped},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.infer_types()

proc fold_setup(index: var LinearIndex, kernel: Kernel) =
  var regs = new_seq[LinearIndex](kernel.regs.len)
  for loop in kernel.loops:
    # TODO: How should we handle registers defined in the setup section of the kernel?
    regs[loop.iter] = LinearIndex(
      factors: to_table({loop.iter: 1})
    )
  
  for instr in index.setup:
    template binary_op(op) =
      regs[instr.res] = op(regs[instr.args[0]], regs[instr.args[1]])
    
    template unary_op(op) =
      regs[instr.res] = op(regs[instr.args[0]])
    
    case instr.kind:
      of InstrIndex: regs[instr.res] = LinearIndex(constant: instr.index_lit)
      of InstrAdd: binary_op(`+`)
      of InstrSub: binary_op(`-`)
      of InstrMul: binary_op(`*`)
      of InstrNegate: unary_op(`-`)
      else:
        regs[instr.res] = LinearIndex(
          factors: to_table({instr.res: 1})
        )
  
  var sum = LinearIndex()
  for reg, factor in index.factors:
    sum = sum + regs[reg] * factor
  
  var used = new_seq[bool](kernel.regs.len)
  for reg, factor in sum.factors:
    used[reg] = true
  
  for it in countdown(index.setup.len - 1, 0):
    let instr = index.setup[it]
    if used[instr.res]:
      sum.setup.add(instr)
      for arg in instr.args:
        used[arg] = true
  
  sum.setup.reverse()
  index = sum

proc fold_linear_indices(kernel: Kernel) =
  for loop in kernel.loops.mitems:
    loop.start.fold_setup(kernel)
    loop.stop.fold_setup(kernel)
  for read in kernel.reads.mitems:
    for dim in read.dims.mitems:
      dim.fold_setup(kernel)
  for dim in kernel.write.dims.mitems:
    dim.fold_setup(kernel)

proc fold_linear_indices*(program: Program) =
  program.assert_pass("fold_linear_indices",
    produces={StageFolded},
    preserves={StageTensors}
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.fold_linear_indices()

proc dead_code_elim(instrs: var seq[Instr], used: var seq[bool]) =
  var it = instrs.len - 1
  while it >= 0:
    let instr = instrs[it]
    let is_instr_used = used[instr.res] or instr.kind in SIDE_EFFECT_INSTRS
    if is_instr_used:
      for arg in instr.args:
        used[arg] = true
    else:
      instrs.delete(it)
    it -= 1

proc dead_code_elim(index: var LinearIndex, used: var seq[bool]) =
  for reg, factor in index.factors:
    used[reg] = true
  index.setup.dead_code_elim(used)

proc dead_code_elim(loops: var seq[Loop], used: var seq[bool]) =
  for it in countdown(loops.len - 1, 0):
    loops[it].start.dead_code_elim(used)
    loops[it].stop.dead_code_elim(used)

proc dead_code_elim(reads: var seq[TensorOp], used: var seq[bool]) =
  var it = 0
  while it < reads.len:
    if not used[reads[it].data]:
      reads.delete(it)
    else:
      for dim in reads[it].dims.mitems:
        dim.dead_code_elim(used)
      it += 1

proc dead_code_elim*(kernel: Kernel) =
  if kernel.generator.kind == GenNone:
    var used = new_seq[bool](kernel.regs.len)
    used[kernel.write.data] = true
    for dim in kernel.write.dims.mitems:
      dim.dead_code_elim(used)
    kernel.expr.instrs.dead_code_elim(used)
    kernel.reads.dead_code_elim(used)
    kernel.loops.dead_code_elim(used)
    kernel.setup.dead_code_elim(used)

proc dead_code_elim*(program: Program) =
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.dead_code_elim()

proc dead_kernel_elim*(program: Program) =
  for name, target in program.targets.mpairs:
    var
      used = new_seq[bool](program.tensors.len)
      it = target.kernels.len - 1
    
    for it, tensor in program.tensors:
      if tensor.kind != TensorResult:
        used[it] = true
    if target.output != TensorId(0):
      used[target.output] = true
    
    while it >= 0:
      let kernel = target.kernels[it]
      if used[kernel.write.tensor]:
        for read in kernel.reads:
          used[read.tensor] = true
      else:
        target.kernels.delete(it)
      it -= 1

proc deduplicate_reads*(kernel: Kernel) =
  var
    unique = init_table[TensorOp, RegId]()
    subs = init_table[RegId, RegId]()
    it = 0
  while it < kernel.reads.len:
    var base_read = kernel.reads[it]
    base_read.data = RegId(0)
    if base_read in unique:
      subs[kernel.reads[it].data] = unique[base_read]
      kernel.reads.delete(it)
    else:
      unique[base_read] = kernel.reads[it].data
      it += 1
  
  kernel.expr.substitute(subs)
  kernel.write.substitute(subs)

proc deduplicate_reads*(program: Program) =
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.deduplicate_reads()

proc derive(instrs: seq[Instr],
            regs: var seq[Register],
            grad_regs: var Table[RegId, RegId]): seq[Instr] =
  for it in countdown(instrs.len - 1, 0):
    let instr = instrs[it]
    if instr.res notin grad_regs:
      continue
    let grad = grad_regs[instr.res]
    var grad_args = new_seq[RegId]()
    case instr.kind:
      of InstrAdd:
        grad_args = @[grad, grad]
      of InstrSub:
        let neg_grad = regs.alloc()
        result.add(Instr(kind: InstrNegate, args: @[grad], res: neg_grad))
        grad_args = @[grad, neg_grad]
      of InstrMul:
        let (grad_a, grad_b) = (regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrMul, args: @[grad, instr.args[1]], res: grad_a))
        result.add(Instr(kind: InstrMul, args: @[grad, instr.args[0]], res: grad_b))
        grad_args = @[grad_a, grad_b]
      of InstrDiv:
        # d/dx (x / y) = 1 / y
        # d/dy (x / y) = d/dy (x * y ^ -1) = -x * y ^ -2
        let
          (grad_a, grad_b) = (regs.alloc(), regs.alloc())
          (neg_x, sq_y, div_grad_sq_y) = (regs.alloc(), regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrDiv, args: @[grad, instr.args[1]], res: grad_a))
        result.add(Instr(kind: InstrMul, args: @[instr.args[1], instr.args[1]], res: sq_y))
        result.add(Instr(kind: InstrDiv, args: @[grad, sq_y], res: div_grad_sq_y))
        result.add(Instr(kind: InstrNegate, args: @[instr.args[0]], res: neg_x))
        result.add(Instr(kind: InstrMul, args: @[neg_x, div_grad_sq_y], res: grad_b))
        grad_args = @[grad_a, grad_b]
      of InstrNegate:
        let neg_grad = regs.alloc()
        result.add(Instr(kind: InstrNegate, args: @[grad], res: neg_grad))
        grad_args = @[neg_grad]
      of InstrLn:
        let grad_x = regs.alloc()
        result.add(Instr(kind: InstrDiv, args: @[grad, instr.args[0]], res: grad_x))
        grad_args = @[grad_x]
      of InstrExp:
        let grad_x = regs.alloc()
        result.add(Instr(kind: InstrMul, args: @[grad, instr.res], res: grad_x))
        grad_args = @[grad_x]
      of InstrSin:
        let (cos, grad_x) = (regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrCos, args: @[instr.args[0]], res: cos))
        result.add(Instr(kind: InstrMul, args: @[cos, grad], res: grad_x))
        grad_args = @[grad_x]
      of InstrCos:
        let (sin, neg_sin, grad_x) = (regs.alloc(), regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrSin, args: @[instr.args[0]], res: sin))
        result.add(Instr(kind: InstrNegate, args: @[sin], res: neg_sin))
        result.add(Instr(kind: InstrMul, args: @[neg_sin, grad], res: grad_x))
        grad_args = @[grad_x]
      of InstrSelect:
        let (grad_a, grad_b, zero) = (regs.alloc(), regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrScalar, res: zero))
        result.add(Instr(kind: InstrSelect, args: @[instr.args[0], grad, zero], res: grad_a))
        result.add(Instr(kind: InstrSelect, args: @[instr.args[0], zero, grad], res: grad_b))
        grad_args = @[RegId(0), grad_a, grad_b]
      of InstrToScalar, InstrToIndex: grad_args = @[RegId(0)]
      else: discard
    
    if grad_args.len != instr.args.len:
      raise GradientError(msg: "Unable to derive " & $instr.kind)
    
    for it, arg in instr.args:
      if grad_args[it] != RegId(0):
        if arg in grad_regs:
          let sum = regs.alloc()
          result.add(Instr(kind: InstrAdd, args: @[grad_regs[arg], grad_args[it]], res: sum))
          grad_regs[arg] = sum
        else:
          grad_regs[arg] = grad_args[it]

proc derive*(kernel: Kernel, grad_tensors: Table[TensorId, TensorId]): seq[Kernel] =
  let base_kernel = kernel.clone()
  var grad_regs = init_table[RegId, RegId]()
  
  block derive_write:
    let write_grad = base_kernel.regs.alloc()
    base_kernel.reads.add(TensorOp(
      is_raw: kernel.write.is_raw,
      data: write_grad,
      dims: kernel.write.dims,
      tensor: grad_tensors[kernel.write.tensor]
    ))
    grad_regs[kernel.write.data] = write_grad
  
  block derive_expr:
    base_kernel.expr.instrs &= kernel.expr.instrs.derive(
      base_kernel.regs, grad_regs
    )
  
  for read in kernel.reads:
    let grad_kernel = base_kernel.clone()
    if read.data in grad_regs:
      grad_kernel.expr.res = grad_regs[read.data]
      grad_kernel.write = TensorOp(
        tensor: grad_tensors[read.tensor],
        is_raw: read.is_raw,
        dims: read.dims,
        data: grad_regs[read.data]
      )
      grad_kernel.dead_code_elim()
      result.add(grad_kernel)

proc copy_shape(target: var Target, dest, src: TensorId) =
  target.shapes.add(ShapeConstraint(kind: ShapeCopy, dest: dest, src: src))

proc generate*(program: Program) =
  program.assert_pass("generate",
    produces={StageGenerated},
    preserves={StageShapes, StageFolded, StageTensors}
  )

  for name, target in program.targets.mpairs:
    var it = 0
    while it < target.kernels.len:
      let kernel = target.kernels[it]
      case kernel.generator.kind:
        of GenBackwards:
          var
            grad_tensors = init_table[TensorId, TensorId]()
            grad_kernels: seq[Kernel] = @[]
          
          block:
            let
              loss = kernel.generator.tensor
              grad_loss = program.tensors.alloc(TensorDef(
                kind: TensorResult
              ))
            grad_kernels.add(Kernel(
              regs: @[
                Register(typ: Type(kind: TypeScalar, count: 1)),
                Register(typ: Type(kind: TypeIndex, count: 1)),
                Register(typ: Type(kind: TypeIndex, count: 1))
              ],
              loops: @[Loop(iter: RegId(2),
                has_bounds: true,
                stop: LinearIndex(
                  setup: @[Instr(kind: InstrLen, tensor: loss, res: RegId(3))],
                  factors: to_table({RegId(3): 1})
                )
              )],
              expr: Expr(
                instrs: @[Instr(kind: InstrScalar, scalar_lit: 1, res: RegId(1))],
                res: RegId(1)
              ),
              write: TensorOp(
                is_raw: true,
                tensor: grad_loss,
                dims: @[LinearIndex(factors: to_table({RegId(2): 1}))],
                data: RegId(1)
              )
            ))
            target.copy_shape(grad_loss, loss)
            grad_tensors[loss] = grad_loss
          
          for it2 in (it + 1)..<target.kernels.len:
            let kernel = target.kernels[it2]
            if kernel.generator.kind == GenGradient:
              grad_tensors[kernel.generator.tensor] = kernel.write.tensor
              target.copy_shape(kernel.write.tensor, kernel.generator.tensor)
          
          for it2 in countdown(it - 1, 0):
            let kernel = target.kernels[it2]
            for read in kernel.reads:
              if read.tensor notin grad_tensors:
                let grad_tensor = program.tensors.alloc(TensorDef(
                  kind: TensorResult
                ))
                target.copy_shape(grad_tensor, read.tensor)
                grad_tensors[read.tensor] = grad_tensor
            grad_kernels.add(kernel.derive(grad_tensors))
          
          target.kernels.delete(it)
          target.kernels.insert(grad_kernels, it)
          it += grad_kernels.len
        of GenGradient:
          target.kernels.delete(it)
        of GenReshape:
          target.kernels[it] = Kernel(
            regs: @[
              Register(typ: Type(kind: TypeScalar, count: 1)),
              Register(typ: Type(kind: TypeIndex, count: 1)),
              Register(typ: Type(kind: TypeIndex, count: 1))
            ],
            loops: @[Loop(iter: RegId(2),
              has_bounds: true,
              stop: LinearIndex(
                setup: @[Instr(kind: InstrLen,
                  tensor: kernel.generator.tensor, res: RegId(3)
                )],
                factors: to_table({RegId(3): 1})
              )
            )],
            reads: @[TensorOp(
              tensor: kernel.generator.tensor,
              dims: @[LinearIndex(factors: to_table({RegId(2): 1}))],
              data: RegId(1),
              is_raw: true
            )],
            expr: Expr(res: RegId(1)),
            write: TensorOp(
              tensor: kernel.write.tensor,
              dims: @[LinearIndex(factors: to_table({RegId(2): 1}))],
              data: RegId(1),
              is_raw: true
            )
          )
          var
            shape = ShapeConstraint(kind: ShapeDims,
              dest: kernel.write.tensor
            )
            prod = 1
          for dim, size in kernel.generator.reshape:
            if size >= 0:
              prod *= size
          for dim, size in kernel.generator.reshape:
            if size >= 0:
              shape.dims.add(LinearIndex(constant: size))
            else:
              shape.dims.add(LinearIndex(
                setup: @[
                  Instr(kind: InstrLen, tensor: kernel.generator.tensor, res: RegId(1)),
                  Instr(kind: InstrIndex, index_lit: prod, res: RegId(2)),
                  Instr(kind: InstrDiv, args: @[RegId(1), RegId(2)], res: RegId(3))
                ],
                factors: to_table({RegId(3): 1})
              ))
          target.shapes.add(shape)
          it += 1
        of GenNone:
          it += 1

proc reorder_loops*(kernel: Kernel) =
  var loop_iters = new_seq[LoopId](kernel.regs.len)
  for it, loop in kernel.loops:
    loop_iters[loop.iter] = LoopId(it + 1)
  
  var graph = new_seq[array[TensorOpKind, seq[LoopId]]](kernel.loops.len)
  for kind, op in kernel.tensor_ops:
    for it in 1..<op.dims.len:
      for reg_a, factor_a in op.dims[it - 1].factors:
        for reg_b, factor_b in op.dims[it].factors:
          if loop_iters[reg_a] != LoopId(0) and
             loop_iters[reg_b] != LoopId(0):
            graph[int(loop_iters[reg_b]) - 1][kind].add(loop_iters[reg_a])
  
  const SCORE_VALS = [OpRead: 10, OpWrite: 1]
  var scores = new_seq[int](kernel.loops.len)
  for it, edges in graph:
    for kind, kind_edges in edges:
      for target in kind_edges:
        scores[target] += SCORE_VALS[kind]
  
  var
    closed = new_seq[bool](kernel.loops.len)
    order = new_seq[LoopId]()
  for it in 0..<kernel.loops.len:
    var
      min_score = 0
      min_loop = LoopId(0)
    for it, score in scores:
      if not closed[it]:
        if score < min_score or min_loop == LoopId(0):
          min_loop = LoopId(it + 1)
          min_score = score
    
    assert min_loop != LoopId(0)
    closed[min_loop] = true
    order.add(min_loop)
    
    for kind, edges in graph[min_loop]:
      for target in edges:
        scores[target] -= SCORE_VALS[kind]
  
  var new_loops = new_seq[Loop](order.len)
  for it, loop_id in order:
    new_loops[it] = kernel.loops[loop_id]
  kernel.loops = new_loops

proc reorder_loops*(program: Program) =
  program.assert_pass("reorder_loops",
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.reorder_loops()

proc unfold(linear: LinearIndex, regs: var seq[Register]): Expr =
  result.instrs = linear.setup
  
  var terms = new_seq[RegId]()
  for reg, factor in linear.factors:
    if factor != 0:
      if factor == 1:
        terms.add(reg)
      else:
        let (product, factor_reg) = (regs.alloc(), regs.alloc())
        result.instrs.add(Instr(kind: InstrIndex, index_lit: factor, res: factor_reg))
        result.instrs.add(Instr(kind: InstrMul, args: @[reg, factor_reg], res: product))
        terms.add(product)
  
  if linear.constant != 0:
    let reg = regs.alloc()
    result.instrs.add(Instr(kind: InstrIndex, index_lit: linear.constant, res: reg))
    terms.add(reg)
  
  if terms.len > 0:
    var sum = terms[0]
    for it in 1..<terms.len:
      let res = regs.alloc()
      result.instrs.add(Instr(kind: InstrAdd, args: @[sum, terms[it]], res: res))
      sum = res
    result.res = sum
  else:
    let zero = regs.alloc()
    result.instrs.add(Instr(kind: InstrIndex, res: zero))
    result.res = zero

proc expand_tensor_index(dims: seq[LinearIndex],
                         tensor: TensorId,
                         regs: var seq[Register]): Expr =
  var
    stride = RegId(0)
    terms = new_seq[RegId]()
  for it, dim in dims:
    let dim_expr = dim.unfold(regs)
    result.instrs.add(dim_expr.instrs)
    
    if stride == RegId(0):
      terms.add(dim_expr.res)
    else:
      let product = regs.alloc()
      result.instrs.add(Instr(kind: InstrMul,
        args: @[dim_expr.res, stride],
        res: product
      ))
      terms.add(product)
    
    if it != dims.len - 1:
      let size = regs.alloc()
      result.instrs.add(Instr(kind: InstrShape,
        tensor: tensor, dim: it, res: size
      ))
      if stride == RegId(0):
        stride = size
      else:
        let new_stride = regs.alloc()
        result.instrs.add(Instr(kind: InstrMul,
          args: @[size, stride],
          res: new_stride
        ))
        stride = new_stride
  
  if terms.len == 0:
    let zero = regs.alloc()
    result.instrs.add(Instr(kind: InstrIndex, res: zero))
    result.res = zero
  else:
    var sum = terms[0]
    for it in 1..<terms.len:
      let new_sum = regs.alloc()
      result.instrs.add(Instr(kind: InstrAdd,
        args: @[sum, terms[it]],
        res: new_sum
      ))
      sum = new_sum
    result.res = sum

proc inline_tensor_ops(kernel: Kernel) =
  const ACCESS_INSTR = [OpRead: InstrRead, OpWrite: InstrWrite]
  var instrs = [
    OpRead: new_seq[Instr](),
    OpWrite: new_seq[Instr]()
  ]
  
  for kind, tensor_op in kernel.tensor_ops:
    var args = new_seq[RegId]()
    if tensor_op.is_raw:
      let dim = tensor_op.dims[0].unfold(kernel.regs)
      instrs[kind].add(dim.instrs)
      args.add(dim.res)
    else:
      let index = tensor_op.dims.expand_tensor_index(tensor_op.tensor, kernel.regs)
      instrs[kind].add(index.instrs)
      args.add(index.res)
    
    var res = RegId(0)
    case kind:
      of OpRead: res = tensor_op.data
      of OpWrite: args.add(tensor_op.data)
    
    instrs[kind].add(Instr(kind: ACCESS_INSTR[kind],
      tensor: tensor_op.tensor,
      args: args,
      res: res
    ))
  
  kernel.expr.instrs = instrs[OpRead] & kernel.expr.instrs & instrs[OpWrite]
  kernel.expr.res = RegId(0)
  kernel.reads = new_seq[TensorOp]()
  kernel.write = TensorOp()

proc inline_tensor_ops*(program: Program) =
  program.assert_pass("inline_tensor_ops",
    requires={StageFolded},
    produces={StageTensorInstrs},
    preserves={
      StageFolded, StageTensors, StageGenerated, StageBounds,
      StageTensorInstrs, StageShapes, StageSortedShapes,
      StageStaticShapes
    }
  )

  for name, target in program.targets.mpairs:
    for kernel in target.kernels:
      kernel.inline_tensor_ops()

proc collect_tensors(instrs: seq[Instr], tensors: var HashSet[TensorId]) =
  for instr in instrs:
    if instr.tensor != TensorId(0):
      tensors.incl(instr.tensor)

proc collect_tensors(kernel: Kernel, tensors: var HashSet[TensorId]) =
  for kind, op in kernel.tensor_ops:
    tensors.incl(op.tensor)
  for loop in kernel.loops:
    loop.start.setup.collect_tensors(tensors)
    loop.stop.setup.collect_tensors(tensors)
  kernel.expr.instrs.collect_tensors(tensors)

proc collect_tensors*(program: Program) =
  program.assert_pass("collect_tensors",
    requires={},
    produces={StageCollected},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets.mpairs:
    target.tensors = init_hash_set[TensorId]()
    for kernel in target.kernels:
      kernel.collect_tensors(target.tensors)

proc unfold_inplace(index: var LinearIndex, regs: var seq[Register]) =
  let expr = index.unfold(regs)
  index.setup = expr.instrs
  index.factors = to_table({expr.res: 1})

proc unfold_loop_bounds*(program: Program) =
  program.assert_pass("unfold_loop_bounds",
    requires={StageFolded},
    preserves={
      StageTensors, StageGenerated, StageBounds,
      StageTensorInstrs, StageShapes, StageSortedShapes
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      for loop in kernel.loops.mitems:
        loop.start.unfold_inplace(kernel.regs)
        loop.stop.unfold_inplace(kernel.regs)

proc peek_key[K, V](tab: Table[K, V]): K =
  for key, value in tab:
    return key

proc peek_value[K, V](tab: Table[K, V]): V =
  for key, value in tab:
    return value

proc only_register*(linear: LinearIndex): RegId =
  if linear.constant == 0 and
     linear.factors.len == 1 and
     linear.factors.peek_value() == 1:
    result = linear.factors.peek_key()

proc use_bounds(loop: var Loop, op: TensorOp, dim: int, regs: var seq[Register]) =
  loop.has_bounds = true
  loop.start = LinearIndex(constant: 0)
  let size = regs.alloc()
  loop.stop = LinearIndex(factors: to_table({size: 1}))
  if op.is_raw:
    loop.stop.setup = @[Instr(kind: InstrLen,
      tensor: op.tensor, res: size
    )]
  else:
    loop.stop.setup = @[Instr(kind: InstrShape,
      tensor: op.tensor, dim: dim, res: size
    )]

proc infer_loop_bounds*(program: Program) =
  program.assert_pass("infer_loop_bounds",
    requires={StageFolded},
    produces={StageBounds},
    preserves={
      StageFolded, StageShapes, StageSortedShapes,
      StageTensors, StageGenerated, StageStaticShapes
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      var iters = init_table[RegId, LoopId]()
      for it, loop in kernel.loops:
        if not loop.has_bounds:
          iters[loop.iter] = LoopId(it + 1)
      for kind, op in kernel.tensor_ops:
        for it, dim in op.dims:
          if dim.only_register != RegId(0) and
             dim.only_register in iters:
            let loop_id = iters[dim.only_register]
            if not kernel.loops[loop_id].has_bounds:
              kernel.loops[loop_id].use_bounds(op, it, kernel.regs)

proc simplify_max_index(indices: var seq[LinearIndex]) =
  var
    max_constants = init_table[Table[RegId, int], int]()
    complex_indices = new_seq[LinearIndex]()
  for it, index in indices:
    if index.setup.len == 0:
      if index.factors notin max_constants:
        max_constants[index.factors] = index.constant
      else:
        max_constants[index.factors] = max(max_constants[index.factors], index.constant)
    else:
      complex_indices.add(index)
  
  indices = complex_indices
  for factors, constant in max_constants:
    indices.add(LinearIndex(
      factors: factors, constant: constant
    ))

proc infer_shape_constraints(kernel: Kernel): ShapeConstraint =
  if kernel.write.is_raw:
    if kernel.reads.len == 1:
      result = ShapeConstraint(kind: ShapeCopy,
        src: kernel.reads[0].tensor,
        dest: kernel.write.tensor
      )
  else:
    result = ShapeConstraint(kind: ShapeLinear)
    for op in kernel.reads:
      if not op.is_raw:
        if op.tensor notin result.reads:
          result.reads[op.tensor] = new_seq[seq[LinearIndex]](op.dims.len)
        for it, dim in op.dims:
          result.reads[op.tensor][it].add(dim)
    
    result.dest = kernel.write.tensor
    for it, dim in kernel.write.dims:
      result.write.add(dim)
    
    for tensor, dims in result.reads.mpairs:
      for dim in dims.mitems:
        dim.simplify_max_index()

proc infer_shape_constraints*(program: Program) =
  program.assert_pass("infer_shape_constraints",
    requires={StageFolded, StageTensors},
    produces={StageShapes},
    preserves={
      StageGenerated, StageFolded, StageTyped, StageTensors
    }
  )
  
  for name, target in program.targets.mpairs:
    for tensor in program.caches:
      let tensor_def = program.tensors[tensor]
      target.shapes.add(ShapeConstraint(kind: ShapeCopy,
        src: tensor_def.cache, dest: tensor
      ))
    
    for it, kernel in target.kernels:
      if kernel.generator.kind == GenNone:
        target.shapes.add(kernel.infer_shape_constraints())

iterator deps(shape: ShapeConstraint): TensorId =
  case shape.kind:
    of ShapeNone: discard
    of ShapeDims:
      for dim in shape.dims:
        for instr in dim.setup:
          if instr.tensor != TensorId(0):
            yield instr.tensor
    of ShapeLinear:
      for tensor in shape.reads.keys:
        yield tensor
    of ShapeCopy: yield shape.src

proc flatten_constraints(tensor: TensorId,
                         tensors: Table[TensorId, ShapeConstraint],
                         closed: var seq[bool],
                         order: var seq[ShapeConstraint],
                         program: Program) =
  if program.tensors[tensor].kind in {TensorResult, TensorCache, TensorRandom} and
     not closed[tensor]:
    closed[tensor] = true
    if tensor notin tensors:
      raise ShapeError(msg: $tensor & " (" & program.tensors[tensor].name & ") requires shape")
    let constr = tensors[tensor]
    for dep in constr.deps:
      dep.flatten_constraints(tensors, closed, order, program)
    order.add(constr)

proc sort_shape_constraints*(program: Program) =
  program.assert_pass("sort_shape_constraints",
    requires={StageShapes, StageCollected},
    produces={StageSortedShapes},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets.mpairs:
    var
      tensors = init_table[TensorId, ShapeConstraint]()
      closed = new_seq[bool](program.tensors.len)
    
    for constr in target.shapes:
      if constr.dest notin tensors:
        tensors[constr.dest] = constr
      else:
        discard # TODO: Unify current and new constraint
    
    var order = new_seq[ShapeConstraint]()
    for tensor in target.tensors:
      tensor.flatten_constraints(tensors, closed, order, program)
    
    target.shapes = order

type Matrix[T] = object
  data: seq[T]
  width: int

{.push inline.}
proc height(matrix: Matrix): int = matrix.data.len div matrix.width
proc `[]`[T](matrix: Matrix[T], x, y: int): T = matrix.data[x + y * matrix.width]
proc `[]`[T](matrix: var Matrix[T], x, y: int): var T = matrix.data[x + y * matrix.width]
proc `[]=`[T](matrix: var Matrix[T], x, y: int, value: T) = matrix.data[x + y * matrix.width] = value

proc `$`[T](matrix: Matrix[T]): string =
  for y in 0..<matrix.height:
    if y != 0:
      result &= "; "
    for x in 0..<matrix.width:
      if x != 0:
        result &= ", "
      result &= $matrix[x, y]
  result = "[" & result & "]"

proc swap_rows[T](matrix: var Matrix[T], a, b: int) =
  for x in 0..<matrix.width:
    swap(matrix[x, a], matrix[x, b])
{.pop.}

proc init_matrix[T](w, h: int): Matrix[T] =
  result = Matrix[T](width: w, data: new_seq[T](w * h))

type Fraction = Rational[int]
proc solve(equations: seq[LinearIndex]): Table[RegId, Fraction] =
  var indices = init_table[RegId, int]()
  for equation in equations:
    for reg, factor in equation.factors:
      if reg notin indices:
        indices[reg] = indices.len
  
  if indices.len == 0:
    return
  
  if equations.len < indices.len:
    raise new_exception(ValueError, "Underconstrained linear system")
  
  var
    matrix = init_matrix[int](indices.len + 1, indices.len)
    known = init_hash_set[seq[Fraction]]()
    y = 0
  for equation in equations:
    if equation.factors.len == 0:
      if equation.constant != 0:
        raise new_exception(ValueError, "No solution")
      continue
    
    var row = new_seq[int](matrix.width)
    for reg, factor in equation.factors:
      row[indices[reg]] = factor
    row[indices.len] = -equation.constant
    var
      normalized = new_seq[Fraction](matrix.width)
      first_value = 0
    for x, value in row:
      if first_value == 0:
        first_value = value
      if first_value == 0:
        normalized[x] = 0//1
      else:
        normalized[x] = value // first_value
    
    if normalized notin known:
      for x in 0..<matrix.width:
        matrix[x, y] = row[x]
      known.incl(normalized)
      y += 1
      if y >= matrix.height:
        break
  
  if y < matrix.height:
    raise new_exception(ValueError, "Underconstrained linear system")
  
  for pivot in 0..<matrix.height:
    var max_row = pivot
    for y in (pivot + 1)..<matrix.height:
      if abs(matrix[pivot, y]) > abs(matrix[pivot, max_row]):
        max_row = y
    if max_row != pivot:
      matrix.swap_rows(max_row, pivot)
    let target = matrix[pivot, pivot]
    for y in (pivot + 1)..<matrix.height:
      let cur = matrix[pivot, y]
      if cur != 0:
        for x in 0..<matrix.width:
          matrix[x, y] = matrix[x, y] * target - matrix[x, pivot] * cur
  
  var solutions = new_seq[Fraction](indices.len)
  for y in countdown(matrix.height - 1, 0):
    var sum = matrix[indices.len, y] // 1
    for x in (y + 1)..<indices.len:
      sum -= solutions[x] * matrix[x, y]
    solutions[y] = sum / (matrix[y, y] // 1)
  
  for reg, index in indices:
    result[reg] = solutions[index]

proc eval*(instrs: seq[Instr],
           shapes: Table[TensorId, seq[int]],
           regs: var Table[RegId, int]): bool =
  for instr in instrs:
    var can_eval = true
    for arg in instr.args:
      if arg notin regs:
        can_eval = false
        break
    if can_eval and instr.tensor != TensorId(0):
      can_eval = instr.tensor in shapes
    if can_eval:
      case instr.kind:
        of InstrShape:
          var size = -1
          let shape = shapes[instr.tensor]
          if instr.dim < 0:
            size = shape[shape.len + instr.dim]
          else:
            size = shape[instr.dim]
          regs[instr.res] = size
          if size < 0:
            result = true
        of InstrLen: regs[instr.res] = shapes[instr.tensor].prod()
        of InstrShapeLen: regs[instr.res] = shapes[instr.tensor].len
        of InstrIndex: regs[instr.res] = instr.index_lit
        of InstrAdd: regs[instr.res] = regs[instr.args[0]] + regs[instr.args[1]]
        of InstrSub: regs[instr.res] = regs[instr.args[0]] - regs[instr.args[1]]
        of InstrMul: regs[instr.res] = regs[instr.args[0]] * regs[instr.args[1]]
        of InstrDiv: regs[instr.res] = regs[instr.args[0]] div regs[instr.args[1]]
        of InstrNegate: regs[instr.res] = -regs[instr.args[0]]
        else: raise ShapeError(msg: $instr.kind & " is not allowed in tensor shape definition")
    else:
      result = true

proc matches(static_shape, shape: seq[int]): bool =
  if static_shape.len == 0:
    result = true
  elif static_shape.len == shape.len:
    for dim, size in static_shape:
      if size >= 0:
        if shape[dim] != static_shape[dim]:
          return false
    result = true

proc infer_shapes*(program: Program,
                   target: string,
                   inputs: openArray[(TensorId, seq[int])]): Table[TensorId, seq[int]] =
  result = init_table[TensorId, seq[int]]()
  for (tensor, shape) in inputs:
    result[tensor] = shape
    let static_shape = program.tensors[tensor].shape
    if not static_shape.matches(shape):
      raise ShapeError(msg: "Given shape for " & $tensor & " is " & $shape & ", but its static shape is " & $static_shape)
  for tensor_id in program.params:
    result[tensor_id] = program.tensors[tensor_id].shape
  for shape in program.targets[target].shapes:
    case shape.kind:
      of ShapeNone: discard
      of ShapeDims:
        var sizes = new_seq[int](shape.dims.len)
        for dim, index in shape.dims:
          var regs = init_table[RegId, int]()
          if index.setup.eval(result, regs):
            raise ShapeError(msg: "Unable to evaluate all instructions")
          sizes[dim] = index.eval(regs)
        result[shape.dest] = sizes
      of ShapeCopy:
        result[shape.dest] = result[shape.src]
      of ShapeLinear:
        var equations: seq[LinearIndex] = @[]
        for tensor, dims in shape.reads:
          for dim, indices in dims:
            assert indices.len == 1
            let index = indices[0]
            equations.add(index - (result[tensor][dim] - 1))
        
        var max_values = init_table[RegId, int]()
        for reg, max_value in solve(equations):
          max_values[reg] = max_value.num div max_value.den
        
        result[shape.dest] = new_seq[int](shape.write.len)
        for dim, index in shape.write:
          result[shape.dest][dim] = index.eval(max_values) + 1

proc infer_static_shapes*(program: Program) =
  program.assert_pass("infer_static_shapes",
    requires={StageSortedShapes},
    produces={StageStaticShapes},
    preserves=ALL_STAGES
  )
  
  var shapes = init_table[TensorId, seq[int]]()
  for it, tensor in program.tensors:
    let id = TensorId(it + 1)
    if tensor.shape.len > 0:
      shapes[id] = tensor.shape
  
  for name, target in program.targets:
    for shape in target.shapes:
      var dims: seq[int] = @[]
      case shape.kind:
        of ShapeNone: discard
        of ShapeDims:
          dims = new_seq[int](shape.dims.len)
          for dim, size in shape.dims:
            var regs = init_table[RegId, int]()
            if size.setup.eval(shapes, regs):
              dims[dim] = -1
            else:
              dims[dim] = size.eval(regs)
        of ShapeLinear:
          var equations: seq[LinearIndex] = @[]
          for tensor, dims in shape.reads:
            if tensor in shapes and shapes[tensor].len == dims.len:
              for dim, index in dims:
                assert index.len == 1
                let size = shapes[tensor][dim]
                if size >= 0:
                  equations.add(index[0] - (size - 1))
          
          var max_values = init_table[RegId, int]()
          for reg, max_value in solve(equations):
            max_values[reg] = max_value.num div max_value.den
          
          dims = new_seq[int](shape.write.len)
          for dim, size in shape.write:
            var can_eval = true
            for reg, factor in size.factors:
              if reg notin max_values:
                can_eval = false
                break
            if can_eval:
              dims[dim] = size.eval(max_values) + 1
            else:
              dims[dim] = -1
        of ShapeCopy:
          if shape.src in shapes:
            dims = shapes[shape.src]
      
      if dims.len > 0:
        if shape.dest in shapes:
          do_assert shapes[shape.dest] == dims
        else:
          shapes[shape.dest] = dims
  
  for it, tensor in program.tensors.mpairs:
    let id = TensorId(it + 1)
    case tensor.kind:
      of TensorResult, TensorRandom:
        if id in shapes:
          tensor.shape = shapes[id]
      of TensorCache:
        if id notin shapes or shapes[id].any_it(it < 0):
          let kind = ($tensor.kind)[len("Tensor")..^1].to_lower_ascii()
          raise ShapeError(msg: "Shape of " & kind & " \"" & tensor.name & "\" must be inferred at compile time")
        tensor.shape = shapes[id]
      else:
        if id in shapes:
          assert tensor.shape == shapes[id]

proc inline_static_shapes(instrs: var seq[Instr], tensors: seq[TensorDef]) =
  for instr in instrs.mitems:
    var size = -1
    if instr.tensor != TensorId(0) and
       tensors[instr.tensor].shape.len > 0:
      let shape = tensors[instr.tensor].shape
      case instr.kind:
        of InstrLen: size = shape.prod()
        of InstrShape:
          if instr.dim < 0:
            size = shape[shape.len + instr.dim]
          else:
            size = shape[instr.dim]
        of InstrShapeLen: size = shape.len
        else: discard
    if size >= 0:
      instr = Instr(kind: InstrIndex, index_lit: size, res: instr.res)

proc inline_static_shapes*(program: Program) =
  program.assert_pass("inline_static_shapes",
    produces={},
    requires={StageStaticShapes, StageBounds, StageTensorInstrs},
    preserves={
      StageTensors, StageFolded, StageShapes, StageSortedShapes,
      StageGenerated, StageTyped, StageBounds, StageTensorInstrs
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.setup.inline_static_shapes(program.tensors)
      for loop in kernel.loops.mitems:
        loop.start.setup.inline_static_shapes(program.tensors)
        loop.stop.setup.inline_static_shapes(program.tensors)
      kernel.expr.instrs.inline_static_shapes(program.tensors)

proc make_tensor_lookups*(program: Program) =
  program.assert_pass("make_tensor_lookups",
    produces={StageTensors},
    preserves=ALL_STAGES
  )
  
  for it, tensor in program.tensors:
    let id = TensorId(it + 1)
    case tensor.kind:
      of TensorParam: program.params.add(id)
      of TensorInput: program.inputs[tensor.name] = id
      of TensorCache: program.caches.add(id)
      else: discard

proc lift_shape_instrs(kernel: Kernel) =
  var it = 0
  while it < kernel.expr.instrs.len:
    let instr = kernel.expr.instrs[it]
    case instr.kind:
      of InstrShape, InstrLen, InstrShapeLen, InstrEpoch:
        kernel.setup.add(instr)
        kernel.expr.instrs.delete(it)
      else:
        it += 1

proc lift_shape_instrs*(program: Program) =
  program.assert_pass("lift_shape_instrs",
    produces={},
    requires={StageTensorInstrs},
    preserves={
      StageTyped, StageFolded, StageGenerated,
      StageTensors, StageShapes, StageSortedShapes, StageBounds,
      StageTensorInstrs
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.lift_shape_instrs()

proc identify_independent*(kernel: Kernel) =
  var independent = init_hash_set[RegId]()
  for dim in kernel.write.dims:
    if dim.only_register != RegId(0):
      independent.incl(dim.factors.peek_key())
  for loop in kernel.loops.mitems:
    if loop.iter in independent:
      loop.mode = LoopIndependent

proc identify_independent*(program: Program) =
  program.assert_pass("identify_independent",
    produces={StageIndependent},
    requires={},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.identify_independent()

proc is_reduce(kernel: Kernel): bool =
  for loop in kernel.loops:
    if loop.mode <= LoopNone:
      return true

proc can_fuse(a, b: Kernel): bool =
  if not a.is_reduce():
    for read in b.reads:
      if read.tensor == a.write.tensor:
        if read.is_raw:
          return false
    return true

proc is_map(kernel: Kernel): bool =
  result = not kernel.is_reduce() and kernel.reads.len == 1

proc should_fuse(a, b: Kernel): bool =
  result = can_fuse(a, b) and b.is_map()

proc fuse_loops(a, b: Kernel): Kernel =
  assert can_fuse(a, b)
  result = b.clone()
  result.reads = @[]
  for read in b.reads:
    if read.tensor == a.write.tensor:
      assert not read.is_raw
      var subs = init_table[RegId, RegId]()
      for dim, index in read.dims:
        subs[a.write.dims[dim].only_register] = index.only_register
      subs[a.write.data] = read.data
      for it, register in a.regs:
        let reg = RegId(it + 1)
        if reg notin subs:
          subs[reg] = result.regs.alloc(register)
      var
        instrs = a.expr.instrs
        setup = a.setup
      instrs.substitute(subs)
      setup.substitute(subs)
      result.expr.instrs = instrs & result.expr.instrs
      result.setup = setup & result.setup
      for op in a.reads:
        var new_read = op
        new_read.substitute(subs)
        result.reads.add(new_read)
      result.deduplicate_reads()
    else:
      result.reads.add(read)

proc fuse_loops*(program: Program) =
  program.assert_pass("fuse_loops",
    produces={},
    requires={StageIndependent},
    preserves={
      StageShapes, StageSortedShapes, StageFolded, StageIndependent,
      StageBounds, StageTensors, StageGenerated
    }
  )
  
  for name, target in program.targets.mpairs:
    var usage_counts = new_seq[int](program.tensors.len)
    for kernel in target.kernels:
      for read in kernel.reads:
        usage_counts[read.tensor] += 1
  
    var
      producers = new_seq[KernelId](program.tensors.len)
      multiple = init_hash_set[TensorId]()
    
    for it, initial_kernel in target.kernels:
      var kernel = initial_kernel
      for read in initial_kernel.reads:
        if usage_counts[read.tensor] == 1 and
           producers[read.tensor] != KernelId(0):
          let producer = target.kernels[producers[read.tensor]]
          if should_fuse(producer, kernel):
            echo "Fuse!"
            kernel = fuse_loops(producer, kernel)
      
      let id = KernelId(it + 1)
      if producers[kernel.write.tensor] != KernelId(0) or
         kernel.write.tensor in multiple:
        producers[kernel.write.tensor] = KernelId(0)
        multiple.incl(kernel.write.tensor)
      else:
        producers[kernel.write.tensor] = id
      
      target.kernels[it] = kernel
  
