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

import std/[tables, algorithm, sets, math, rationals, sequtils, strutils, sugar]
import ir, irprint

proc inferTypes(instrs: seq[Instr], regs: var seq[Register]) =
  for instr in instrs:
    template retType(): var Type = regs[instr.res].typ
    template argType(index: int): Type = regs[instr.args[index]].typ
    
    case instr.kind:
      of InstrIndex: retType = Type(kind: TypeIndex, count: 1)
      of InstrScalar: retType = Type(kind: TypeScalar, count: 1)
      of InstrBoolean: retType = Type(kind: TypeBoolean, count: 1)
      of InstrAdd, InstrSub, InstrMul,
         InstrEq, InstrLe, InstrLt:
        let (a, b) = (argType(0), argType(1))
        if a != b:
          raise TypeError(msg: "Arguments of " & $instr.kind & " must have the same type, but got " & $a & " and " & $b & " instead.")
        case instr.kind:
          of InstrEq, InstrLe, InstrLt:
            retType = Type(kind: TypeBoolean, count: a.count)
          else: retType = a
      of InstrDiv:
        if argType(0).kind != TypeScalar or argType(0).kind != TypeScalar:
          raise TypeError(msg: "Arguments of " & $instr.kind & " must be of type Scalar.")
        retType = argType(0)
      of InstrIndexDiv, InstrMod, InstrWrap:
        if argType(0).kind != TypeIndex or argType(0).kind != TypeIndex:
          raise TypeError(msg: "Arguments of " & $instr.kind & " must be of type Index.")
        retType = argType(0)
      of InstrNegate:
        if argType(0).kind notin {TypeScalar, TypeIndex}:
          raise TypeError(msg: "Argument to " & $instr.kind & " must be a Scalar or an Index")
        retType = argType(0)
      of InstrAnd, InstrOr:
        if argType(0).kind != TypeBoolean or argType(0).kind != TypeBoolean:
          raise TypeError(msg: "Arguments of " & $instr.kind & " must be of type Boolean.")
        retType = argType(0)
      of InstrSelect:
        let (cond, a, b) = (argType(0), argType(1), argType(2))
        if a != b:
          raise TypeError(msg: "The second and the third argument of " & $instr.kind & " must have the same type")
        if cond.kind != TypeBoolean:
          raise TypeError(msg: "The first argument of " & $instr.kind & " must be a Boolean")
        if cond.count != a.count:
          raise TypeError(msg: "All arguments of " & $instr.kind & " must have the same count")
        retType = a
      of InstrToScalar:
        if argType(0).kind notin {TypeIndex}:
          raise TypeError(msg: "Unable to convert " & $argType(0) & " to Scalar")
        retType = Type(kind: TypeScalar, count: argType(0).count)
      of InstrToIndex:
        if argType(0).kind notin {TypeScalar}:
          raise TypeError(msg: "Unable to convert " & $argType(0) & " to Index")
        retType = Type(kind: TypeIndex, count: argType(0).count)
      of InstrSin, InstrCos, InstrExp, InstrPow, InstrSqrt,
         InstrLog, InstrLog10, InstrLog2, InstrLn:
        for it in 0..<instr.args.len:
          if argType(it).kind != TypeScalar:
            raise TypeError(msg: "Argument " & $it & " to " & $instr.kind & " is currently of type " & $argType(it) & ", but must be of type Scalar.")
        retType = argType(0)
      of InstrShape, InstrLen, InstrShapeLen:
        retType = Type(kind: TypeIndex, count: 1)
      of InstrArray:
        for it in 1..<instr.args.len:
          if argType(it) != argType(0):
            raise TypeError(msg: "All items in array must be of the same type")
        retType = Type(kind: TypeArray,
          count: 1,
          len: instr.args.len,
          item: argType(0)
        )
      of InstrArrayLen:
        if argType(0).kind != TypeArray:
          raise TypeError(msg: "Argument to " & $instr.kind & " must be an array")
        retType = Type(kind: TypeIndex, count: argType(0).count)
      of InstrArrayRead:
        if argType(0).kind != TypeArray:
          raise TypeError(msg: "First argument to " & $instr.kind & " must be an array")
        if argType(1).kind != TypeIndex:
          raise TypeError(msg: "Second argument to " & $instr.kind & " must be an index")
        if argType(0).count != argType(1).count:
          raise TypeError() 
        retType = argType(0).item
      of InstrRead, InstrWrite, InstrOverwrite:
        if instr.tensor == TensorId(0):
          raise TypeError(msg: $instr.kind & " must have a tensor argument")
        if argType(0).kind != TypeIndex:
          raise TypeError(msg: "First argument to " & $instr.kind & " must be an Index")
        case instr.kind:
          of InstrRead: retType = Type(kind: TypeScalar, count: 1)
          of InstrWrite:
            if argType(1).kind != TypeScalar:
              raise TypeError(msg: "Second argument of " & $instr.kind & " must be a Scalar")
          else: discard
      of InstrEpoch: retType = Type(kind: TypeIndex, count: 1)
      of InstrLoop:
        if argType(0).kind != TypeIndex or argType(1).kind != TypeIndex:
          raise TypeError(msg: "Loop bounds must be of type Index, but are currently of types " & $argType(0) & " and " & $argType(1))
        regs[instr.loopIter].typ = Type(kind: TypeIndex, count: 1)
        instr.body.inferTypes(regs)
      of InstrThreads:
        if argType(0).kind != TypeIndex or argType(1).kind != TypeIndex:
          raise TypeError(msg: "Thread range must be of type Index")
        regs[instr.threadsBegin].typ = Type(kind: TypeIndex, count: 1)
        regs[instr.threadsEnd].typ = Type(kind: TypeIndex, count: 1)
        instr.body.inferTypes(regs)
      of InstrGpu:
        for it, arg in instr.args:
          if argType(it).kind != TypeIndex:
            raise TypeError(msg: "Gpu ranges must be of type Index")
        for index in instr.gpuIndices:
          regs[index.group].typ = Type(kind: TypeIndex, count: 1)
          regs[index.local].typ = Type(kind: TypeIndex, count: 1)
        instr.body.inferTypes(regs)
      of InstrIf:
        if argType(0).kind != TypeBoolean:
          raise TypeError(msg: "If condition must be of type Boolean")
        instr.body.inferTypes(regs)
      of InstrSharedCache:
        retType = Type(kind: TypeArray,
          count: 1,
          len: instr.cacheSize,
          item: Type(kind: TypeScalar, count: 1)
        )
      of InstrCacheWrite:
        if argType(0).kind != TypeArray:
          raise TypeError(msg: "Local cache must be of type Array")
        if argType(1).kind != TypeIndex:
          raise TypeError(msg: "Index into local cache must be of type Index")
        if argType(2).kind != TypeScalar:
          raise TypeError(msg: "Third argument of " & $instr.kind & " must be of type Scalar")
      of InstrBarrier: discard

proc inferTypes(expr: Expr, regs: var seq[Register]) =
  expr.instrs.inferTypes(regs)

proc inferTypes(index: LinearIndex, regs: var seq[Register]) =
  index.setup.inferTypes(regs)
  for reg, factor in index.factors:
    if regs[reg].typ.kind != TypeIndex:
      raise TypeError(msg: "LinearIndex factor have the type Index")

proc inferTypes(tensorOp: TensorOp, regs: var seq[Register]) =
  for dim in tensorOp.dims:
    dim.inferTypes(regs)
  if tensorOp.isRaw and tensorOp.dims.len != 1:
    raise TypeError(msg: "A raw tensor operation must have exactly one index")

proc inferTypes*(kernel: Kernel) =
  if kernel.generator.kind == GenNone:
    kernel.setup.inferTypes(kernel.regs)
    for loop in kernel.loops:
      loop.start.inferTypes(kernel.regs)
      loop.stop.inferTypes(kernel.regs)
      kernel.regs[loop.iter].typ = Type(kind: TypeIndex, count: 1)
    for cond in kernel.conds:
      cond.inferTypes(kernel.regs)
      if kernel.regs[cond.res].typ.kind != TypeBoolean:
        raise TypeError(msg: "Condition must be of type Boolean")
    for read in kernel.reads:
      read.inferTypes(kernel.regs)
      kernel.regs[read.data].typ = Type(kind: TypeScalar, count: 1)
    kernel.expr.inferTypes(kernel.regs)
    kernel.write.inferTypes(kernel.regs)
    if kernel.write.data != RegId(0) and
       kernel.regs[kernel.write.data].typ.kind != TypeScalar:
      raise TypeError(msg: "Kernel must write a Scalar to the output tensor")

proc inferTypes*(program: Program) =
  program.assertPass("inferTypes",
    produces={StageTyped},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.inferTypes()

proc foldSetup(index: var LinearIndex, kernel: Kernel) =
  var regs = newSeq[LinearIndex](kernel.regs.len)
  for loop in kernel.loops:
    # TODO: How should we handle registers defined in the setup section of the kernel?
    regs[loop.iter] = LinearIndex(
      factors: toTable({loop.iter: 1})
    )
  
  for instr in index.setup:
    template binaryOp(op) =
      regs[instr.res] = op(regs[instr.args[0]], regs[instr.args[1]])
    
    template unaryOp(op) =
      regs[instr.res] = op(regs[instr.args[0]])
    
    case instr.kind:
      of InstrIndex: regs[instr.res] = initLinearIndex(instr.indexLit)
      of InstrAdd: binaryOp(`+`)
      of InstrSub: binaryOp(`-`)
      of InstrMul:
        try:
          binaryOp(`*`)
        except ValueError: # TODO: Do not use exception?
          regs[instr.res] = LinearIndex(
            factors: toTable({instr.res: 1})
          )
      of InstrNegate: unaryOp(`-`)
      else:
        regs[instr.res] = LinearIndex(
          factors: toTable({instr.res: 1})
        )
  
  var sum = LinearIndex()
  for reg, factor in index.factors:
    sum = sum + regs[reg] * factor
  
  var used = newSeq[bool](kernel.regs.len)
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

proc foldLinearIndices(kernel: Kernel) =
  for loop in kernel.loops.mitems:
    loop.start.foldSetup(kernel)
    loop.stop.foldSetup(kernel)
  for read in kernel.reads.mitems:
    for dim in read.dims.mitems:
      dim.foldSetup(kernel)
  for dim in kernel.write.dims.mitems:
    dim.foldSetup(kernel)

proc foldLinearIndices*(program: Program) =
  program.assertPass("foldLinearIndices",
    produces={StageFolded},
    preserves={StageTensors}
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.foldLinearIndices()
      if kernel.grad.isCustom:
        for gradKernel in kernel.grad.kernels:
          gradKernel.foldLinearIndices()

proc deadCodeElim(instrs: var seq[Instr], used: var seq[bool]) =
  var it = instrs.len - 1
  while it >= 0:
    let instr = instrs[it]
    let isInstrUsed = used[instr.res] or instr.kind in SIDE_EFFECT_INSTRS
    if isInstrUsed:
      for arg in instr.args:
        used[arg] = true
    else:
      instrs.delete(it)
    it -= 1

proc deadCodeElim(index: var LinearIndex, used: var seq[bool]) =
  for reg, factor in index.factors:
    used[reg] = true
  index.setup.deadCodeElim(used)

proc deadCodeElim(loops: var seq[Loop], used: var seq[bool]) =
  for it in countdown(loops.len - 1, 0):
    loops[it].start.deadCodeElim(used)
    loops[it].stop.deadCodeElim(used)

proc deadCodeElim(reads: var seq[TensorOp], used: var seq[bool]) =
  var it = 0
  while it < reads.len:
    if not used[reads[it].data]:
      reads.delete(it)
    else:
      for dim in reads[it].dims.mitems:
        dim.deadCodeElim(used)
      it += 1

proc deadCodeElim*(kernel: Kernel) =
  if kernel.generator.kind == GenNone:
    var used = newSeq[bool](kernel.regs.len)
    used[kernel.write.data] = true
    for dim in kernel.write.dims.mitems:
      dim.deadCodeElim(used)
    kernel.expr.instrs.deadCodeElim(used)
    kernel.reads.deadCodeElim(used)
    kernel.loops.deadCodeElim(used)
    kernel.setup.deadCodeElim(used)

proc deadCodeElim*(program: Program) =
  program.assertPass("deadCodeElim",
    produces={},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.deadCodeElim()
      if kernel.grad.isCustom:
        for gradKernel in kernel.grad.kernels:
          gradKernel.deadCodeElim()

proc deadKernelElim*(program: Program) =
  for name, target in program.targets.mpairs:
    var
      used = newSeq[bool](program.tensors.len)
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

proc deduplicateReads*(kernel: Kernel) =
  var
    unique = initTable[TensorOp, RegId]()
    subs = initTable[RegId, RegId]()
    it = 0
  while it < kernel.reads.len:
    var baseRead = kernel.reads[it]
    baseRead.data = RegId(0)
    if baseRead in unique:
      subs[kernel.reads[it].data] = unique[baseRead]
      kernel.reads.delete(it)
    else:
      unique[baseRead] = kernel.reads[it].data
      it += 1
  
  kernel.expr.substitute(subs)
  kernel.write.substitute(subs)

proc deduplicateReads*(program: Program) =
  program.assertPass("deduplicateReads",
    produces={},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.deduplicateReads()
      if kernel.grad.isCustom:
        for gradKernel in kernel.grad.kernels:
          gradKernel.deduplicateReads()

proc derive(instrs: seq[Instr],
            regs: var seq[Register],
            gradRegs: var Table[RegId, RegId]): seq[Instr] =
  for it in countdown(instrs.len - 1, 0):
    let instr = instrs[it]
    if instr.res notin gradRegs:
      continue
    let grad = gradRegs[instr.res]
    var gradArgs = newSeq[RegId]()
    case instr.kind:
      of InstrAdd:
        gradArgs = @[grad, grad]
      of InstrSub:
        let negGrad = regs.alloc()
        result.add(Instr(kind: InstrNegate, args: @[grad], res: negGrad))
        gradArgs = @[grad, negGrad]
      of InstrMul:
        let (gradA, gradB) = (regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrMul, args: @[grad, instr.args[1]], res: gradA))
        result.add(Instr(kind: InstrMul, args: @[grad, instr.args[0]], res: gradB))
        gradArgs = @[gradA, gradB]
      of InstrDiv:
        # d/dx (x / y) = 1 / y
        # d/dy (x / y) = d/dy (x * y ^ -1) = -x * y ^ -2
        let
          (gradA, gradB) = (regs.alloc(), regs.alloc())
          (negX, sqY, divGradSqY) = (regs.alloc(), regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrDiv, args: @[grad, instr.args[1]], res: gradA))
        result.add(Instr(kind: InstrMul, args: @[instr.args[1], instr.args[1]], res: sqY))
        result.add(Instr(kind: InstrDiv, args: @[grad, sqY], res: divGradSqY))
        result.add(Instr(kind: InstrNegate, args: @[instr.args[0]], res: negX))
        result.add(Instr(kind: InstrMul, args: @[negX, divGradSqY], res: gradB))
        gradArgs = @[gradA, gradB]
      of InstrNegate:
        let negGrad = regs.alloc()
        result.add(Instr(kind: InstrNegate, args: @[grad], res: negGrad))
        gradArgs = @[negGrad]
      of InstrLn:
        let gradX = regs.alloc()
        result.add(Instr(kind: InstrDiv, args: @[grad, instr.args[0]], res: gradX))
        gradArgs = @[gradX]
      of InstrExp:
        let gradX = regs.alloc()
        result.add(Instr(kind: InstrMul, args: @[grad, instr.res], res: gradX))
        gradArgs = @[gradX]
      of InstrSin:
        let (cos, gradX) = (regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrCos, args: @[instr.args[0]], res: cos))
        result.add(Instr(kind: InstrMul, args: @[cos, grad], res: gradX))
        gradArgs = @[gradX]
      of InstrCos:
        let (sin, negSin, gradX) = (regs.alloc(), regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrSin, args: @[instr.args[0]], res: sin))
        result.add(Instr(kind: InstrNegate, args: @[sin], res: negSin))
        result.add(Instr(kind: InstrMul, args: @[negSin, grad], res: gradX))
        gradArgs = @[gradX]
      of InstrSelect:
        let (gradA, gradB, zero) = (regs.alloc(), regs.alloc(), regs.alloc())
        result.add(Instr(kind: InstrScalar, res: zero))
        result.add(Instr(kind: InstrSelect, args: @[instr.args[0], grad, zero], res: gradA))
        result.add(Instr(kind: InstrSelect, args: @[instr.args[0], zero, grad], res: gradB))
        gradArgs = @[RegId(0), gradA, gradB]
      of InstrSqrt:
        let
          (two, denom) = (regs.alloc(), regs.alloc())
          gradX = regs.alloc()
        result.add(Instr(kind: InstrScalar, scalarLit: 2.0, res: two))
        result.add(Instr(kind: InstrMul, args: @[two, instr.res], res: denom))
        result.add(Instr(kind: InstrDiv, args: @[grad, denom], res: gradX))
        gradArgs = @[gradX]
      of InstrToScalar, InstrToIndex: gradArgs = @[RegId(0)]
      else: discard
    
    if gradArgs.len != instr.args.len:
      raise GradientError(msg: "Unable to derive " & $instr.kind)
    
    for it, arg in instr.args:
      if gradArgs[it] != RegId(0):
        if arg in gradRegs:
          let sum = regs.alloc()
          result.add(Instr(kind: InstrAdd, args: @[gradRegs[arg], gradArgs[it]], res: sum))
          gradRegs[arg] = sum
        else:
          gradRegs[arg] = gradArgs[it]

proc derive*(kernel: Kernel, gradTensors: Table[TensorId, TensorId]): seq[Kernel] =
  let baseKernel = kernel.clone()
  var gradRegs = initTable[RegId, RegId]()
  
  block deriveWrite:
    let writeGrad = baseKernel.regs.alloc()
    baseKernel.reads.add(TensorOp(
      isRaw: kernel.write.isRaw,
      data: writeGrad,
      dims: kernel.write.dims,
      tensor: gradTensors[kernel.write.tensor]
    ))
    gradRegs[kernel.write.data] = writeGrad
  
  block deriveExpr:
    baseKernel.expr.instrs &= kernel.expr.instrs.derive(
      baseKernel.regs, gradRegs
    )
  
  for read in kernel.reads:
    let gradKernel = baseKernel.clone()
    if read.data in gradRegs:
      gradKernel.expr.res = gradRegs[read.data]
      gradKernel.write = TensorOp(
        tensor: gradTensors[read.tensor],
        isRaw: read.isRaw,
        dims: read.dims,
        data: gradRegs[read.data]
      )
      gradKernel.deadCodeElim()
      result.add(gradKernel)

proc copyShape(target: Target, dest, src: TensorId) =
  target.shapes.add(ShapeConstraint(kind: ShapeCopy,
    priority: PriorityInferred,
    dest: dest,
    src: src
  ))

proc generate*(program: Program) =
  program.assertPass("generate",
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
            gradTensors = initTable[TensorId, TensorId]()
            gradKernels: seq[Kernel] = @[]
          
          block:
            let
              loss = kernel.generator.tensor
              gradLoss = program.tensors.alloc(TensorDef(
                kind: TensorResult
              ))
            gradKernels.add(Kernel(
              regs: @[
                Register(typ: Type(kind: TypeScalar, count: 1)),
                Register(typ: Type(kind: TypeIndex, count: 1)),
                Register(typ: Type(kind: TypeIndex, count: 1))
              ],
              loops: @[Loop(iter: RegId(2),
                hasBounds: true,
                stop: LinearIndex(
                  setup: @[Instr(kind: InstrLen, tensor: loss, res: RegId(3))],
                  factors: toTable({RegId(3): 1})
                ),
                step: 1
              )],
              expr: Expr(
                instrs: @[Instr(kind: InstrScalar, scalarLit: 1, res: RegId(1))],
                res: RegId(1)
              ),
              write: TensorOp(
                isRaw: true,
                tensor: gradLoss,
                dims: @[initLinearIndex(RegId(2))],
                data: RegId(1)
              )
            ))
            target.copyShape(gradLoss, loss)
            gradTensors[loss] = gradLoss
          
          for it2 in (it + 1)..<target.kernels.len:
            let kernel = target.kernels[it2]
            if kernel.generator.kind == GenGradient:
              gradTensors[kernel.generator.tensor] = kernel.write.tensor
              target.copyShape(kernel.write.tensor, kernel.generator.tensor)
          
          for it2 in countdown(it - 1, 0):
            let kernel = target.kernels[it2]
            for read in kernel.reads:
              if read.tensor notin gradTensors:
                let gradTensor = program.tensors.alloc(TensorDef(
                  kind: TensorResult
                ))
                target.copyShape(gradTensor, read.tensor)
                gradTensors[read.tensor] = gradTensor
            
            if kernel.grad.isCustom:
              var subs = kernel.grad.subs
              for initialTensor, grad in kernel.grad.tensors:
                var tensor = initialTensor
                if tensor in kernel.grad.subs:
                  tensor = kernel.grad.subs[tensor]
                subs[grad] = gradTensors[tensor]
              for it in countdown(kernel.grad.kernels.len - 1, 0):
                var gradKernel = kernel.grad.kernels[it].clone()
                gradKernel.substitute(subs)
                gradKernels.add(gradKernel)
            else:
              gradKernels.add(kernel.derive(gradTensors))
          
          target.kernels.delete(it)
          target.kernels.insert(gradKernels, it)
          it += gradKernels.len
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
              hasBounds: true,
              stop: LinearIndex(
                setup: @[Instr(kind: InstrLen,
                  tensor: kernel.generator.tensor, res: RegId(3)
                )],
                factors: toTable({RegId(3): 1})
              ),
              step: 1
            )],
            reads: @[TensorOp(
              tensor: kernel.generator.tensor,
              dims: @[initLinearIndex(RegId(2))],
              data: RegId(1),
              isRaw: true
            )],
            expr: Expr(res: RegId(1)),
            write: TensorOp(
              tensor: kernel.write.tensor,
              dims: @[initLinearIndex(RegId(2))],
              data: RegId(1),
              isRaw: true
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
              shape.dims.add(initLinearIndex(size))
            else:
              shape.dims.add(LinearIndex(
                setup: @[
                  Instr(kind: InstrLen, tensor: kernel.generator.tensor, res: RegId(1)),
                  Instr(kind: InstrIndex, indexLit: prod, res: RegId(2)),
                  Instr(kind: InstrIndexDiv, args: @[RegId(1), RegId(2)], res: RegId(3))
                ],
                factors: toTable({RegId(3): 1})
              ))
          target.shapes.add(shape)
          it += 1
        of GenNone:
          it += 1

proc reorderLoops*(kernel: Kernel) =
  var loopIters = newSeq[LoopId](kernel.regs.len)
  for it, loop in kernel.loops:
    loopIters[loop.iter] = LoopId(it + 1)
  
  var graph = newSeq[array[TensorOpKind, seq[LoopId]]](kernel.loops.len)
  for kind, op in kernel.tensorOps:
    for it in 1..<op.dims.len:
      for regA, factorA in op.dims[it - 1].factors:
        for regB, factorB in op.dims[it].factors:
          if loopIters[regA] != LoopId(0) and
             loopIters[regB] != LoopId(0):
            graph[int(loopIters[regA]) - 1][kind].add(loopIters[regB])
  
  const SCORE_VALS = [OpRead: 10, OpWrite: 1]
  var scores = newSeq[int](kernel.loops.len)
  for it, edges in graph:
    for kind, kindEdges in edges:
      for target in kindEdges:
        scores[target] += SCORE_VALS[kind]
  
  var
    closed = newSeq[bool](kernel.loops.len)
    order = newSeq[LoopId]()
  for it in 0..<kernel.loops.len:
    var
      minScore = 0
      minLoop = LoopId(0)
    for it, score in scores:
      if not closed[it]:
        if score < minScore or minLoop == LoopId(0):
          minLoop = LoopId(it + 1)
          minScore = score
    
    assert minLoop != LoopId(0)
    closed[minLoop] = true
    order.add(minLoop)
    
    for kind, edges in graph[minLoop]:
      for target in edges:
        scores[target] -= SCORE_VALS[kind]
  
  var newLoops = newSeq[Loop](order.len)
  for it, loopId in order:
    newLoops[it] = kernel.loops[loopId]
  kernel.loops = newLoops

proc reorderLoops*(program: Program) =
  program.assertPass("reorderLoops",
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.reorderLoops()

proc unfold(linear: LinearIndex, regs: var seq[Register]): Expr =
  result.instrs = linear.setup
  
  var terms = newSeq[RegId]()
  for reg, factor in linear.factors:
    if factor != 0:
      if factor == 1:
        terms.add(reg)
      else:
        let (product, factorReg) = (regs.alloc(), regs.alloc())
        result.instrs.add(Instr(kind: InstrIndex, indexLit: factor, res: factorReg))
        result.instrs.add(Instr(kind: InstrMul, args: @[reg, factorReg], res: product))
        terms.add(product)
  
  if linear.constant != 0:
    let reg = regs.alloc()
    result.instrs.add(Instr(kind: InstrIndex, indexLit: linear.constant, res: reg))
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

proc expandTensorIndex(dims: seq[LinearIndex],
                         tensor: TensorId,
                         regs: var seq[Register],
                         shape: openArray[int] = []): Expr =
  var
    stride = RegId(0)
    terms = newSeq[RegId]()
  for it in countdown(dims.len - 1, 0):
    let
      dim = dims[it]
      dimExpr = dim.unfold(regs)
    result.instrs.add(dimExpr.instrs)
    
    if stride == RegId(0):
      terms.add(dimExpr.res)
    else:
      let product = regs.alloc()
      result.instrs.add(Instr(kind: InstrMul,
        args: @[dimExpr.res, stride],
        res: product
      ))
      terms.add(product)
    
    if it != 0:
      let size = regs.alloc()
      if it < shape.len and shape[it] >= 0:
        result.instrs.add(Instr(kind: InstrIndex,
          indexLit: shape[it], res: size
        ))
      else:
        result.instrs.add(Instr(kind: InstrShape,
          tensor: tensor, dim: it, res: size
        ))
      if stride == RegId(0):
        stride = size
      else:
        let newStride = regs.alloc()
        result.instrs.add(Instr(kind: InstrMul,
          args: @[size, stride],
          res: newStride
        ))
        stride = newStride
  
  if terms.len == 0:
    let zero = regs.alloc()
    result.instrs.add(Instr(kind: InstrIndex, res: zero))
    result.res = zero
  else:
    var sum = terms[0]
    for it in 1..<terms.len:
      let newSum = regs.alloc()
      result.instrs.add(Instr(kind: InstrAdd,
        args: @[sum, terms[it]],
        res: newSum
      ))
      sum = newSum
    result.res = sum

proc inlineTensorOps(kernel: Kernel, hasWritten: var seq[bool]) =
  var instrs = [
    OpRead: newSeq[Instr](),
    OpWrite: newSeq[Instr]()
  ]
  
  for kind, tensorOp in kernel.tensorOps:
    var args = newSeq[RegId]()
    if tensorOp.cache.exists:
      args.add(tensorOp.cache.reg)
    
    if tensorOp.isRaw:
      let dim = tensorOp.dims[0].unfold(kernel.regs)
      instrs[kind].add(dim.instrs)
      args.add(dim.res)
    else:
      let index =
        if tensorOp.cache.exists:
          var
            dims: seq[LinearIndex] = @[]
            cacheShape: seq[int] = @[]
          for it, dim in tensorOp.dims:
            let cacheDim = tensorOp.cache.dims[it]
            dims.add(dim - cacheDim.offset - initLinearIndex(cacheDim.interval.min))
            cacheShape.add(cacheDim.interval.max - cacheDim.interval.min + 1)
          expandTensorIndex(dims, tensorOp.tensor, kernel.regs, cacheShape)
        else:
          expandTensorIndex(tensorOp.dims, tensorOp.tensor, kernel.regs)
      
      instrs[kind].add(index.instrs)
      args.add(index.res)
    
    var res = RegId(0)
    case kind:
      of OpRead: res = tensorOp.data
      of OpWrite: args.add(tensorOp.data)
    
    let instrKind = case kind:
      of OpRead:
        if tensorOp.cache.exists:
          InstrArrayRead
        else:
          InstrRead
      of OpWrite:
        var canOverwrite = not hasWritten[tensorOp.tensor]
        for loop in kernel.loops:
          if loop.mode < LoopIndependent:
            canOverwrite = false
            break
        if canOverwrite:
          InstrOverwrite
        else:
          InstrWrite
    
    let tensor =
      if instrKind == InstrArrayRead:
        TensorId(0)
      else:
        tensorOp.tensor
    
    instrs[kind].add(Instr(kind: instrKind,
      tensor: tensor,
      args: args,
      res: res
    ))
  
  hasWritten[kernel.write.tensor] = true
  kernel.expr.instrs = instrs[OpRead] & kernel.expr.instrs & instrs[OpWrite]
  kernel.expr.res = RegId(0)
  kernel.reads = newSeq[TensorOp]()
  kernel.write = TensorOp()

proc inlineTensorOps*(program: Program) =
  program.assertPass("inlineTensorOps",
    requires={StageFolded, StageCacheSizes},
    produces={StageTensorInstrs},
    preserves={
      StageFolded, StageTensors, StageGenerated, StageBounds,
      StageTensorInstrs, StageShapes, StageSortedShapes,
      StageStaticShapes
    }
  )

  var hasWritten = newSeq[bool](program.tensors.len)
  for it, tensor in program.tensors:
    if tensor.kind != TensorResult:
      hasWritten[it] = true
  for name, target in program.targets.mpairs:
    for kernel in target.kernels:
      kernel.inlineTensorOps(hasWritten)

proc collectTensors(instrs: seq[Instr], tensors: var HashSet[TensorId]) =
  for instr in instrs:
    if instr.tensor != TensorId(0):
      tensors.incl(instr.tensor)
    instr.body.collectTensors(tensors)

proc collectTensors(instrs: seq[Instr]): HashSet[TensorId] =
  instrs.collectTensors(result)

proc collectTensors(kernel: Kernel, tensors: var HashSet[TensorId]) =
  for kind, op in kernel.tensorOps:
    tensors.incl(op.tensor)
  for loop in kernel.loops:
    loop.start.setup.collectTensors(tensors)
    loop.stop.setup.collectTensors(tensors)
  kernel.expr.instrs.collectTensors(tensors)

proc collectTensors*(program: Program) =
  program.assertPass("collectTensors",
    requires={},
    produces={StageCollected},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets.mpairs:
    target.tensors = initHashSet[TensorId]()
    for kernel in target.kernels:
      kernel.collectTensors(target.tensors)

proc unfoldInplace(index: var LinearIndex, regs: var seq[Register]) =
  let expr = index.unfold(regs)
  index.setup = expr.instrs
  index.factors = toTable({expr.res: 1})
  index.constant = 0

proc unfoldLoopBounds*(program: Program) =
  program.assertPass("unfoldLoopBounds",
    requires={StageFolded},
    preserves={
      StageTensors, StageGenerated, StageBounds,
      StageTensorInstrs, StageShapes, StageSortedShapes
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      for loop in kernel.loops.mitems:
        loop.start.unfoldInplace(kernel.regs)
        loop.stop.unfoldInplace(kernel.regs)

proc peekKey[K, V](tab: Table[K, V]): K =
  for key, value in tab:
    return key

proc peekValue[K, V](tab: Table[K, V]): V =
  for key, value in tab:
    return value

proc onlyRegister*(linear: LinearIndex): RegId =
  if linear.constant == 0 and
     linear.factors.len == 1 and
     linear.factors.peekValue() == 1:
    result = linear.factors.peekKey()

proc useBounds(loop: var Loop, op: TensorOp, dim: int, regs: var seq[Register]) =
  loop.hasBounds = true
  loop.start = initLinearIndex(0)
  let size = regs.alloc()
  loop.stop = initLinearIndex(size)
  if op.isRaw:
    loop.stop.setup = @[Instr(kind: InstrLen,
      tensor: op.tensor, res: size
    )]
  else:
    loop.stop.setup = @[Instr(kind: InstrShape,
      tensor: op.tensor, dim: dim, res: size
    )]
  loop.step = 1

proc inferLoopBounds*(program: Program) =
  program.assertPass("inferLoopBounds",
    requires={StageFolded},
    produces={StageBounds},
    preserves={
      StageFolded, StageShapes, StageSortedShapes,
      StageTensors, StageGenerated, StageStaticShapes
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      var iters = initTable[RegId, LoopId]()
      for it, loop in kernel.loops:
        if not loop.hasBounds:
          iters[loop.iter] = LoopId(it + 1)
      for kind, op in kernel.tensorOps:
        for it, dim in op.dims:
          if dim.onlyRegister != RegId(0) and
             dim.onlyRegister in iters:
            let loopId = iters[dim.onlyRegister]
            if not kernel.loops[loopId].hasBounds:
              kernel.loops[loopId].useBounds(op, it, kernel.regs)

proc simplifyMaxIndex(indices: var seq[LinearIndex]) =
  var
    maxConstants = initTable[Table[RegId, int], int]()
    complexIndices = newSeq[LinearIndex]()
  for it, index in indices:
    if index.setup.len == 0:
      if index.factors notin maxConstants:
        maxConstants[index.factors] = index.constant
      else:
        maxConstants[index.factors] = max(maxConstants[index.factors], index.constant)
    else:
      complexIndices.add(index)
  
  indices = complexIndices
  for factors, constant in maxConstants:
    indices.add(LinearIndex(
      factors: factors, constant: constant
    ))

proc inferShapeConstraints(kernel: Kernel): seq[ShapeConstraint] =
  if kernel.write.isRaw:
    if kernel.reads.len == 1:
      result.add(ShapeConstraint(kind: ShapeCopy,
        priority: PriorityInferred,
        src: kernel.reads[0].tensor,
        dest: kernel.write.tensor
      ))
  else:
    var linear = ShapeConstraint(
      kind: ShapeLinear,
      priority: PriorityInferred
    )
    
    for op in kernel.reads:
      if not op.isRaw:
        if op.tensor notin linear.reads:
          linear.reads[op.tensor] = newSeq[seq[LinearIndex]](op.dims.len)
        for it, dim in op.dims:
          linear.reads[op.tensor][it].add(dim)
    
    linear.dest = kernel.write.tensor
    for it, dim in kernel.write.dims:
      linear.write.add(dim)
    
    for tensor, dims in linear.reads.mpairs:
      for dim in dims.mitems:
        dim.simplifyMaxIndex()
    result.add(linear)
  
  for kind, op in kernel.tensorOps:
    if not op.isRaw:
      result.add(ShapeConstraint(kind: ShapeRank,
        dest: op.tensor,
        priority: PriorityCondition,
        rank: op.dims.len
      ))

proc inferShapeConstraints*(program: Program) =
  program.assertPass("inferShapeConstraints",
    requires={StageFolded, StageTensors},
    produces={StageShapes},
    preserves={
      StageGenerated, StageFolded, StageTyped, StageTensors
    }
  )
  
  for name, target in program.targets.mpairs:
    for tensor in program.caches:
      let tensorDef = program.tensors[tensor]
      target.shapes.add(ShapeConstraint(kind: ShapeCopy,
        priority: PriorityInferred,
        src: tensorDef.cache,
        dest: tensor
      ))
    
    for it, kernel in target.kernels:
      if kernel.generator.kind == GenNone:
        target.shapes.add(kernel.inferShapeConstraints())

proc isUnderconstrained(shape: ShapeConstraint): bool =
  case shape.kind:
    of ShapeNone: result = true
    of ShapeRank: result = shape.rank > 0
    of ShapeDims, ShapeCopy: result = false
    of ShapeLinear:
      result = false
      # TODO: Linear system
      var defined = initHashSet[RegId]()
      for tensor, dims in shape.reads:
        for dim, indices in dims:
          assert indices.len == 1
          let index = indices[0]
          for reg, factor in index.factors:
            defined.incl(reg)
      for dim in shape.write:
        for reg, factor in dim.factors:
          if reg notin defined:
            return true

iterator deps(shape: ShapeConstraint): TensorId =
  case shape.kind:
    of ShapeNone, ShapeRank: discard
    of ShapeDims:
      for dim in shape.dims:
        for instr in dim.setup:
          if instr.tensor != TensorId(0):
            yield instr.tensor
    of ShapeLinear:
      for tensor in shape.reads.keys:
        yield tensor
    of ShapeCopy: yield shape.src

proc flattenConstraints(tensor: TensorId,
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
    if constr.isUnderconstrained():
      raise ShapeError(msg: "Shape for " & $tensor & " is underconstrained")
    for dep in constr.deps:
      dep.flattenConstraints(tensors, closed, order, program)
    order.add(constr)

proc sortShapeConstraints*(program: Program) =
  program.assertPass("sortShapeConstraints",
    requires={StageShapes, StageCollected},
    produces={StageSortedShapes},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets.mpairs:
    var
      tensors = initTable[TensorId, ShapeConstraint]()
      closed = newSeq[bool](program.tensors.len)
    
    var conditions = newSeq[ShapeConstraint]()
    for constr in target.shapes:
      if constr.dest notin tensors or
         tensors[constr.dest].priority < constr.priority:
        tensors[constr.dest] = constr
      
      if constr.priority == PriorityCondition:
        conditions.add(constr)
    
    for cond in conditions:
      assert cond.kind == ShapeRank # TODO
      if cond.dest in tensors:
        var constr = tensors[cond.dest]
        while constr.kind == ShapeCopy and
              constr.src in tensors and
              program.tensors[constr.dest].shape.len == 0:
          constr = tensors[constr.src]
        
        if constr.kind == ShapeCopy and
           program.tensors[constr.dest].shape.len == 0:
          tensors[constr.src] = cond
        else:
          let rank = block:
            if program.tensors[constr.dest].shape.len > 0:
              program.tensors[constr.dest].shape.len
            else:
              assert constr.kind in {ShapeDims, ShapeLinear, ShapeRank}
              case constr.kind:
                of ShapeDims: constr.dims.len
                of ShapeLinear: constr.write.len
                of ShapeRank: constr.rank
                else: -1
          
          if cond.rank != rank:
            raise ShapeError(msg: "A condition requires that " & $cond.dest & " has rank " & $cond.rank & ", but it has rank " & $rank)
    
    var order = newSeq[ShapeConstraint]()
    for tensor in target.tensors:
      tensor.flattenConstraints(tensors, closed, order, program)
    
    target.shapes = order

type Matrix[T] = object
  data: seq[T]
  width: int

{.push inline.}
proc height(matrix: Matrix): int = matrix.data.len div matrix.width
proc `[]`[T](matrix: Matrix[T], y, x: int): T = matrix.data[x + y * matrix.width]
proc `[]`[T](matrix: var Matrix[T], y, x: int): var T = matrix.data[x + y * matrix.width]
proc `[]=`[T](matrix: var Matrix[T], y, x: int, value: T) = matrix.data[x + y * matrix.width] = value

proc `$`[T](matrix: Matrix[T]): string =
  for y in 0..<matrix.height:
    if y != 0:
      result &= "; "
    for x in 0..<matrix.width:
      if x != 0:
        result &= ", "
      result &= $matrix[y, x]
  result = "[" & result & "]"

proc swapRows[T](matrix: var Matrix[T], a, b: int) =
  for x in 0..<matrix.width:
    swap(matrix[a, x], matrix[b, x])
{.pop.}

proc initMatrix[T](h, w: int): Matrix[T] =
  result = Matrix[T](width: w, data: newSeq[T](w * h))

type Fraction = Rational[int]
proc solve(equations: seq[LinearIndex]): Table[RegId, Fraction] =
  var indices = initTable[RegId, int]()
  for equation in equations:
    for reg, factor in equation.factors:
      if reg notin indices:
        indices[reg] = indices.len
  
  if indices.len == 0:
    return
  
  if equations.len < indices.len:
    raise newException(ValueError, "Underconstrained linear system")
  
  var
    matrix = initMatrix[int](indices.len, indices.len + 1)
    known = initHashSet[seq[Fraction]]()
    y = 0
  for equation in equations:
    if equation.factors.len == 0:
      if equation.constant != 0:
        raise newException(ValueError, "No solution")
      continue
    
    var row = newSeq[int](matrix.width)
    for reg, factor in equation.factors:
      row[indices[reg]] = factor
    row[indices.len] = -equation.constant
    var
      normalized = newSeq[Fraction](matrix.width)
      firstValue = 0
    for x, value in row:
      if firstValue == 0:
        firstValue = value
      if firstValue == 0:
        normalized[x] = 0//1
      else:
        normalized[x] = value // firstValue
    
    if normalized notin known:
      for x in 0..<matrix.width:
        matrix[y, x] = row[x]
      known.incl(normalized)
      y += 1
      if y >= matrix.height:
        break
  
  if y < matrix.height:
    raise newException(ValueError, "Underconstrained linear system")
  
  for pivot in 0..<matrix.height:
    var maxRow = pivot
    for y in (pivot + 1)..<matrix.height:
      if abs(matrix[y, pivot]) > abs(matrix[maxRow, pivot]):
        maxRow = y
    if maxRow != pivot:
      matrix.swapRows(maxRow, pivot)
    let target = matrix[pivot, pivot]
    for y in (pivot + 1)..<matrix.height:
      let cur = matrix[y, pivot]
      if cur != 0:
        for x in 0..<matrix.width:
          matrix[y, x] = matrix[y, x] * target - matrix[pivot, x] * cur
  
  var solutions = newSeq[Fraction](indices.len)
  for y in countdown(matrix.height - 1, 0):
    var sum = matrix[y, indices.len] // 1
    for x in (y + 1)..<indices.len:
      sum -= solutions[x] * matrix[y, x]
    solutions[y] = sum / (matrix[y, y] // 1)
  
  for reg, index in indices:
    result[reg] = solutions[index]

type EvalResult* = enum
  EvalSuccess, EvalDynamicReg, EvalDynamicShape, EvalInvalidInstruction

proc eval*(instrs: seq[Instr],
           shapes: Table[TensorId, seq[int]],
           regs: var Table[RegId, int]): EvalResult =
  result = EvalSuccess
  for instr in instrs:
    var canEval = true
    for arg in instr.args:
      if arg notin regs:
        canEval = false
        break
    if canEval and instr.tensor != TensorId(0):
      canEval = instr.tensor in shapes
    if not canEval:
      return EvalDynamicReg
    case instr.kind:
      of InstrShape:
        let shape = shapes[instr.tensor]
        if shape.len == 0:
          return EvalDynamicShape
        let size =
          if instr.dim < 0:
            shape[shape.len + instr.dim]
          else:
            shape[instr.dim]
        if size < 0:
          return EvalDynamicShape
        regs[instr.res] = size
      of InstrLen:
        if shapes[instr.tensor].len == 0 or shapes[instr.tensor].anyIt(it < 0):
          return EvalDynamicShape
        regs[instr.res] = shapes[instr.tensor].prod()
      of InstrShapeLen:
        regs[instr.res] = shapes[instr.tensor].len
        if regs[instr.res] < 0:
          return EvalDynamicShape
      of InstrIndex: regs[instr.res] = instr.indexLit
      of InstrAdd: regs[instr.res] = regs[instr.args[0]] + regs[instr.args[1]]
      of InstrSub: regs[instr.res] = regs[instr.args[0]] - regs[instr.args[1]]
      of InstrMul: regs[instr.res] = regs[instr.args[0]] * regs[instr.args[1]]
      of InstrIndexDiv: regs[instr.res] = regs[instr.args[0]] div regs[instr.args[1]]
      of InstrMod: regs[instr.res] = regs[instr.args[0]] mod regs[instr.args[1]]
      of InstrWrap:
        regs[instr.res] = regs[instr.args[0]] mod regs[instr.args[1]]
        if regs[instr.res] < 0:
          regs[instr.res] += regs[instr.args[1]]
      of InstrNegate: regs[instr.res] = -regs[instr.args[0]]
      else: return EvalInvalidInstruction

proc matches(staticShape, shape: seq[int]): bool =
  if staticShape.len == 0:
    result = true
  elif staticShape.len == shape.len:
    for dim, size in staticShape:
      if size >= 0:
        if shape[dim] != staticShape[dim]:
          return false
    result = true

proc inferShapes*(program: Program,
                  target: string,
                  inputs: openArray[(TensorId, seq[int])]): Table[TensorId, seq[int]] =
  result = initTable[TensorId, seq[int]]()
  for (tensor, shape) in inputs:
    result[tensor] = shape
    let staticShape = program.tensors[tensor].shape
    if not staticShape.matches(shape):
      raise ShapeError(msg: "Given shape for " & $tensor & " is " & $shape & ", but its static shape is " & $staticShape)
  for tensorId in program.params:
    result[tensorId] = program.tensors[tensorId].shape
  for shape in program.targets[target].shapes:
    case shape.kind:
      of ShapeNone: discard
      of ShapeRank: result[shape.dest] = newSeq[int](shape.rank)
      of ShapeDims:
        var sizes = newSeq[int](shape.dims.len)
        for dim, index in shape.dims:
          var regs = initTable[RegId, int]()
          case index.setup.eval(result, regs):
            of EvalSuccess: discard
            of EvalDynamicShape: raise ShapeError(msg: "Not all shapes are known. Maybe you forgot to pass a required input tensor.")
            of EvalInvalidInstruction: raise ShapeError(msg: "Invalid instruction in tensor shape")
            of EvalDynamicReg: raise ShapeError(msg: "Unable to evaluate all instructions.")
          sizes[dim] = index.eval(regs)
        result[shape.dest] = sizes
      of ShapeCopy:
        result[shape.dest] = result[shape.src]
      of ShapeLinear:
        var equations: seq[LinearIndex] = @[]
        for tensor, dims in shape.reads:
          if tensor notin result:
            raise ShapeError(msg: "Shape of " & $tensor & " is not known, but required to infer the shape of " & $shape.dest & ". Maybe you forgot to pass a required input tensor.")
          for dim, indices in dims:
            assert indices.len == 1
            let index = indices[0]
            equations.add(index - (result[tensor][dim] - 1))
        
        var maxValues = initTable[RegId, int]()
        for reg, maxValue in solve(equations):
          maxValues[reg] = maxValue.num div maxValue.den
        
        result[shape.dest] = newSeq[int](shape.write.len)
        for dim, index in shape.write:
          result[shape.dest][dim] = index.eval(maxValues) + 1

proc staticShapeTable(tensors: seq[TensorDef]): Table[TensorId, seq[int]] =
  for it, tensor in tensors:
    let id = TensorId(it + 1)
    if tensor.shape.len > 0:
      result[id] = tensor.shape

proc inferStaticShapes*(program: Program) =
  program.assertPass("inferStaticShapes",
    requires={StageSortedShapes},
    produces={StageStaticShapes},
    preserves=ALL_STAGES
  )
  
  var shapes = program.tensors.staticShapeTable()
  for name, target in program.targets:
    for shape in target.shapes:
      var dims: seq[int] = @[]
      case shape.kind:
        of ShapeNone: discard
        of ShapeRank:
          dims = newSeq[int](shape.rank)
          for dim in dims.mitems:
            dim = -1
        of ShapeDims:
          dims = newSeq[int](shape.dims.len)
          for dim, size in shape.dims:
            var regs = initTable[RegId, int]()
            if size.setup.eval(shapes, regs) == EvalSuccess:
              dims[dim] = size.eval(regs)
            else:
              dims[dim] = -1
        of ShapeLinear:
          var equations: seq[LinearIndex] = @[]
          for tensor, dims in shape.reads:
            if tensor in shapes and shapes[tensor].len == dims.len:
              for dim, index in dims:
                assert index.len == 1
                let size = shapes[tensor][dim]
                if size >= 0:
                  equations.add(index[0] - (size - 1))
          
          var maxValues = initTable[RegId, int]()
          for reg, maxValue in solve(equations):
            maxValues[reg] = maxValue.num div maxValue.den
          
          dims = newSeq[int](shape.write.len)
          for dim, size in shape.write:
            var canEval = true
            for reg, factor in size.factors:
              if reg notin maxValues:
                canEval = false
                break
            if canEval:
              dims[dim] = size.eval(maxValues) + 1
            else:
              dims[dim] = -1
        of ShapeCopy:
          if shape.src in shapes:
            dims = shapes[shape.src]
      
      if dims.len > 0:
        if shape.dest in shapes:
          doAssert shapes[shape.dest] == dims
        else:
          shapes[shape.dest] = dims
  
  for it, tensor in program.tensors.mpairs:
    let id = TensorId(it + 1)
    case tensor.kind:
      of TensorResult, TensorRandom:
        if id in shapes:
          tensor.shape = shapes[id]
      of TensorCache:
        if id notin shapes or shapes[id].anyIt(it < 0):
          let kind = ($tensor.kind)[len("Tensor")..^1].toLowerAscii()
          raise ShapeError(msg: "Shape of " & kind & " \"" & tensor.name & "\" must be inferred at compile time")
        tensor.shape = shapes[id]
      else:
        if id in shapes:
          assert tensor.shape == shapes[id]

proc inlineStaticShapes(instrs: var seq[Instr], tensors: seq[TensorDef]) =
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
      instr = Instr(kind: InstrIndex, indexLit: size, res: instr.res)

proc inlineStaticShapes*(program: Program) =
  program.assertPass("inlineStaticShapes",
    produces={},
    requires={StageStaticShapes, StageBounds, StageTensorInstrs},
    preserves={
      StageTensors, StageFolded, StageShapes, StageSortedShapes,
      StageGenerated, StageTyped, StageBounds, StageTensorInstrs
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.setup.inlineStaticShapes(program.tensors)
      for loop in kernel.loops.mitems:
        loop.start.setup.inlineStaticShapes(program.tensors)
        loop.stop.setup.inlineStaticShapes(program.tensors)
        loop.cache.inlineStaticShapes(program.tensors)
      kernel.expr.instrs.inlineStaticShapes(program.tensors)

proc makeTensorLookups*(program: Program) =
  program.assertPass("makeTensorLookups",
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

proc identifyIndependent*(kernel: Kernel) =
  var independent = initHashSet[RegId]()
  for dim in kernel.write.dims:
    if dim.onlyRegister != RegId(0):
      independent.incl(dim.factors.peekKey())
  for loop in kernel.loops.mitems:
    if loop.iter in independent:
      loop.mode = LoopIndependent

proc identifyIndependent*(program: Program) =
  program.assertPass("identifyIndependent",
    produces={StageIndependent},
    requires={},
    preserves=ALL_STAGES
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.identifyIndependent()

proc chooseParallel*(program: Program) =
  program.assertPass("chooseParallel",
    requires={StageIndependent},
    produces={},
    preserves=ALL_STAGES
  )
  
  const LOOP_COUNT = [
    CompileCpu: 0,
    CompileThreads: 1,
    CompileGpu: 3
  ]
  
  for name, target in program.targets:
    if LOOP_COUNT[target.compileTarget] > 0:
      for kernel in target.kernels:
        var
          count = LOOP_COUNT[target.compileTarget]
          parallel: seq[Loop] = @[]
          it = 0
        while it < kernel.loops.len and count > 0:
          var loop = kernel.loops[it]
          if loop.mode >= LoopIndependent:
            loop.mode = LoopParallel
            parallel.add(loop)
            count -= 1
            kernel.loops.delete(it)
          else:
            it += 1
        kernel.loops = parallel & kernel.loops

type
  BoundsMode = enum BoundsNone, BoundsDim, BoundsLen
  BoundsInfo = object
    mode: BoundsMode
    tensor: TensorId
    dim: int

proc boundsInfo(loop: Loop): BoundsInfo =
  if loop.start.factors.len == 0 and
     loop.start.constant == 0 and
     loop.stop.onlyRegister != RegId(0) and
     loop.stop.setup.len > 0 and
     loop.stop.onlyRegister == loop.stop.setup[^1].res:
    result.tensor = loop.stop.setup[^1].tensor
    case loop.stop.setup[^1].kind:
      of InstrShape:
        result.mode = BoundsDim
        result.dim = loop.stop.setup[^1].dim
      of InstrLen:
        result.mode = BoundsLen
      else: discard

type
  TokenId = distinct int
  ShapeTokens = seq[seq[TokenId]]

proc `==`(a, b: TokenId): bool {.borrow.}
proc `$`(token: TokenId): string =
  if token == TokenId(0):
    result = "noToken"
  else:
    result = "token" & $(int(token) - 1)

proc alloc(tokens: var TokenId): TokenId =
  tokens = TokenId(int(tokens) + 1)
  result = tokens

proc buildShapeTokens(program: Program): ShapeTokens =
  program.assertAnalysis("buildShapeTokens", requires={
    StageSortedShapes, StageStaticShapes, StageFolded
  })
  
  result = newSeq[seq[TokenId]](program.tensors.len)
  var
    tokens = TokenId(0)
    valueTokens = initTable[int, TokenId]()
  for it, tensor in program.tensors:
    result[it] = newSeq[TokenId](tensor.shape.len)
    for dim, size in tensor.shape:
      if size != -1:
        if size notin valueTokens:
          valueTokens[size] = tokens.alloc()
        result[it][dim] = valueTokens[size]
  
  for name, target in program.targets:
    for shape in target.shapes:
      case shape.kind:
        of ShapeNone, ShapeRank: discard
        of ShapeDims:
          if result[shape.dest].len == 0:
            result[shape.dest] = newSeq[TokenId](shape.dims.len)
          for dim, size in shape.dims:
            if result[shape.dest][dim] == TokenId(0):
              if size.onlyRegister != RegId(0) and
                 size.setup.len > 0 and
                 size.setup[^1].res == size.onlyRegister and
                 size.setup[^1].kind == InstrShape:
                let instr = size.setup[^1]
                while result[instr.tensor].len <= instr.dim:
                  result[instr.tensor].add(tokens.alloc())
                result[shape.dest][dim] = result[instr.tensor][instr.dim]
              else:
                result[shape.dest][dim] = tokens.alloc()
        of ShapeLinear:
          var regs = initTable[RegId, TokenId]()
          for tensor, dims in shape.reads:
            while result[tensor].len < dims.len:
              result[tensor].add(tokens.alloc())
            for dim, size in dims:
              assert size.len == 1
              if size[0].onlyRegister != RegId(0):
                regs[size[0].onlyRegister] = result[tensor][dim]
          if result[shape.dest].len == 0:
            result[shape.dest] = newSeq[TokenId](shape.write.len)
          for dim, size in shape.write:
            if result[shape.dest][dim] == TokenId(0):
              if size.onlyRegister in regs:
                result[shape.dest][dim] = regs[size.onlyRegister]
              else:
                result[shape.dest][dim] = tokens.alloc()
        of ShapeCopy:
          result[shape.dest] = result[shape.src]

proc sameRange(tokens: ShapeTokens, a, b: BoundsInfo): bool =
  if a.mode == b.mode:
    case a.mode:
      of BoundsNone: result = false
      of BoundsDim:
        result = a.dim < tokens[a.tensor].len and
                 b.dim < tokens[b.tensor].len and
                 tokens[a.tensor][a.dim] == tokens[b.tensor][b.dim]
      of BoundsLen:
        result = tokens[a.tensor] == tokens[b.tensor]

proc isElementwiseMap(kernel: Kernel): bool =
  if kernel.loops.len == 1:
    let
      iter = kernel.loops[0].iter
      info = kernel.loops[0].boundsInfo
    result = kernel.reads.len == 1 and kernel.reads[0].isRaw and
             kernel.reads[0].dims[0].onlyRegister == iter and
             kernel.write.isRaw and
             kernel.write.dims[0].onlyRegister == iter and
             info.mode == BoundsLen and
             (info.tensor == kernel.reads[0].tensor or
              info.tensor == kernel.write.tensor)

proc nestElementwiseMap(kernel: Kernel, tensors: seq[TensorDef]) =
  kernel.loops = @[]
  kernel.reads[0].isRaw = false
  kernel.write.isRaw = false
  
  let tensorId = kernel.reads[0].tensor
  var iters: seq[LinearIndex] = @[]
  for dim, size in tensors[tensorId].shape:
    let iter = kernel.regs.alloc()
    iters.add(initLinearIndex(iter))
    kernel.loops.add(Loop(iter: iter, hasBounds: true))
    kernel.loops[^1].useBounds(kernel.reads[0], dim, kernel.regs)
  kernel.reads[0].dims = iters
  kernel.write.dims = iters

proc fuseLoops*(program: Program) =
  program.assertPass("fuseLoops",
    requires={StageBounds, StageIndependent, StageStaticShapes},
    produces={},
    preserves={
      StageGenerated, StageTensors, StageShapes,
      StageSortedShapes, StageTensorInstrs, StageFolded,
      StageStaticShapes, StageBounds
    }
  )
  
  let shapeTokens = program.buildShapeTokens()
  for name, target in program.targets:
    for kernelIt in 1..<target.kernels.len:
      let
        a = target.kernels[kernelIt - 1]
        b = target.kernels[kernelIt]
      
      if b.isElementwiseMap() and
         a.write.tensor == b.reads[0].tensor and
         a.loops.len > 0 and
         a.loops[0].boundsInfo.mode == BoundsDim and
         a.loops[0].mode >= LoopIndependent and
         shapeTokens[b.reads[0].tensor] == shapeTokens[b.write.tensor]:
        b.nestElementwiseMap(program.tensors)
      
      if not a.write.isRaw and
         not b.reads.anyIt(it.tensor == a.write.tensor and it.isRaw):
        for it in 0..<min(a.loops.len, b.loops.len):
          let
            aLoop = a.loops[it]
            bLoop = b.loops[it]
          if not shapeTokens.sameRange(aLoop.boundsInfo, bLoop.boundsInfo):
            break
          var dim = -1
          for dimIt, index in a.write.dims:
            if index.onlyRegister == aLoop.iter:
              dim = dimIt
              break
          if dim == -1:
            break
          let hasDependentRead = b.reads.anyIt(
            it.tensor == a.write.tensor and
            it.dims[dim].onlyRegister != bLoop.iter
          )
          if hasDependentRead:
            break
          a.loops[it].fuseNext = true

proc inlineConditions(kernel: Kernel) =
  if kernel.conds.len > 0:
    let body = kernel.expr.instrs
    kernel.expr.instrs = @[]
    
    var res = RegId(0)
    for cond in kernel.conds:
      kernel.expr.instrs.add(cond.instrs)
      if res == RegId(0):
        res = cond.res
      else:
        let newRes = kernel.regs.alloc()
        kernel.expr.instrs.add(Instr(kind: InstrAnd,
          args: @[res, cond.res],
          res: newRes
        ))
        res = newRes
    kernel.conds = @[]
    
    kernel.expr.instrs.add(Instr(kind: InstrIf,
      args: @[res],
      body: body
    ))

proc inlineConditions*(program: Program) =
  program.assertPass("inlineConditions",
    produces={StageConditions},
    preserves={
      StageBounds, StageGenerated, StageTensors, StageShapes,
      StageSortedShapes, StageTensorInstrs
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.inlineConditions()

proc tileLoops(kernel: Kernel) =
  var it = 0
  while it < kernel.loops.len:
    let loop = kernel.loops[it]
    if loop.schedule.tile and loop.mode < LoopParallel:
      let
        outer = Loop(
          iter: kernel.regs.alloc(),
          mode: loop.mode,
          hasBounds: true,
          start: loop.start,
          stop: loop.stop,
          step: loop.schedule.tileSize,
          schedule: loop.schedule.dup(_.tile = false)
        )
        inner = Loop(
          iter: loop.iter,
          mode: LoopNone,
          hasBounds: true,
          start: LinearIndex(
            factors: toTable({outer.iter: 1})
          ),
          stop: LinearIndex(
            factors: toTable({outer.iter: 1}), # TODO: Check if in bounds
            constant: loop.schedule.tileSize
          ),
          step: 1,
          schedule: DEFAULT_LOOP_SCHEDULE.dup(_.shareCache = true)
        )
      kernel.loops.delete(it)
      kernel.loops.insert([outer, inner], it)
      it += 2
    else:
      it += 1

proc tileLoops*(program: Program) =
  program.assertPass("tileLoops",
    requires = {StageBounds, StageFolded},
    produces = {StageCacheSizes},
    preserves = {
      StageBounds, StageFolded, StageStaticShapes, StageGenerated,
      StageTensors, StageShapes, StageSortedShapes
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.tileLoops()

proc boundsSize(loop: Loop, shapes: Table[TensorId, seq[int]]): (bool, int) =
  let size = loop.stop - loop.start
  var regs = initTable[RegId, int]()
  if size.setup.eval(shapes, regs) == EvalSuccess:
    result = (true, size.eval(regs))
  else:
    result = (false, 0)

proc eval(index: LinearIndex, regs: Table[RegId, OffsetInterval]): OffsetInterval =
  result.interval.min += index.constant
  result.interval.max += index.constant
  for reg, factor in index.factors:
    if reg in regs:
      result.offset = result.offset + regs[reg].offset * factor
      result.interval = result.interval + regs[reg].interval * factor
    else:
      result.offset = result.offset + LinearIndex(factors: toTable({reg: factor}))

proc inferCacheSizes(kernel: Kernel,
                       compileTarget: CompileTarget,
                       shapes: Table[TensorId, seq[int]]) =
  if kernel.reads.anyIt(it.schedule.cache):
    var
      cacheLevel = kernel.loops.len
      sizes: seq[int] = @[]
    while cacheLevel > 0:
      let loop = kernel.loops[cacheLevel - 1]
      if loop.mode >= LoopParallel or not loop.schedule.shareCache:
        break
      let (isStatic, size) = loop.boundsSize(shapes)
      if not isStatic:
        break
      sizes.add(size)
      cacheLevel -= 1
    
    var regs = initTable[RegId, OffsetInterval]()
    for it in cacheLevel..<kernel.loops.len:
      let loop = kernel.loops[it]
      regs[loop.iter] = OffsetInterval(
        offset: loop.start,
        interval: Interval(min: 0, max: sizes[kernel.loops.len - it - 1] - 1)
      )
    if compileTarget == CompileGpu:
      for it in 0..<cacheLevel:
        template loop: var Loop = kernel.loops[it]
        if loop.mode >= LoopParallel:
          if loop.tileOffset == RegId(0):
            loop.tileOffset = kernel.regs.alloc()
          regs[loop.iter] = OffsetInterval(
            offset: initLinearIndex(loop.tileOffset),
            interval: Interval(min: 0, max: loop.schedule.tileSize - 1)
          )
    
    for read in kernel.reads.mitems:
      if read.schedule.cache:
        if read.isRaw:
          continue # TODO: Support raw reads
        var cache = LocalCache(
          exists: true,
          level: cacheLevel,
          reg: kernel.regs.alloc()
        )
        for dim in read.dims:
          cache.dims.add(dim.eval(regs))
        read.cache = cache

proc inferCacheSizes*(program: Program) =
  program.assertPass("inferCacheSizes",
    requires = {StageBounds, StageFolded},
    produces = {StageCacheSizes},
    preserves = {
      StageBounds, StageFolded, StageStaticShapes, StageGenerated,
      StageTensors, StageShapes, StageSortedShapes
    }
  )
  
  let shapes = program.tensors.staticShapeTable()
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.inferCacheSizes(target.compileTarget, shapes)

proc cacheTensor(read: TensorOp, kernel: Kernel, compileTarget: CompileTarget): seq[Instr] =
  type CacheSize = enum
    CacheEqualShape,
    CacheSmaller,
    CacheEqualSize,
    CacheLarger
  
  var cacheShape: seq[int] = @[]
  for dim in read.cache.dims:
    cacheShape.add(dim.interval.max - dim.interval.min + 1)
  result.add(Instr(kind: InstrSharedCache,
    cacheSize: cacheShape.prod(),
    res: read.cache.reg
  ))
  
  var
    threadShape: seq[int] = @[]
    localOffsetIters: seq[RegId] = @[]
    offset = LinearIndex()
    stride = 1
  if compileTarget == CompileGpu:
    for it in countdown(kernel.loops.len - 1, 0):
      template loop: var Loop = kernel.loops[it]
      if loop.mode >= LoopParallel:
        threadShape.add(loop.schedule.tileSize)
        if loop.localOffset == RegId(0):
          loop.localOffset = kernel.regs.alloc()
        localOffsetIters.add(loop.localOffset)
        offset.factors[loop.localOffset] = stride
        stride *= loop.schedule.tileSize
  
  threadShape.reverse()
  localOffsetIters.reverse()
  
  let cacheSize =
    if threadShape == cacheShape:
      CacheEqualShape
    elif cacheShape.prod() < threadShape.prod():
      CacheSmaller
    elif cacheShape.prod() == threadShape.prod():
      CacheEqualSize
    else:
      CacheLarger
  
  let start = offset.unfold(kernel.regs)
  result.add(start.instrs)
  
  let iter =
    if cacheSize in {CacheEqualShape, CacheSmaller, CacheEqualSize}:
      start.res
    else:
      kernel.regs.alloc()
  
  var body: seq[Instr] = @[]
  
  var
    dims: seq[LinearIndex] = @[]
    cur = iter
  for it in countdown(read.cache.dims.len - 1, 0):
    let
      dim = read.cache.dims[it]
      size = dim.interval.max - dim.interval.min + 1
      sizeReg = kernel.regs.alloc()
    
    let localOffset =
      if cacheSize == CacheEqualShape:
        localOffsetIters[it]
      elif it == 0:
        cur
      else:
        let offset = kernel.regs.alloc()
        body.add(Instr(kind: InstrIndex,
          indexLit: size,
          res: sizeReg
        ))
        body.add(Instr(kind: InstrMod,
          args: @[cur, sizeReg],
          res: offset
        ))
        let newCur = kernel.regs.alloc()
        body.add(Instr(kind: InstrIndexDiv,
          args: @[cur, sizeReg],
          res: newCur
        ))
        cur = newCur
        offset
    
    let readDim = unfold(dim.offset + localOffset.initLinearIndex(), kernel.regs)
    body.add(readDim.instrs)
    dims.add(initLinearIndex(readDim.res))
  
  dims.reverse()
  let index = expandTensorIndex(dims, read.tensor, kernel.regs)
  body.add(index.instrs)
  let value = kernel.regs.alloc()
  body.add(Instr(kind: InstrRead,
    args: @[index.res],
    tensor: read.tensor,
    res: value
  ))
  body.add(Instr(kind: InstrCacheWrite,
    args: @[read.cache.reg, iter, value]
  ))
  
  if cacheSize in {CacheEqualShape, CacheEqualSize}:
    result.add(body)
  else:
    let stop = kernel.regs.alloc()
    result.add(Instr(kind: InstrIndex,
      indexLit: cacheShape.prod(),
      res: stop
    ))
    if cacheSize == CacheSmaller:
      let cond = kernel.regs.alloc()
      result.add(Instr(kind: InstrLt,
        args: @[iter, stop],
        res: cond
      ))
      result.add(Instr(kind: InstrIf,
        args: @[cond],
        body: body
      ))
    else:
      result.add(Instr(kind: InstrLoop,
        args: @[start.res, stop],
        loopIter: iter,
        loopStep: threadShape.prod(),
        body: body
      ))

proc cacheTensors*(kernel: Kernel, compileTarget: CompileTarget) =
  for read in kernel.reads:
    if read.cache.exists:
      let instrs = cacheTensor(read, kernel, compileTarget)
      if read.cache.level == 0:
        kernel.setup.add(instrs)
      else:
        kernel.loops[read.cache.level - 1].cache.add(instrs)

proc cacheTensors*(program: Program) =
  program.assertPass("cacheTensors",
    requires = {StageCacheSizes},
    preserves = {
      StageBounds, StageFolded, StageStaticShapes, StageGenerated,
      StageTensors, StageShapes, StageSortedShapes, StageCacheSizes
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.cacheTensors(target.compileTarget)

proc inlineLoop(kernel: Kernel, compileTarget: CompileTarget) =
  let loop = kernel.loops.pop()
  if loop.cache.len > 0:
    if compileTarget == CompileGpu:
      kernel.expr.instrs.insert(Instr(kind: InstrBarrier))
    kernel.expr.instrs.insert(loop.cache)
    if compileTarget == CompileGpu:
      kernel.expr.instrs.insert(Instr(kind: InstrBarrier))
  if loop.mode >= LoopParallel:
    case compileTarget:
      of CompileCpu:
        raise StageError(msg: "Parallel loops are not supported by CPU target")
      of CompileThreads:
        let (rangeBegin, rangeEnd) = (kernel.regs.alloc(), kernel.regs.alloc())
        kernel.expr.instrs = @[Instr(kind: InstrThreads,
          args: @[loop.start.onlyRegister, loop.stop.onlyRegister],
          threadsBegin: rangeBegin, threadsEnd: rangeEnd,
          body: @[Instr(kind: InstrLoop,
            args: @[rangeBegin, rangeEnd],
            loopIter: loop.iter,
            loopStep: 1,
            body: kernel.expr.instrs
          )]
        )]
      of CompileGpu:
        var
          instr = Instr(kind: InstrGpu,
            args: @[loop.start.onlyRegister, loop.stop.onlyRegister]
          )
          loops = @[loop]
        while kernel.loops.len > 0 and
              kernel.loops[^1].mode >= LoopParallel:
          let loop = kernel.loops.pop()
          loops.add(loop)
          instr.args.add([loop.start.onlyRegister, loop.stop.onlyRegister])
        
        var conds: seq[RegId] = @[]
        for it, loop in loops:
          let
            localOffset =
              if loop.localOffset != RegId(0):
                loop.localOffset
              else:
                kernel.regs.alloc()
            index = GpuIndex(
              group: kernel.regs.alloc(),
              local: localOffset,
              size: loop.schedule.tileSize
            )
            offset =
              if loop.tileOffset != RegId(0):
                loop.tileOffset
              else:
                kernel.regs.alloc()
            sizeReg = kernel.regs.alloc()
          instr.body.add(Instr(kind: InstrIndex,
            indexLit: index.size,
            res: sizeReg
          ))
          instr.body.add(Instr(kind: InstrMul,
            args: @[index.group, sizeReg],
            res: offset
          ))
          instr.body.add(Instr(kind: InstrAdd,
            args: @[offset, index.local],
            res: loop.iter
          ))
          instr.gpuIndices.add(index)
          
          if loop.stop.setup[^1].kind != InstrIndex or
             loop.stop.setup[^1].indexLit mod index.size != 0:
            let
              isInRange = kernel.regs.alloc()
              max = loop.stop.setup[^1].res
            instr.body.add(Instr(kind: InstrLt,
              args: @[loop.iter, max],
              res: isInRange
            ))
            conds.add(isInRange)
        
        if conds.len > 0:
          var cond = conds[0]
          for it in 1..<conds.len:
            let res = kernel.regs.alloc()
            instr.body.add(Instr(kind: InstrAnd,
              args: @[cond, conds[it]],
              res: res
            ))
            cond = res
          instr.body.add(Instr(kind: InstrIf,
            args: @[cond],
            body: kernel.expr.instrs
          ))
        else:
          instr.body.add(kernel.expr.instrs)
        
        kernel.expr.instrs = @[instr]
        for loop in loops:
          kernel.expr.instrs.insert(loop.start.setup)
          kernel.expr.instrs.insert(loop.stop.setup)
        return
  else:
    kernel.expr.instrs = @[Instr(kind: InstrLoop,
      args: @[loop.start.onlyRegister, loop.stop.onlyRegister],
      loopIter: loop.iter,
      loopStep: loop.step,
      loopFuseNext: loop.fuseNext,
      body: kernel.expr.instrs
    )]
  kernel.expr.instrs.insert(loop.start.setup)
  kernel.expr.instrs.insert(loop.stop.setup)

proc inlineLoops(target: Target, cur, untilLevel: int) =
  let kernel = target.kernels[cur]
  while kernel.loops.len > untilLevel:
    while kernel.loops[^1].fuseNext:
      target.inlineLoops(cur + 1, kernel.loops.len)
      let nextKernel = target.kernels[cur + 1]
      var
        instrs = nextKernel.expr.instrs
        setup = nextKernel.setup
        subs = initTable[RegId, RegId]()
      for it in 0..<kernel.loops.len:
        subs[nextKernel.loops[it].iter] = kernel.loops[it].iter
      for it in 0..<nextKernel.regs.len:
        let reg = RegId(it + 1)
        if reg notin subs:
          subs[reg] = kernel.regs.alloc(nextKernel.regs[it])
      instrs.substitute(subs)
      setup.substitute(subs)
      kernel.expr.instrs.add(instrs)
      kernel.setup.add(setup)
      for it in 0..<kernel.loops.len:
        kernel.loops[it].fuseNext = nextKernel.loops[it].fuseNext
      target.kernels.delete(cur + 1)
    kernel.inlineLoop(target.compileTarget)

proc inlineLoops*(program: Program) =
  program.assertPass("inlineLoops",
    requires={StageBounds, StageConditions},
    produces={StageLoops},
    preserves={
      StageGenerated, StageTensors, StageShapes,
      StageSortedShapes, StageTensorInstrs, StageConditions
    }
  )
  
  for name, target in program.targets.mpairs:
    var it = 0
    while it < target.kernels.len:
      target.inlineLoops(it, 0)
      it += 1
    
    for kernel in target.kernels:
      kernel.setup.add(kernel.expr.instrs)
      kernel.expr = Expr()

proc liftInvariants(instrs: var seq[Instr],
                     regs: var seq[int],
                     levels: var seq[seq[Instr]],
                     minLevel: int) =
  var it = 0
  while it < instrs.len:
    let instr = instrs[it]
    if instr.body.len > 0:
      levels.add(@[])
      var bodyMinLevel = minLevel
      case instr.kind:
        of InstrLoop:
          regs[instr.loopIter] = levels.len
        of InstrThreads:
          regs[instr.threadsBegin] = levels.len
          regs[instr.threadsEnd] = levels.len
          bodyMinLevel = levels.len
        of InstrGpu:
          for index in instr.gpuIndices:
            regs[index.local] = levels.len
            regs[index.group] = levels.len
          bodyMinLevel = levels.len
        of InstrIf:
          bodyMinLevel = levels.len # TODO: Only for InstrRead, InstrArrayRead, ...
        else: discard
      instrs[it].body.liftInvariants(regs, levels, bodyMinLevel)
      let level = levels.pop()
      instrs.insert(level, it)
      it += level.len
    
    if instr.kind in SIDE_EFFECT_INSTRS:
      if instr.res != RegId(0):
        regs[instr.res] = levels.len 
    else:
      var instrLevel = 0
      if instr.kind notin {InstrShape, InstrLen, InstrShapeLen, InstrEpoch}:
        instrLevel = minLevel
      
      for arg in instr.args:
        if regs[arg] > instrLevel:
          instrLevel = regs[arg]
      
      if instr.res != RegId(0):
        regs[instr.res] = instrLevel
      
      if instrLevel < levels.len:
        levels[instrLevel].add(instr)
        instrs.delete(it)
        continue
    
    it += 1

proc liftInvariants*(program: Program) =
  program.assertPass("liftInvariants",
    produces={},
    requires={StageTensorInstrs},
    preserves={
      StageGenerated, StageTensors, StageShapes,
      StageSortedShapes, StageBounds,
      StageTensorInstrs, StageLoops, StageConditions
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      var
        regs = newSeq[int](kernel.regs.len)
        levels: seq[seq[Instr]] = @[]
      kernel.setup.liftInvariants(regs, levels, 0)

proc collectClosures(instrs: var seq[Instr], regs: var seq[int], level: int): HashSet[RegId] =
  for instr in instrs.mitems:
    var used = instr.body.collectClosures(regs, level + 1)
    
    # TODO: Refactor using iterator which returns all registers defined by the instruction
    case instr.kind:
      of InstrLoop:
        regs[instr.loopIter] = level + 1
      of InstrThreads:
        regs[instr.threadsBegin] = level + 1
        regs[instr.threadsEnd] = level + 1
      of InstrGpu:
        for index in instr.gpuIndices:
          regs[index.local] = level + 1
          regs[index.group] = level + 1
      else: discard
    
    if instr.kind in {InstrThreads, InstrGpu}:
      var closure = ParallelClosure()
      for reg in used:
        if regs[reg] <= level:
          closure.regs.add(reg)
      
      var tensors: seq[TensorId] = @[]
      for tensor in instr.body.collectTensors():
        closure.tensors.add(tensor)
      
      case instr.kind:
        of InstrThreads: instr.threadsClosure = closure
        of InstrGpu: instr.gpuClosure = closure
        else: discard
    
    for arg in instr.args:
      used.incl(arg)
    
    if instr.res != RegId(0):
      regs[instr.res] = level
    
    result = result.union(used)

proc collectClosures*(program: Program) =
  program.assertPass("collectClosures",
    produces={},
    requires={StageLoops},
    preserves={
      StageGenerated, StageTensors, StageShapes,
      StageSortedShapes, StageBounds,
      StageTensorInstrs, StageLoops, StageConditions
    }
  )
  
  for name, target in program.targets:
    for kernel in target.kernels:
      var regs = newSeq[int](kernel.regs.len)
      discard kernel.setup.collectClosures(regs, 0)

proc extractClosure(regs: seq[bool], closure: ParallelClosure): seq[bool] =
  result = newSeq[bool](regs.len)
  for reg in closure.regs:
    if not regs[reg]:
      raise ValidationError(msg: $reg & " cannot be captured because it is not defined")
    result[reg] = regs[reg]

proc validate(instrs: seq[Instr], regs: var seq[bool]) =
  for instr in instrs:
    for arg in instr.args:
      if not regs[arg]:
        raise ValidationError(msg: $arg & " is not defined")
    
    case instr.kind:
      of InstrIf:
        instr.body.validate(regs)
      of InstrLoop:
        regs[instr.loopIter] = true
        instr.body.validate(regs)
      of InstrThreads:
        var closure = regs.extractClosure(instr.threadsClosure)
        closure[instr.threadsBegin] = true
        closure[instr.threadsEnd] = true
        instr.body.validate(closure)
      of InstrGpu:
        var closure = regs.extractClosure(instr.gpuClosure)
        for index in instr.gpuIndices:
          closure[index.local] = true
          closure[index.group] = true
        instr.body.validate(closure)
      else: discard
    
    if instr.res != RegId(0):
      regs[instr.res] = true

proc validate(index: LinearIndex, regs: var seq[bool]) =
  index.setup.validate(regs)
  for reg, factor in index.factors:
    if not regs[reg]:
      raise ValidationError(msg: $reg & " is not defined")

proc validate(kernel: Kernel) =
  if kernel.generator.kind != GenNone:
    return
  var regs = newSeq[bool](kernel.regs.len)
  kernel.setup.validate(regs)
  for loop in kernel.loops:
    loop.start.validate(regs)
    loop.stop.validate(regs)
    regs[loop.iter] = true
  # TODO: Conditions, Write, Read Index
  for read in kernel.reads:
    regs[read.data] = true
  kernel.expr.instrs.validate(regs)

proc validate*(program: Program) =
  program.assertPass("validate", preserves = ALL_STAGES)
  
  for name, target in program.targets:
    for kernel in target.kernels:
      kernel.validate()
