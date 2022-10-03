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

# Compile exprgrad's DSL to LLVM IR

import std/[tables, sets]
import wrappers/llvm
import runtimes/gpu
import ir, clgen

proc toLlvm(scalarType: ScalarType): TypeRef =
  case scalarType:
    of Scalar32: result = float_type()
    of Scalar64: result = double_type()

type Builtin = object
  scalarType: ScalarType
  # Tensor
  tensor: ValueRef
  shape: ValueRef
  len: ValueRef
  shapeLen: ValueRef
  # Debug
  debugIndex: ValueRef
  debugScalar: ValueRef
  # Values
  epoch: ValueRef
  # Threads
  runThreads: ValueRef
  joinThreads: ValueRef
  # Gpu
  runGpuKernel: ValueRef
  setGpuKernelIndex: ValueRef
  setGpuKernelTensor: ValueRef
  # Intrinsics
  sin: ValueRef
  cos: ValueRef
  exp: ValueRef
  ln: ValueRef
  sqrt: ValueRef
  pow: ValueRef

proc modelPtrType(): TypeRef = pointer_type(int8_type(), 0)
proc voidPtrType(): TypeRef = pointer_type(int8_type(), 0)

proc tensorSignature(builtin: Builtin): TypeRef =
  functionType(pointer_type(builtin.scalarType.toLlvm(), 0), [
    modelPtrType(), nimIntType()
  ])

proc shapeSignature(builtin: Builtin): TypeRef =
  functionType(nimIntType(), [
    modelPtrType(), nimIntType(), nimIntType()
  ])

proc lenSignature(builtin: Builtin): TypeRef =
  functionType(nimIntType(), [modelPtrType(), nimIntType()])

proc shapeLenSignature(builtin: Builtin): TypeRef =
  functionType(nimIntType(), [modelPtrType(), nimIntType()])

proc debugIndexSignature(builtin: Builtin): TypeRef =
  functionType(void_type(), [modelPtrType(), nimIntType()])

proc debugScalarSignature(builtin: Builtin): TypeRef =
  functionType(void_type(), [modelPtrType(), builtin.scalarType.toLlvm()])

proc epochSignature(builtin: Builtin): TypeRef =
  functionType(nimIntType(), [modelPtrType()])

proc taskProcSignature(): TypeRef =
  functionType(void_type(), [
    modelPtrType(), nimIntType(), nimIntType(), voidPtrType()
  ])

proc runThreadsSignature(builtin: Builtin): TypeRef =
  functionType(void_type(), [
    modelPtrType(), nimIntType(), nimIntType(), voidPtrType(),
    taskProcSignature().pointer_type(0)
  ])

proc joinThreadsSignature(builtin: Builtin): TypeRef =
  functionType(void_type(), [modelPtrType()])

proc setGpuKernelIndexSignature(builtin: Builtin): TypeRef =
  functionType(void_type(), [
    modelPtrType(), nimIntType(),
    nimIntType(), nimIntType()
  ])

proc setGpuKernelTensorSignature(builtin: Builtin): TypeRef =
  functionType(void_type(), [
    modelPtrType(), nimIntType(),
    nimIntType(), nimIntType()
  ])

proc runGpuKernelSignature(builtin: Builtin): TypeRef =
  functionType(void_type(), [
    modelPtrType(), nimIntType(),
    nimIntType(),
    nimIntType().pointer_type(0),
    nimIntType().pointer_type(0)
  ])

proc scalarUnaryIntrinsicSignature(builtin: Builtin): TypeRef =
  functionType(builtin.scalarType.toLlvm(), [builtin.scalarType.toLlvm()])

proc scalarBinaryIntrinsicSignature(builtin: Builtin): TypeRef =
  functionType(builtin.scalarType.toLlvm(), [
    builtin.scalarType.toLlvm(), builtin.scalarType.toLlvm()
  ])

proc initBuiltin(module: ModuleRef, program: Program): Builtin =
  result = Builtin(scalarType: program.scalarType)
  result.tensor = module.add_function("tensor", result.tensorSignature())
  result.shape = module.add_function("shape", result.shapeSignature())
  result.len = module.add_function("len", result.lenSignature())
  result.shapeLen = module.add_function("shapeLen", result.shapeLenSignature())
  result.debugIndex = module.add_function("debugIndex", result.debugIndexSignature())
  result.debugScalar = module.add_function("debugScalar", result.debugScalarSignature())
  result.epoch = module.add_function("epoch", result.epochSignature())
  result.runThreads = module.add_function("runThreads", result.runThreadsSignature())
  result.joinThreads = module.add_function("joinThreads", result.joinThreadsSignature())
  result.runGpuKernel = module.add_function("runGpuKernel", result.runGpuKernelSignature())
  result.setGpuKernelIndex = module.add_function("setGpuKernelIndex", result.setGpuKernelIndexSignature())
  result.setGpuKernelTensor = module.add_function("setGpuKernelTensor", result.setGpuKernelTensorSignature())
  
  let typePostfix = [Scalar32: "f32", Scalar64: "f64"][result.scalarType]
  result.sin = module.add_function(cstring("llvm.sin." & typePostfix), result.scalarUnaryIntrinsicSignature())
  result.cos = module.add_function(cstring("llvm.cos." & typePostfix), result.scalarUnaryIntrinsicSignature())
  result.exp = module.add_function(cstring("llvm.exp." & typePostfix), result.scalarUnaryIntrinsicSignature())
  result.ln = module.add_function(cstring("llvm.log." & typePostfix), result.scalarUnaryIntrinsicSignature())
  result.sqrt = module.add_function(cstring("llvm.sqrt." & typePostfix), result.scalarUnaryIntrinsicSignature())
  result.pow = module.add_function(cstring("llvm.pow." & typePostfix), result.scalarBinaryIntrinsicSignature())

type Context = ref object
  program: Program
  target: string
  kernel: Kernel
  kernelId: KernelId
  module: ModuleRef
  builder: BuilderRef
  fn: ValueRef
  builtin: Builtin
  tensors: seq[ValueRef]
  regs: seq[ValueRef]
  gpuSources: seq[GpuKernelSource]

proc `[]`(ctx: Context, reg: RegId): ValueRef = ctx.regs[reg]
proc `[]=`(ctx: Context, reg: RegId, val: ValueRef) = ctx.regs[reg] = val
proc `[]`(ctx: Context, tensor: TensorId): ValueRef = ctx.tensors[tensor]
proc `[]=`(ctx: Context, tensor: TensorId, val: ValueRef) = ctx.tensors[tensor] = val

proc scalarType(ctx: Context): TypeRef =
  ctx.program.scalarType.toLlvm()

proc toLlvm(typ: Type, ctx: Context): TypeRef =
  case typ.kind:
    of TypeIndex: result = nimIntType()
    of TypeScalar: result = ctx.scalarType()
    of TypeBoolean: result = int1_type()
    of TypeArray: result = pointer_type(typ.item.toLlvm(ctx), 0)

proc buildArray(ctx: Context, itemType: TypeRef, items: openArray[ValueRef], res: cstring): ValueRef =
  let currentBlock = ctx.builder.get_insert_block()
  ctx.builder.positionBuilderAtStart(ctx.fn.get_entry_basic_block())
  result = ctx.builder.build_array_alloca(
    itemType,
    constNimInt(items.len),
    res
  )
  ctx.builder.position_builder_at_end(currentBlock)
  for it, item in items:
    let valuePtr = ctx.builder.buildGep2(
      itemType,
      result,
      [constNimInt(it)],
      "array_value_ptr"
    )
    discard ctx.builder.build_store(item, valuePtr)

proc toLlvm(instrs: seq[Instr], ctx: Context) =
  let builder = ctx.builder
  for instr in instrs:
    var res = ValueRef(nil)
    
    template binop(op) =
      res = builder.op(
        ctx[instr.args[0]], ctx[instr.args[1]], cstring($instr.res)
      )
    
    template unop(op) =
      res = builder.op(ctx[instr.args[0]], cstring($instr.res))
    
    template genericOp(opKind, indexOp, scalarOp) =
      if ctx.kernel.regs[instr.args[0]].typ.kind == TypeScalar:
        op_kind(scalar_op)
      else:
        op_kind(index_op)
    
    case instr.kind:
      of InstrIndex:
        res = constNimInt(instr.indexLit)
      of InstrScalar:
        res = const_real(ctx.scalarType(), cdouble(instr.scalarLit))
      of InstrBoolean:
        res = const_int(int1_type(), culonglong(ord(instr.booleanLit)), 0)
      of InstrAdd: genericOp(binop, build_nsw_add, build_fadd)
      of InstrSub: genericOp(binop, build_nsw_sub, build_fsub)
      of InstrMul: genericOp(binop, build_nsw_mul, build_fmul)
      of InstrDiv: binop(build_fdiv)
      of InstrIndexDiv: binop(build_sdiv)
      of InstrMod: binop(build_srem)
      of InstrWrap:
        res = builder.build_srem(ctx[instr.args[0]], ctx[instr.args[1]], cstring($instr.res & "_mod"))
        res = builder.build_add(res, ctx[instr.args[1]], cstring($instr.res & "_offset"))
        res = builder.build_srem(res, ctx[instr.args[1]], cstring($instr.res))
      of InstrNegate: genericOp(unop, buildNegate, buildFnegate)
      of InstrSelect:
        res = builder.build_select(
          ctx[instr.args[0]],
          ctx[instr.args[1]],
          ctx[instr.args[2]],
          cstring($instr.res)
        )
      of InstrSin, InstrCos, InstrExp, InstrLn, InstrSqrt:
        let fn = case instr.kind:
          of InstrSin: ctx.builtin.sin
          of InstrCos: ctx.builtin.cos
          of InstrExp: ctx.builtin.exp
          of InstrLn: ctx.builtin.ln
          of InstrSqrt: ctx.builtin.sqrt
          else: nil
        res = builder.buildCall2(
          ctx.builtin.scalarUnaryIntrinsicSignature(),
          fn, [ctx[instr.args[0]]], cstring($instr.res)
        )
      of InstrPow:
        let fn = case instr.kind:
          of InstrPow: ctx.builtin.pow
          else: nil
        res = builder.buildCall2(
          ctx.builtin.scalarBinaryIntrinsicSignature(),
          fn, [ctx[instr.args[0]], ctx[instr.args[1]]],
          cstring($instr.res)
        )
      of InstrEq: genericOp(binop, buildIcmpEq, buildFcmpOeq)
      of InstrLt: genericOp(binop, buildIcmpSlt, buildFcmpOlt)
      of InstrLe: genericOp(binop, buildIcmpSle, buildFcmpOle)
      of InstrAnd: binop(build_and)
      of InstrOr: binop(build_or)
      of InstrToScalar, InstrToIndex:
        let
          fromTyp = ctx.kernel.regs[instr.args[0]].typ
          toTyp = ctx.kernel.regs[instr.res].typ
        
        template convert(name: untyped) =
          res = builder.name(ctx[instr.args[0]], toTyp.toLlvm(ctx), cstring($instr.res))
        
        if fromTyp.kind == TypeIndex and toTyp.kind == TypeScalar:
          convert(build_si_to_fp)
        elif fromTyp.kind == TypeScalar and toTyp.kind == TypeIndex:
          convert(build_fp_to_si)
        else:
          raise GeneratorError(msg: "Unable to convert " & $fromTyp & " to " & $toTyp)
      of InstrRead, InstrWrite, InstrOverwrite:
        let
          align = cuint(4) # TODO
          valuePtr = builder.buildGep2(
            ctx.scalarType(), ctx[instr.tensor],
            [ctx[instr.args[0]]], "value_ptr"
          )
        valuePtr.set_is_in_bounds(1)
        
        case instr.kind:
          of InstrWrite, InstrRead:
            let value = builder.build_load2(
              ctx.scalarType(), valuePtr, cstring($instr.res)
            )
            value.set_alignment(align)
            case instr.kind:
              of InstrRead: res = value
              of InstrWrite:
                builder.build_store(builder.build_fadd(
                  value, ctx[instr.args[1]], "new_value"
                ), valuePtr).set_alignment(align)
              else: discard
          of InstrOverwrite:
            builder.build_store(ctx[instr.args[1]], valuePtr).set_alignment(align)
          else: discard
      of InstrLen:
        res = builder.buildCall2(ctx.builtin.lenSignature(), ctx.builtin.len, [
          ctx.fn.get_param(0),
          constNimInt(int(instr.tensor))
        ], cstring($instr.res))
      of InstrShape:
        res = builder.buildCall2(ctx.builtin.shapeSignature(), ctx.builtin.shape, [
          ctx.fn.get_param(0),
          constNimInt(int(instr.tensor)),
          constNimInt(instr.dim)
        ], cstring($instr.res))
      of InstrShapeLen:
        res = builder.buildCall2(ctx.builtin.shapeLenSignature(), ctx.builtin.shapeLen, [
          ctx.fn.get_param(0),
          constNimInt(int(instr.tensor))
        ], cstring($instr.res))
      of InstrEpoch:
        res = builder.buildCall2(ctx.builtin.epochSignature(),
          ctx.builtin.epoch, [ctx.fn.get_param(0)], cstring($instr.res)
        )
      of InstrLoop:
        let
          headerBlock = builder.get_insert_block()
          condBlock = ctx.fn.append_basic_block("cond")
          bodyBlock = ctx.fn.append_basic_block("body")
          endBlock = ctx.fn.append_basic_block("end")
          incrBlock = ctx.fn.append_basic_block("incr")
        discard builder.build_br(condBlock)
        builder.position_builder_at_end(condBlock)
        ctx[instr.loopIter] = builder.build_phi(
          ctx.kernel.regs[instr.loopIter].typ.toLlvm(ctx),
          cstring("iter_" & $instr.loopIter)
        )
        let
          cond = builder.buildIcmpEq(
            ctx[instr.loopIter],
            ctx[instr.args[1]],
            "exitcond"
          )
        discard builder.build_cond_br(cond, endBlock, bodyBlock)
        builder.position_builder_at_end(bodyBlock)
        instr.body.toLlvm(ctx)
        discard builder.build_br(incrBlock)
        
        if instr.loopStep <= 0:
          raise GeneratorError(msg: "Loop step size must be a positive integer.")
        
        builder.position_builder_at_end(incrBlock)
        let newIter = builder.build_add(
          ctx[instr.loopIter],
          constNimInt(instr.loopStep),
          "incr_iter"
        )
        discard builder.build_br(condBlock)
        
        ctx[instr.loopIter].addIncoming(
          [ctx[instr.args[0]], newIter],
          [headerBlock, incrBlock]
        )
        builder.position_builder_at_end(endBlock)
      of InstrThreads:
        var closureFields: seq[TypeRef] = @[]
        for reg in instr.threadsClosure.regs:
          closureFields.add(ctx.kernel.regs[reg].typ.toLlvm(ctx))
        for tensor in instr.threadsClosure.tensors:
          closureFields.add(pointer_type(ctx.scalarType(), 0))
        let closureType = structType(closureFields)
        
        let
          currentBlock = builder.get_insert_block()
          sig = taskProcSignature()
          task = ctx.module.add_function(cstring($ctx.kernelId & "_task"), sig)
          entry = task.append_basic_block(cstring("entry"))
        
        let taskCtx = Context(
          program: ctx.program,
          target: ctx.target,
          kernel: ctx.kernel,
          kernel_id: ctx.kernelId,
          module: ctx.module,
          builder: ctx.builder,
          fn: task,
          builtin: ctx.builtin,
          regs: newSeq[ValueRef](ctx.kernel.regs.len),
          tensors: newSeq[ValueRef](ctx.program.tensors.len)
        )
        taskCtx[instr.threadsBegin] = task.get_param(1)
        taskCtx[instr.threadsEnd] = task.get_param(2)
        
        var offset = 0
        builder.position_builder_at_end(currentBlock)
        let closure = builder.build_alloca(closureType, "closure")
        builder.position_builder_at_end(entry)
        let taskClosure = builder.build_bit_cast(task.get_param(3), closureType.pointer_type(0), "closure")
        
        template makeClosure(ids) =
          for id in ids:
            block:
              builder.position_builder_at_end(currentBlock)
              let fieldPtr = builder.buildGep2(closureType, closure, [
                constInt32(0), constInt32(int32(offset))
              ], "field_ptr")
              discard builder.build_store(ctx[id], field_ptr)
            block:
              builder.position_builder_at_end(entry)
              let fieldPtr = builder.buildGep2(closureType, taskClosure, [
                constInt32(0), constInt32(int32(offset))
              ], "field_ptr")
              taskCtx[id] = builder.build_load2(closureFields[offset], field_ptr, cstring($id))
            offset += 1
        
        makeClosure(instr.threadsClosure.regs)
        makeClosure(instr.threadsClosure.tensors)
        
        builder.position_builder_at_end(currentBlock)
        discard builder.buildCall2(ctx.builtin.runThreadsSignature(), ctx.builtin.runThreads, [
          ctx.fn.get_param(0), ctx[instr.args[0]], ctx[instr.args[1]],
          builder.build_bit_cast(closure, voidPtrType(), "data"),
          task
        ], cstring(""))
        
        builder.position_builder_at_end(entry)
        instr.body.toLlvm(taskCtx)
        discard builder.build_ret()
        
        builder.position_builder_at_end(currentBlock)
        discard builder.buildCall2(
          ctx.builtin.joinThreadsSignature(),
          ctx.builtin.joinThreads,
          [ctx.fn.get_param(0)],
          cstring("")
        )
      of InstrArray:
        let
          arrayType = ctx.kernel.regs[instr.res].typ
          itemType = arrayType.item.toLlvm(ctx)
        var items = newSeq[ValueRef](instr.args.len)
        for it, arg in instr.args:
          items[it] = ctx[arg]
        res = ctx.buildArray(itemType, items, cstring($instr.res))
      of InstrArrayRead:
        let
          arrayType = ctx.kernel.regs[instr.args[0]].typ
          itemType = arrayType.item.toLlvm(ctx)
          valuePtr = builder.buildGep2(
            itemType,
            ctx[instr.args[0]],
            [ctx[instr.args[1]]],
            "array_value_ptr"
          )
        res = builder.build_load2(itemType, valuePtr, cstring($instr.res))
      of InstrArrayLen:
        res = constNimInt(ctx.kernel.regs[instr.args[0]].typ.len)
      of InstrGpu:
        when defined(opencl):
          let source = instr.body.to_cl(instr.gpu_closure, instr.gpu_indices, ctx.kernel, ctx.program)
          ctx.gpu_sources.add(GpuKernelSource(name: "cl_kernel", source: source))
        let gpuKernelId = ctx.gpuSources.len
        
        var arg = 0
        for tensor in instr.gpuClosure.tensors:
          discard builder.buildCall2(ctx.builtin.setGpuKernelTensorSignature(), ctx.builtin.setGpuKernelTensor, [
            ctx.fn.get_param(0),
            constNimInt(gpuKernelId),
            constNimInt(arg),
            constNimInt(int(tensor))
          ], cstring(""))
          arg += 1
        for reg in instr.gpuClosure.regs:
          let typ = ctx.kernel.regs[reg].typ
          case typ.kind:
            of TypeIndex:
              discard builder.buildCall2(ctx.builtin.setGpuKernelIndexSignature(), ctx.builtin.setGpuKernelIndex, [
                ctx.fn.get_param(0),
                constNimInt(gpuKernelId),
                constNimInt(arg),
                ctx[reg]
              ], cstring(""))
            else:
              raise GeneratorError(msg: "Unable to pass " & $reg & " of type " & $typ & " to gpu kernel")
          arg += 1
        
        var
          globalSize: seq[ValueRef] = @[]
          localSize: seq[ValueRef] = @[]
        for it, index in instr.gpuIndices:
          globalSize.add(ctx[instr.args[2 * it + 1]])
          localSize.add(constNimInt(index.size))
        
        let
          globalSizeArray = ctx.buildArray(nimIntType(), globalSize, "global_size")
          localSizeArray = ctx.buildArray(nimIntType(), localSize, "local_size")
        discard builder.buildCall2(ctx.builtin.runGpuKernelSignature(), ctx.builtin.runGpuKernel, [
          ctx.fn.get_param(0),
          constNimInt(gpuKernelId),
          constNimInt(instr.args.len div 2),
          globalSizeArray,
          localSizeArray
        ], cstring(""))
      else:
        raise GeneratorError(msg: "Unable to generate LLVM IR for " & $instr.kind)
    
    if not res.isNil:
      assert instr.res != RegId(0)
      ctx[instr.res] = res

proc toLlvm(kernel: Kernel, kernelId: KernelId, ctx: Context) =
  let
    builder = ctx.builder
    kernelBlock = ctx.fn.append_basic_block(cstring($kernelId))
  discard builder.build_br(kernelBlock)
  builder.position_builder_at_end(kernelBlock)
  
  ctx.regs = newSeq[ValueRef](kernel.regs.len)
  kernel.setup.toLlvm(ctx)

proc toLlvm*(program: Program): (ModuleRef, seq[GpuKernelSource]) =
  program.assertGen("llvm", requires={
    StageTyped, StageGenerated, StageTensors, StageShapes,
    StageLoops, StageTensorInstrs, StageSortedShapes,
    StageConditions
  })

  let
    module = module_create_with_name("module")
    builtin = initBuiltin(module, program)
  var gpuSources: seq[GpuKernelSource] = @[]
  for name, target in program.targets:
    let
      sig = functionType(void_type(), [modelPtrType()])
      fn = module.add_function(cstring("target_" & name), sig)
      entry = fn.append_basic_block("entry")
      builder = create_builder()
    
    builder.enableFastMath()
    
    builder.position_builder_at_end(entry)
    var ctx = Context(
      program: program,
      module: module,
      builder: builder,
      fn: fn,
      builtin: builtin,
      target: name,
      tensors: newSeq[ValueRef](program.tensors.len)
    )
    if target.compileTarget in {CompileCpu, CompileThreads}:
      for tensorId in target.tensors:
        ctx[tensorId] = builder.buildCall2(
          ctx.builtin.tensorSignature(),
          ctx.builtin.tensor,
          [ctx.fn.get_param(0), constNimInt(int(tensorId))],
          cstring($tensorId)
        )
    for it, kernel in target.kernels:
      ctx.kernel = kernel
      ctx.kernelId = KernelId(it + 1)
      kernel.toLlvm(KernelId(it + 1), ctx)
    discard builder.build_ret()
    dispose_builder(builder)
    gpuSources.add(ctx.gpuSources)
  result = (module, gpuSources)

type
  JitBuiltin* = object
    tensor*: pointer
    shape*: pointer
    len*: pointer
    shapeLen*: pointer
    debugIndex*: pointer
    debugScalar*: pointer
    epoch*: pointer
    runThreads*: pointer
    joinThreads*: pointer
    runGpuKernel*: pointer
    setGpuKernelIndex*: pointer
    setGpuKernelTensor*: pointer
  
  Jit* = ref object
    module: ModuleRef
    engine: ExecutionEngineRef
    builtin: JitBuiltin
    gpuContext: GpuContext

proc finalize*(jit: Jit) =
  if not jit.engine.isNil:
    dispose_execution_engine(jit.engine)

proc newJit*(module: ModuleRef, builtin: JitBuiltin): Jit =
  new(result, finalizer=finalize)
  result.module = module
  result.builtin = builtin
  
  link_in_mcjit()
  initialize_native_target()
  initialize_native_asm_printer()
  
  let reg = get_global_pass_registry()
  initialize_transform_utils(reg)
  initialize_scalar_opts(reg)
  initialize_obj_carc_opts(reg)
  initialize_vectorization(reg)
  initialize_inst_combine(reg)
  initialize_aggressive_inst_combiner(reg)
  initialize_ipo(reg)
  initialize_analysis(reg)
  initialize_ipa(reg)
  initialize_code_gen(reg)
  
  var err: cstring
  defer:
    if not err.isNil:
      dispose_message(err)
  
  let 
    triple = get_default_target_triple()
    targetFeatures = get_host_cpu_features()
    targetCpu = get_host_cpu_name()
  module.set_target(triple)
  var target: TargetRef = nil
  if get_target_from_triple(triple, target.addr, err.addr) != 0:
    raise JitError(msg: $err)
  let machine = create_target_machine(
    target, triple, targetCpu, targetFeatures,
    OptAggressive, RelocDefault, CodeModelJitDefault
  )
  module.set_module_data_layout(machine.create_target_data_layout())
  
  if module.verify_module(AbortProcessAction, err.addr) != 0:
    raise JitError(msg: $err)
  else:
    dispose_message(err)
    err = nil
  
  let
    opts = create_pass_builder_options()
    passErr = module.run_passes("default<O3>", machine, opts)
  if not passErr.isNil:
    let
      msg = passErr.get_error_message()
      str = $msg
    dispose_error_message(msg)
    raise JitError(msg: str)
  dispose_pass_builder_options(opts)
  
  if create_jit_compiler_for_module(result.engine.addr, module, 3, err.addr) != 0:
    raise JitError(msg: $err)
  
  for name, value in builtin.fieldPairs:
    if value.isNil:
      raise JitError(msg: "Builtin " & name & " is nil")
    let fn = result.module.get_named_function(cstring(name))
    if not fn.isNil:
      result.engine.add_global_mapping(fn, value)

proc getProc*[T: proc](jit: Jit, name: string): T =
  result = cast[T](get_function_address(jit.engine, cstring(name)))

proc saveBitcode*(jit: Jit, path: string) =
  jit.module.saveBitcode(path)
