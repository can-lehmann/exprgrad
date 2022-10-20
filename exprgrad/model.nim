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

# Simple API used to compile and run models in exprgrad

import std/[tables, sets, math]
import ir, parser, passes, tensors, llvmgen, irprint, runtimes/gpu

type
  GpuModel[T] = ref object
    ctx: GpuContext
    kernels: seq[GpuKernel]
    tensors: seq[GpuTensor[T]]
  
  CpuModel[T] = ref object
    tensors: seq[Tensor[T]]
  
  ModelObj[T] = object
    program*: Program
    source*: Program
    jit: Jit
    stateLocation: set[CompileTarget]
    cpu: CpuModel[T]
    gpu: GpuModel[T]
    shapes: seq[seq[int]]
    params*: Table[TensorId, Tensor[T]]
    caches*: Table[TensorId, Tensor[T]]
    epoch*: int
  
  ModelPtr[T] = ptr ModelObj[T]
  Model*[T] = ref ModelObj[T]

proc `$`(kernel: Kernel): string = $(kernel[])

proc compile*(program: Program) =
  program.makeTensorLookups()
  program.deadCodeElim()
  program.foldLinearIndices()
  program.deduplicateReads()
  program.inferShapeConstraints()
  program.generate()
  program.deadKernelElim()
  program.inferLoopBounds()
  program.identifyIndependent()
  program.deadKernelElim()
  program.collectTensors()
  program.sortShapeConstraints()
  program.inferStaticShapes()
  program.inferTypes()
  program.reorderLoops()
  program.chooseParallel()
  program.fuseLoops()
  program.tileLoops()
  program.inferCacheSizes()
  program.cacheTensors()
  program.inlineTensorOps()
  program.inlineStaticShapes()
  program.unfoldLoopBounds()
  program.inlineConditions()
  program.inlineLoops()
  program.liftInvariants()
  program.collectClosures()
  program.inferTypes()
  program.validate()

{.push cdecl.}
proc builtinTensor[T](model: ModelPtr[T], id: int): ptr UncheckedArray[T] =
  result = model[].cpu.tensors[TensorId(id)].dataPtr

proc builtinShape[T](model: ModelPtr[T], id, dim: int): int =
  let shape = model[].shapes[TensorId(id)]
  var index = dim
  if index < 0:
    index += shape.len
  result = shape[index]

proc builtinLen[T](model: ModelPtr[T], id: int): int =
  result = model[].shapes[TensorId(id)].prod()

proc builtinShapeLen[T](model: ModelPtr[T], id: int): int =
  result = model[].shapes[TensorId(id)].len

proc builtinDebugIndex[T](model: ModelPtr[T], value: int) =
  echo value

proc builtinDebugScalar[T](model: ModelPtr[T], value: T) =
  echo value

proc builtinEpoch[T](model: ModelPtr[T]): int =
  result = model.epoch
{.pop.}

when TARGET_SUPPORTS_THREADS:
  import runtimes/threadpools
  
  {.push cdecl.}
  proc builtinRunThreads[T](model: ModelPtr[T],
                            start, stop, minCount: int,
                            data: pointer,
                            fn: TaskProc) =
    let size = stop - start
    var
      threadCount = threadPool.len
      offset = start
    if minCount > 0:
      threadCount = threadCount.min(size div minCount).max(1)
    if threadCount == 1 and not defined(alwaysUseThreadpool):
      fn(model, start, stop, data)
    else:
      for thread in 0..<threadCount:
        var threadSize = size div threadCount
        if thread < size mod threadCount:
          threadSize += 1
        threadPool.enqueue(thread, Task(
          fn: fn, data: data, model: model,
          a: offset, b: offset + threadSize
        ))
        offset += threadSize
      assert offset == stop
  
  proc builtinJoinThreads[T](model: ModelPtr[T]) =
    threadPool.join()
  {.pop.}
else:
  type TaskProc = proc(model: pointer, a, b: int, data: pointer) {.cdecl, gcsafe.}
  
  {.push cdecl.}
  proc builtinRunThreads[T](model: ModelPtr[T],
                            start, stop, minCount: int,
                            data: pointer,
                            fn: TaskProc) = discard
  proc builtinJoinThreads[T](model: ModelPtr[T]) = discard
  {.pop.}

when TARGET_SUPPORTS_GPU:
  {.push cdecl.}
  proc builtinRunGpuKernel[T](model: ModelPtr[T],
                              kernelId: int,
                              workDims: int,
                              globalSize: ptr UncheckedArray[int],
                              localSize: ptr UncheckedArray[int]) =
    let kernel = model.gpu.kernels[kernelId - 1]
    var groupSize: seq[int] = @[]
    for it in 0..<workDims:
      var size = globalSize[it] div localSize[it]
      if globalSize[it] mod localSize[it] != 0:
        size += 1
      groupSize.add(size)
    kernel.run(
      groupSize,
      toOpenArray(localSize, 0, workDims - 1)
    )
  
  proc builtinSetGpuKernelIndex[T](model: ModelPtr[T], kernelId, index: int, value: int) =
    discard model.gpu.kernels[kernelId - 1].arg(index, value)
  
  proc builtinSetGpuKernelTensor[T](model: ModelPtr[T], kernelId, index: int, tensor: int) =
    discard model.gpu.kernels[kernelId - 1].arg(index, model.gpu.tensors[tensor - 1].buffer)
  {.pop.}
else:
  {.push cdecl.}
  proc builtinRunGpuKernel[T](model: ModelPtr[T],
                              kernelId: int,
                              workDims: int,
                              globalSize: ptr UncheckedArray[int],
                              localSize: ptr UncheckedArray[int]) =
    discard
  
  proc builtinSetGpuKernelIndex[T](model: ModelPtr[T], kernelId, index: int, value: int) =
    discard
  
  proc builtinSetGpuKernelTensor[T](model: ModelPtr[T], kernelId, index: int, tensor: int) =
    discard
  {.pop.}

proc initBuiltin[T](): JitBuiltin =
  result = JitBuiltin(
    tensor: builtinTensor[T],
    shape: builtinShape[T],
    len: builtinLen[T],
    shapeLen: builtinShapeLen[T],
    debugIndex: builtinDebugIndex[T],
    debugScalar: builtinDebugScalar[T],
    epoch: builtinEpoch[T],
    runThreads: builtinRunThreads[T],
    joinThreads: builtinJoinThreads[T],
    runGpuKernel: builtinRunGpuKernel[T],
    setGpuKernelIndex: builtinSetGpuKernelIndex[T],
    setGpuKernelTensor: builtinSetGpuKernelTensor[T]
  )

proc newCpuModel[T](program: Program): CpuModel[T] =
  result = CpuModel[T]()
  result.tensors = newSeq[Tensor[T]](program.tensors.len)

proc newGpuModel[T](program: Program, ctx: GpuContext, sources: seq[GpuKernelSource]): GpuModel[T] =
  result = GpuModel[T](ctx: ctx)
  result.tensors = newSeq[GpuTensor[T]](program.tensors.len)
  for source in sources:
    result.kernels.add(ctx.compile(source))

proc newModel[T](source: Program,
                  program: Program,
                  params: Table[TensorId, Tensor[T]],
                  caches: Table[TensorId, Tensor[T]],
                  gpuCtx: GpuContext): Model[T] =
  result = Model[T](source: source, program: program, params: params, caches: caches)
  result.stateLocation = {CompileCpu, CompileThreads}
  result.shapes = newSeq[seq[int]](program.tensors.len)
  
  let
    builtin = initBuiltin[T]()
    (module, gpuSources) = program.toLlvm()
  result.jit = newJit(module, builtin)
  result.cpu = newCpuModel[T](program)
  if not gpuCtx.isNil:
    result.gpu = newGpuModel[T](program, gpuCtx, gpuSources)

proc newModel*[T](source: Program, gpuCtx: GpuContext): Model[T] =
  bind `==`, newTensor
  
  let program = source.clone()
  program.compile()
  
  var
    params = initTable[TensorId, Tensor[T]]()
    caches = initTable[TensorId, Tensor[T]]()
  for it, tensorDef in program.tensors:
    let tensorId = TensorId(it + 1)
    case tensorDef.kind:
      of TensorParam:
        params[tensorId] = newRandTensor[T](tensorDef.shape,
          T(tensorDef.initRange.a)..T(tensorDef.initRange.b)
        )
      of TensorCache:
        caches[tensorId] = newTensor[T](tensorDef.shape)
      else: discard
  result = newModel(source, program, params, caches, gpuCtx)

template toScalarType(T: typedesc): ScalarType =
  when T is float32:
    Scalar32
  elif T is float64:
    Scalar64
  else:
    raise newException(ValueError, $T & " is not a valid scalar type")
    Scalar32

proc emitIr*[T](model: Model[T]): string =
  bind irprint.`$`
  result = $model.program

proc saveLlvm*[T](model: Model[T], path: string) =
  bind llvmgen.saveBitcode
  saveBitcode(model.jit, path)

proc compile*[T](graphs: varargs[Fun], gpu: GpuContext = nil): Model[T] =
  let source = graphs.toProgram()
  source.scalarType = toScalarType(T)
  result = newModel[T](source, gpu)

proc allocShapes[T](cpu: CpuModel[T], model: Model[T], target: Target, shapes: Table[ir.TensorId, seq[int]]) =
  bind newTensor
  for tensorId, shape in shapes.pairs:
    let
      tensorDef = model.program.tensors[tensorId]
      required = tensorId in target.tensors
    
    case tensorDef.kind:
      of TensorInput: discard
      of TensorParam: cpu.tensors[tensorId] = model.params[tensorId]
      of TensorCache: cpu.tensors[tensorId] = model.caches[tensorId]
      of TensorRandom:
        if required:
          if cpu.tensors[tensorId].isNil:
            cpu.tensors[tensorId] = allocTensor[T]()
          cpu.tensors[tensorId].allocShape(shape, fillZero=false)
          cpu.tensors[tensorId].fillRand(
            T(tensorDef.randomRange.a)..
            T(tensorDef.randomRange.b)
          )
      of TensorResult:
        if required:
          if cpu.tensors[tensorId].isNil:
            cpu.tensors[tensorId] = newTensor[T](shape)
          else:
            cpu.tensors[tensorId].allocShape(shape, fillZero=true)

proc allocShapes[T](gpu: GpuModel[T], model: Model[T], target: Target, shapes: Table[ir.TensorId, seq[int]]) =
  for tensorId, shape in shapes.pairs:
    let tensorDef = model.program.tensors[tensorId]
    if tensorId notin target.tensors:
      continue
    
    case tensorDef.kind:
      of TensorInput, TensorParam, TensorCache: discard
      of TensorRandom:
        if gpu.tensors[tensorId].isNil or gpu.tensors[tensorId].shape != shape:
          gpu.tensors[tensorId] = allocTensor[T](gpu.ctx, shape)
        let slice = T(tensorDef.randomRange.a)..T(tensorDef.randomRange.b)
        gpu.tensors[tensorId].write(newRandTensor(shape, slice))
      of TensorResult:
        if gpu.tensors[tensorId].isNil or gpu.tensors[tensorId].shape != shape:
          gpu.tensors[tensorId] = allocTensor[T](gpu.ctx, shape)
        gpu.tensors[tensorId].fill(T(0))

iterator stateTensors[T](model: Model[T]): (TensorId, Tensor[T]) =
  for id, tensor in pairs(model.params):
    yield (id, tensor)
  for id, tensor in pairs(model.caches):
    yield (id, tensor)

proc flushStateTensors[T](model: Model[T], to: CompileTarget) =
  if to in model.stateLocation:
    return
  case to:
    of CompileCpu, CompileThreads:
      if CompileCpu in model.stateLocation or
         CompileThreads in model.stateLocation:
        return
      assert CompileGpu in model.stateLocation
      for id, tensor in model.stateTensors:
        model.gpu.tensors[id].readInto(tensor)
    of CompileGpu:
      assert CompileCpu in model.stateLocation or
             CompileThreads in model.stateLocation
      for id, tensor in model.stateTensors:
        if model.gpu.tensors[id].isNil or
           model.gpu.tensors[id].shape != tensor.shape:
          model.gpu.tensors[id] = allocTensor[T](model.gpu.ctx, tensor.shape)
        model.gpu.tensors[id].write(tensor)
  model.stateLocation.incl(to)

proc allocShapes[T](model: Model[T], target: Target, shapes: Table[ir.TensorId, seq[int]]) =
  model.flushStateTensors(target.compileTarget)
  for id, shape in pairs(shapes):
    model.shapes[id] = shape
  case target.compileTarget:
    of CompileCpu, CompileThreads:
      model.cpu.allocShapes(model, target, shapes)
    of CompileGpu:
      model.gpu.allocShapes(model, target, shapes)

proc writeInput[T](model: Model[T], to: CompileTarget, name: string, tensor: Tensor[T]) =
  if name notin model.program.inputs:
    raise RuntimeError(msg: name & " is not an input to the model")
  let tensorId = model.program.inputs[name]
  case to:
    of CompileCpu, CompileThreads:
      model.cpu.tensors[tensorId] = tensor
    of CompileGpu:
      if model.gpu.tensors[tensorId].isNil or
         model.gpu.tensors[tensorId].shape != tensor.shape:
        model.gpu.tensors[tensorId] = allocTensor[T](model.gpu.ctx, tensor.shape)
      model.gpu.tensors[tensorId].write(tensor)

proc readOutput[T](model: Model[T], target: CompileTarget, tensor: TensorId): Tensor[T] =
  case target:
    of CompileCpu, CompileThreads:
      result = model.cpu.tensors[tensor]
      model.cpu.tensors[tensor] = nil
    of CompileGpu:
      result = model.gpu.tensors[tensor].read()

proc zeroResultTensor[T](model: Model[T], target: CompileTarget, tensorId: TensorId) =
  case target:
    of CompileCpu, CompileThreads:
      model.cpu.tensors[tensorId].fillZero()
    of CompileGpu:
      model.gpu.tensors[tensorId].fill(T(0))

proc callJit[T](model: Model[T], targetName: string): Tensor[T] =
  let fn = getProc[proc (model: ModelPtr[T]) {.cdecl.}](model.jit, "target_" & targetName)
  fn(model[].addr)
  let target = model.program.targets[targetName]
  if int(target.output) != 0:
    result = model.readOutput(target.compileTarget, target.output)

proc call*[T](model: Model[T],
              targetName: string,
              args: openArray[(string, Tensor[T])] = []): Tensor[T] =
  if targetName notin model.program.targets:
    raise RuntimeError(msg: targetName & " is not a target of the model")
  let target = model.program.targets[targetName]
  
  var inputShapes = newSeq[(TensorId, seq[int])](args.len)
  for it, (name, tensor) in args:
    model.writeInput(target.compileTarget, name, tensor)
    inputShapes[it] = (model.program.inputs[name], tensor.shape)
  
  let shapes = model.program.inferShapes(targetName, inputShapes)
  model.allocShapes(target, shapes)
  result = model.callJit(targetName)

proc apply*[T](model: Model[T],
               target: string,
               args: openArray[(string, Tensor[T])] = []) =
  discard model.call(target, args)

proc fit*[T](model: Model[T],
             targetName: string,
             args: openArray[(string, Tensor[T])],
             batchSize: int = 32,
             logStatus: bool = true) =
  if args.len == 0:
    raise RuntimeError(msg: "Model.fit requires at least one input tensor. Use Model.apply instead if the target has zero inputs.")
  if targetName notin model.program.targets:
    raise RuntimeError(msg: targetName & " is not a target of the model")
  
  let
    target = model.program.targets[targetName]
    batchCount = args[0][1].shape[0] div batchSize
  
  var inputShapes = newSeq[(TensorId, seq[int])](args.len)
  for it, (name, arg) in args:
    if name notin model.program.inputs:
      raise RuntimeError(msg: name & " is not an input to the model")
    inputShapes[it] = (model.program.inputs[name], @[batchSize] & arg.shape[1..^1])
  
  let shapes = model.program.inferShapes(targetName, inputShapes)
  model.allocShapes(target, shapes)
  
  model.epoch += 1
  for batchId in 0..<batchCount:
    if logStatus:
      stdout.write($batchId & "/" & $batchCount & "\r")
      stdout.flushFile()
    let offset = batchSize * batchId
    for it, (name, arg) in args:
      model.writeInput(target.compileTarget, name, arg.viewFirst(offset, batchSize))
    
    discard model.callJit(targetName)
    
    for tensor in target.tensors.items:
      if model.program.tensors[tensor].kind == TensorResult:
        model.zeroResultTensor(target.compileTarget, tensor)
  
  if logStatus:
    stdout.write($batchCount & "/" & $batchCount & "\r")
    stdout.write("\n")
    stdout.flushFile()
