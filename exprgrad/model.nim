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
    state_location: set[CompileTarget]
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
  program.make_tensor_lookups()
  program.dead_code_elim()
  program.fold_linear_indices()
  program.deduplicate_reads()
  program.infer_shape_constraints()
  program.generate()
  program.dead_kernel_elim()
  program.infer_loop_bounds()
  program.identify_independent()
  program.dead_kernel_elim()
  program.collect_tensors()
  program.sort_shape_constraints()
  program.infer_static_shapes()
  program.infer_types()
  program.reorder_loops()
  program.choose_parallel()
  program.fuse_loops()
  program.tile_loops()
  program.infer_cache_sizes()
  program.cache_tensors()
  program.inline_tensor_ops()
  program.inline_static_shapes()
  program.unfold_loop_bounds()
  program.inline_conditions()
  program.inline_loops()
  program.lift_invariants()
  program.collect_closures()
  program.infer_types()
  program.validate()

{.push cdecl.}
proc builtin_tensor[T](model: ModelPtr[T], id: int): ptr UncheckedArray[T] =
  result = model[].cpu.tensors[TensorId(id)].data_ptr

proc builtin_shape[T](model: ModelPtr[T], id, dim: int): int =
  let shape = model[].shapes[TensorId(id)]
  var index = dim
  if index < 0:
    index += shape.len
  result = shape[index]

proc builtin_len[T](model: ModelPtr[T], id: int): int =
  result = model[].shapes[TensorId(id)].prod()

proc builtin_shape_len[T](model: ModelPtr[T], id: int): int =
  result = model[].shapes[TensorId(id)].len

proc builtin_debug_index[T](model: ModelPtr[T], value: int) =
  echo value

proc builtin_debug_scalar[T](model: ModelPtr[T], value: T) =
  echo value

proc builtin_epoch[T](model: ModelPtr[T]): int =
  result = model.epoch
{.pop.}

when TARGET_SUPPORTS_THREADS:
  import runtimes/threadpools
  
  {.push cdecl.}
  proc builtin_run_threads[T](model: ModelPtr[T],
                            start, stop: int,
                            data: pointer,
                            fn: TaskProc) =
    let size = stop - start
    var offset = start
    for thread in 0..<thread_pool.len:
      var thread_size = size div thread_pool.len
      if thread < size mod thread_pool.len:
        thread_size += 1
      thread_pool.enqueue(thread, Task(
        fn: fn, data: data, model: model,
        a: offset, b: offset + thread_size
      ))
      offset += thread_size
    assert offset == stop
  
  proc builtin_join_threads[T](model: ModelPtr[T]) =
    thread_pool.join()
  {.pop.}
else:
  type TaskProc = proc(model: pointer, a, b: int, data: pointer) {.cdecl, gcsafe.}
  
  {.push cdecl.}
  proc builtin_run_threads[T](model: ModelPtr[T],
                              start, stop: int,
                              data: pointer,
                              fn: TaskProc) = discard
  proc builtin_join_threads[T](model: ModelPtr[T]) = discard
  {.pop.}

when TARGET_SUPPORTS_GPU:
  {.push cdecl.}
  proc builtin_run_gpu_kernel[T](model: ModelPtr[T],
                                 kernel_id: int,
                                 work_dims: int,
                                 global_size: ptr UncheckedArray[int],
                                 local_size: ptr UncheckedArray[int]) =
    let kernel = model.gpu.kernels[kernel_id - 1]
    var group_size: seq[int] = @[]
    for it in 0..<work_dims:
      var size = global_size[it] div local_size[it]
      if global_size[it] mod local_size[it] != 0:
        size += 1
      group_size.add(size)
    kernel.run(
      group_size,
      to_open_array(local_size, 0, work_dims - 1)
    )
  
  proc builtin_set_gpu_kernel_index[T](model: ModelPtr[T], kernel_id, index: int, value: int) =
    discard model.gpu.kernels[kernel_id - 1].arg(index, value)
  
  proc builtin_set_gpu_kernel_tensor[T](model: ModelPtr[T], kernel_id, index: int, tensor: int) =
    discard model.gpu.kernels[kernel_id - 1].arg(index, model.gpu.tensors[tensor - 1].buffer)
  {.pop.}
else:
  {.push cdecl.}
  proc builtin_run_gpu_kernel[T](model: ModelPtr[T],
                                 kernel_id: int,
                                 work_dims: int,
                                 global_size: ptr UncheckedArray[int],
                                 local_size: ptr UncheckedArray[int]) =
    discard
  
  proc builtin_set_gpu_kernel_index[T](model: ModelPtr[T], kernel_id, index: int, value: int) =
    discard
  
  proc builtin_set_gpu_kernel_tensor[T](model: ModelPtr[T], kernel_id, index: int, tensor: int) =
    discard
  {.pop.}

proc init_builtin[T](): JitBuiltin =
  result = JitBuiltin(
    tensor: builtin_tensor[T],
    shape: builtin_shape[T],
    len: builtin_len[T],
    shape_len: builtin_shape_len[T],
    debug_index: builtin_debug_index[T],
    debug_scalar: builtin_debug_scalar[T],
    epoch: builtin_epoch[T],
    run_threads: builtin_run_threads[T],
    join_threads: builtin_join_threads[T],
    run_gpu_kernel: builtin_run_gpu_kernel[T],
    set_gpu_kernel_index: builtin_set_gpu_kernel_index[T],
    set_gpu_kernel_tensor: builtin_set_gpu_kernel_tensor[T]
  )

proc new_cpu_model[T](program: Program): CpuModel[T] =
  result = CpuModel[T]()
  result.tensors = new_seq[Tensor[T]](program.tensors.len)

proc new_gpu_model[T](program: Program, ctx: GpuContext, sources: seq[GpuKernelSource]): GpuModel[T] =
  result = GpuModel[T](ctx: ctx)
  result.tensors = new_seq[GpuTensor[T]](program.tensors.len)
  for source in sources:
    result.kernels.add(ctx.compile(source))

proc new_model[T](source: Program,
                  program: Program,
                  params: Table[TensorId, Tensor[T]],
                  caches: Table[TensorId, Tensor[T]],
                  gpu_ctx: GpuContext): Model[T] =
  result = Model[T](source: source, program: program, params: params, caches: caches)
  result.state_location = {CompileCpu, CompileThreads}
  result.shapes = new_seq[seq[int]](program.tensors.len)
  
  let
    builtin = init_builtin[T]()
    (module, gpu_sources) = program.to_llvm()
  result.jit = new_jit(module, builtin)
  result.cpu = new_cpu_model[T](program)
  if not gpu_ctx.is_nil:
    result.gpu = new_gpu_model[T](program, gpu_ctx, gpu_sources)

proc new_model*[T](source: Program, gpu_ctx: GpuContext): Model[T] =
  bind `==`
  
  let program = source.clone()
  program.compile()
  
  var
    params = init_table[TensorId, Tensor[T]]()
    caches = init_table[TensorId, Tensor[T]]()
  for it, tensor_def in program.tensors:
    let tensor_id = TensorId(it + 1)
    case tensor_def.kind:
      of TensorParam:
        params[tensor_id] = new_rand_tensor[T](tensor_def.shape,
          T(tensor_def.init_range.a)..T(tensor_def.init_range.b)
        )
      of TensorCache:
        caches[tensor_id] = new_tensor[T](tensor_def.shape)
      else: discard
  result = new_model(source, program, params, caches, gpu_ctx)

template to_scalar_type(T: typedesc): ScalarType =
  when T is float32:
    Scalar32
  elif T is float64:
    Scalar64
  else:
    raise new_exception(ValueError, $T & " is not a valid scalar type")
    Scalar32

proc emit_ir*[T](model: Model[T]): string =
  bind irprint.`$`
  result = $model.program

proc save_llvm*[T](model: Model[T], path: string) =
  bind llvmgen.save_bitcode
  save_bitcode(model.jit, path)

proc compile*[T](graphs: varargs[Fun], gpu: GpuContext = nil): Model[T] =
  let source = graphs.to_program()
  source.scalar_type = to_scalar_type(T)
  result = new_model[T](source, gpu)

proc alloc_shapes[T](cpu: CpuModel[T], model: Model[T], target: Target, shapes: Table[ir.TensorId, seq[int]]) =
  for tensor_id, shape in shapes.pairs:
    let
      tensor_def = model.program.tensors[tensor_id]
      required = tensor_id in target.tensors
    
    case tensor_def.kind:
      of TensorInput: discard
      of TensorParam: cpu.tensors[tensor_id] = model.params[tensor_id]
      of TensorCache: cpu.tensors[tensor_id] = model.caches[tensor_id]
      of TensorRandom:
        if required:
          if cpu.tensors[tensor_id].is_nil:
            cpu.tensors[tensor_id] = alloc_tensor[T]()
          cpu.tensors[tensor_id].alloc_shape(shape, fill_zero=false)
          cpu.tensors[tensor_id].fill_rand(
            T(tensor_def.random_range.a)..
            T(tensor_def.random_range.b)
          )
      of TensorResult:
        if required:
          if cpu.tensors[tensor_id].is_nil:
            cpu.tensors[tensor_id] = new_tensor[T](shape)
          else:
            cpu.tensors[tensor_id].alloc_shape(shape, fill_zero=true)

proc alloc_shapes[T](gpu: GpuModel[T], model: Model[T], target: Target, shapes: Table[ir.TensorId, seq[int]]) =
  for tensor_id, shape in shapes.pairs:
    let tensor_def = model.program.tensors[tensor_id]
    if tensor_id notin target.tensors:
      continue
    
    case tensor_def.kind:
      of TensorInput, TensorParam, TensorCache: discard
      of TensorRandom:
        if gpu.tensors[tensor_id].is_nil or gpu.tensors[tensor_id].shape != shape:
          gpu.tensors[tensor_id] = alloc_tensor[T](gpu.ctx, shape)
        let slice = T(tensor_def.random_range.a)..T(tensor_def.random_range.b)
        gpu.tensors[tensor_id].write(new_rand_tensor(shape, slice))
      of TensorResult:
        if gpu.tensors[tensor_id].is_nil or gpu.tensors[tensor_id].shape != shape:
          gpu.tensors[tensor_id] = alloc_tensor[T](gpu.ctx, shape)
        gpu.tensors[tensor_id].fill(T(0))

iterator state_tensors[T](model: Model[T]): (TensorId, Tensor[T]) =
  for id, tensor in pairs(model.params):
    yield (id, tensor)
  for id, tensor in pairs(model.caches):
    yield (id, tensor)

proc flush_state_tensors[T](model: Model[T], to: CompileTarget) =
  if to in model.state_location:
    return
  case to:
    of CompileCpu, CompileThreads:
      if CompileCpu in model.state_location or
         CompileThreads in model.state_location:
        return
      assert CompileGpu in model.state_location
      for id, tensor in model.state_tensors:
        model.gpu.tensors[id].read_into(tensor)
    of CompileGpu:
      assert CompileCpu in model.state_location or
             CompileThreads in model.state_location
      for id, tensor in model.state_tensors:
        if model.gpu.tensors[id].is_nil or
           model.gpu.tensors[id].shape != tensor.shape:
          model.gpu.tensors[id] = alloc_tensor[T](model.gpu.ctx, tensor.shape)
        model.gpu.tensors[id].write(tensor)
  model.state_location.incl(to)

proc alloc_shapes[T](model: Model[T], target: Target, shapes: Table[ir.TensorId, seq[int]]) =
  model.flush_state_tensors(target.compile_target)
  for id, shape in pairs(shapes):
    model.shapes[id] = shape
  case target.compile_target:
    of CompileCpu, CompileThreads:
      model.cpu.alloc_shapes(model, target, shapes)
    of CompileGpu:
      model.gpu.alloc_shapes(model, target, shapes)

proc write_input[T](model: Model[T], to: CompileTarget, name: string, tensor: Tensor[T]) =
  if name notin model.program.inputs:
    raise new_exception(ValueError, name & " is not an input to the model")
  let tensor_id = model.program.inputs[name]
  case to:
    of CompileCpu, CompileThreads:
      model.cpu.tensors[tensor_id] = tensor
    of CompileGpu:
      if model.gpu.tensors[tensor_id].is_nil or
         model.gpu.tensors[tensor_id].shape != tensor.shape:
        model.gpu.tensors[tensor_id] = alloc_tensor[T](model.gpu.ctx, tensor.shape)
      model.gpu.tensors[tensor_id].write(tensor)

proc read_output[T](model: Model[T], target: CompileTarget, tensor: TensorId): Tensor[T] =
  case target:
    of CompileCpu, CompileThreads:
      result = model.cpu.tensors[tensor]
      model.cpu.tensors[tensor] = nil
    of CompileGpu:
      result = model.gpu.tensors[tensor].read()

proc zero_result_tensor[T](model: Model[T], target: CompileTarget, tensor_id: TensorId) =
  case target:
    of CompileCpu, CompileThreads:
      model.cpu.tensors[tensor_id].fill_zero()
    of CompileGpu:
      model.gpu.tensors[tensor_id].fill(T(0))

proc call_jit[T](model: Model[T], target_name: string): Tensor[T] =
  let fn = get_proc[proc (model: ModelPtr[T]) {.cdecl.}](model.jit, "target_" & target_name)
  fn(model[].addr)
  let target = model.program.targets[target_name]
  if int(target.output) != 0:
    result = model.read_output(target.compile_target, target.output)

proc call*[T](model: Model[T],
              target_name: string,
              args: openArray[(string, Tensor[T])] = []): Tensor[T] =
  let target = model.program.targets[target_name]
  var input_shapes = new_seq[(TensorId, seq[int])](args.len)
  for it, (name, tensor) in args:
    model.write_input(target.compile_target, name, tensor)
    input_shapes[it] = (model.program.inputs[name], tensor.shape)
  let shapes = model.program.infer_shapes(target_name, input_shapes)
  model.alloc_shapes(target, shapes)
  result = model.call_jit(target_name)

proc apply*[T](model: Model[T],
               target: string,
               args: openArray[(string, Tensor[T])] = []) =
  discard model.call(target, args)

proc fit*[T](model: Model[T],
             target_name: string,
             args: openArray[(string, Tensor[T])],
             batch_size: int = 32,
             log_status: bool = true) =
  if args.len == 0:
    raise new_exception(ValueError, "Model.fit requires at least one input tensor. Use Model.apply instead if the target has zero inputs.")
  let
    target = model.program.targets[target_name]
    batch_count = args[0][1].shape[0] div batch_size
  
  var input_shapes = new_seq[(TensorId, seq[int])](args.len)
  for it, (name, arg) in args:
    if name notin model.program.inputs:
      raise new_exception(ValueError, name & " is not an input to the model")
    input_shapes[it] = (model.program.inputs[name], @[batch_size] & arg.shape[1..^1])
  
  let shapes = model.program.infer_shapes(target_name, input_shapes)
  model.alloc_shapes(target, shapes)
  
  model.epoch += 1
  for batch_id in 0..<batch_count:
    if log_status:
      stdout.write($batch_id & "/" & $batch_count & "\r")
      stdout.flush_file()
    let offset = batch_size * batch_id
    for it, (name, arg) in args:
      model.write_input(target.compile_target, name, arg.view_first(offset, batch_size))
    
    discard model.call_jit(target_name)
    
    for tensor in target.tensors.items:
      if model.program.tensors[tensor].kind == TensorResult:
        model.zero_result_tensor(target.compile_target, tensor)
  
  if log_status:
    stdout.write($batch_count & "/" & $batch_count & "\r")
    stdout.write("\n")
    stdout.flush_file()
