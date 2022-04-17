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

import std/[tables, sets]
import ir, parser, passes, tensors, llvmgen, irprint, cgen

type
  ModelObj[T] = object
    program*: Program
    jit: Jit
    params*: Table[TensorId, Tensor[T]]
    caches*: Table[TensorId, Tensor[T]]
    tensors: seq[Tensor[T]]
    epoch*: int
  
  ModelPtr[T] = ptr ModelObj[T]
  Model*[T] = ref ModelObj[T]

proc `$`(kernel: Kernel): string = $(kernel[])

{.push cdecl.}
proc builtin_tensor[T](model: ModelPtr[T], id: int): ptr UncheckedArray[T] =
  result = model[].tensors[TensorId(id)].data_ptr

proc builtin_shape[T](model: ModelPtr[T], id, dim: int): int =
  let tensor = model[].tensors[TensorId(id)]
  if dim < 0:
    result = tensor.shape[tensor.shape.len + dim]
  else:
    result = tensor.shape[dim]

proc builtin_len[T](model: ModelPtr[T], id: int): int =
  result = model[].tensors[TensorId(id)].len

proc builtin_shape_len[T](model: ModelPtr[T], id: int): int =
  result = model[].tensors[TensorId(id)].shape.len

proc builtin_debug_index[T](model: ModelPtr[T], value: int) =
  echo value

proc builtin_debug_scalar[T](model: ModelPtr[T], value: T) =
  echo value

proc builtin_epoch[T](model: ModelPtr[T]): int =
  result = model.epoch
{.pop.}

type TaskProc = proc(model: pointer, a, b: int, data: pointer) {.cdecl, gcsafe.}

when TARGET_SUPPORTS_THREADS:
  import osproc
  
  type
    Task = object
      fn: TaskProc
      model: pointer
      data: pointer
      a: int
      b: int
  
    ThreadData = object
      tasks: ptr Channel[Task]
      done: ptr Channel[bool]
    
    ThreadPool = object
      threads: seq[Thread[ThreadData]]
      data: seq[ThreadData]
      open: seq[int]
  
  proc thread_handler(data: ThreadData) {.gcsafe.} =
    while true:
      let task = data.tasks[].recv()
      try:
        task.fn(task.model, task.a, task.b, task.data)
      except CatchableError as err:
        data.done[].send(false)
        continue
      data.done[].send(true)
  
  proc alloc_channel[T](): ptr Channel[T] =
    result = cast[ptr Channel[T]](alloc_shared0(sizeof(Channel[T])))
    result[].open()
  
  proc init_thread_pool(): ThreadPool =
    result.threads = new_seq[Thread[ThreadData]](count_processors())
    result.open = new_seq[int](result.threads.len)
    for it, thread in result.threads.mpairs:
      let data = ThreadData(
        tasks: alloc_channel[Task](),
        done: alloc_channel[bool]()
      )
      result.data.add(data)
      create_thread(thread, thread_handler, data)
      thread.pin_to_cpu(it)
  
  proc len(pool: ThreadPool): int = pool.threads.len
  
  proc enqueue(pool: var ThreadPool, thread: int, task: Task) =
    pool.data[thread].tasks[].send(task)
    pool.open[thread] += 1
  
  proc join(pool: var ThreadPool, thread: int) =
    while pool.open[thread] > 0:
      discard pool.data[thread].done[].recv()
      pool.open[thread] -= 1
  
  proc join(pool: var ThreadPool) =
    for it in 0..<pool.len:
      pool.join(it)
  
  var thread_pool = init_thread_pool()
  
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
  {.push cdecl.}
  proc builtin_run_threads[T](model: ModelPtr[T],
                              start, stop: int,
                              data: pointer,
                              fn: TaskProc) = discard
  proc builtin_join_threads[T](model: ModelPtr[T]) = discard
  {.pop.}

proc new_model*[T](program: Program,
                   params: Table[TensorId, Tensor[T]],
                   caches: Table[TensorId, Tensor[T]]): Model[T] =
  let builtin = JitBuiltin(
    tensor: builtin_tensor[T],
    shape: builtin_shape[T],
    len: builtin_len[T],
    shape_len: builtin_shape_len[T],
    debug_index: builtin_debug_index[T],
    debug_scalar: builtin_debug_scalar[T],
    epoch: builtin_epoch[T],
    run_threads: builtin_run_threads[T],
    join_threads: builtin_join_threads[T]
  )
  result = Model[T](
    program: program,
    params: params,
    caches: caches,
    jit: new_jit(program.to_llvm(), builtin),
    tensors: new_seq[Tensor[T]](program.tensors.len)
  )

proc new_model*[T](program: Program): Model[T] =
  bind `==`
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
  result = new_model(program, params, caches)

template to_scalar_type(T: typedesc): ScalarType =
  when T is float32:
    Scalar32
  elif T is float64:
    Scalar64
  else:
    raise new_exception(ValueError, $T & " is not a valid scalar type")
    Scalar32

proc emit_source*[T](model: Model[T]): string =
  bind cgen.to_c
  result = cgen.to_c(model.program)

proc emit_ir*[T](model: Model[T]): string =
  bind irprint.`$`
  result = $model.program

proc save_llvm*[T](model: Model[T], path: string) =
  bind llvmgen.save_bitcode
  save_bitcode(model.jit, path)

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
  program.inline_tensor_ops()
  program.inline_static_shapes()
  program.unfold_loop_bounds()
  program.inline_conditions()
  program.inline_loops()
  program.lift_invariants()
  program.collect_closures()
  #echo $program.targets["c"]
  program.infer_types()

proc compile*[T](graphs: varargs[Fun]): Model[T] =
  let program = graphs.to_program()
  program.scalar_type = to_scalar_type(T)
  program.compile()
  result = new_model[T](program)

proc alloc_shapes[T](model: Model[T], target: string, shapes: Table[ir.TensorId, seq[int]]) =
  for tensor_id, shape in shapes.pairs:
    let
      tensor_def = model.program.tensors[tensor_id]
      required = tensor_id in model.program.targets[target].tensors
    
    case tensor_def.kind:
      of TensorInput: discard
      of TensorParam: model.tensors[tensor_id] = model.params[tensor_id]
      of TensorCache: model.tensors[tensor_id] = model.caches[tensor_id]
      of TensorRandom:
        if required:
          if model.tensors[tensor_id].is_nil:
            model.tensors[tensor_id] = alloc_tensor[T]()
          model.tensors[tensor_id].alloc_shape(shape, fill_zero=false)
          model.tensors[tensor_id].fill_rand(
            T(tensor_def.random_range.a)..
            T(tensor_def.random_range.b)
          )
      of TensorResult:
        if required:
          if model.tensors[tensor_id].is_nil:
            model.tensors[tensor_id] = new_tensor[T](shape)
          else:
            model.tensors[tensor_id].alloc_shape(shape, fill_zero=true)

proc call_jit[T](model: Model[T], target: string): Tensor[T] =
  let fn = get_proc[proc (model: ModelPtr[T]) {.cdecl.}](model.jit, "target_" & target)
  fn(model[].addr)
  let output = model.program.targets[target].output
  if int(output) != 0:
    result = model.tensors[output]
    model.tensors[output] = nil

proc call*[T](model: Model[T],
              target: string,
              args: openArray[(string, Tensor[T])] = []): Tensor[T] =
  var input_shapes = new_seq[(TensorId, seq[int])](args.len)
  for it, (name, tensor) in args:
    if name notin model.program.inputs:
      raise new_exception(ValueError, name & " is not an input to the model")
    input_shapes[it] = (model.program.inputs[name], tensor.shape)
    model.tensors[model.program.inputs[name]] = tensor
  let shapes = model.program.infer_shapes(target, input_shapes)
  model.alloc_shapes(target, shapes)
  result = model.call_jit(target)

proc apply*[T](model: Model[T],
               target: string,
               args: openArray[(string, Tensor[T])] = []) =
  discard model.call(target, args)

proc fit*[T](model: Model[T],
             target: string,
             args: openArray[(string, Tensor[T])],
             batch_size: int = 32) =
  if args.len == 0:
    raise new_exception(ValueError, "Model.fit requires at least one input tensor. Use Model.apply instead if the target has zero inputs.")
  let batch_count = args[0][1].shape[0] div batch_size
  
  var input_shapes = new_seq[(TensorId, seq[int])](args.len)
  for it, (name, arg) in args:
    if name notin model.program.inputs:
      raise new_exception(ValueError, name & " is not an input to the model")
    input_shapes[it] = (model.program.inputs[name], @[batch_size] & arg.shape[1..^1])
  
  let shapes = model.program.infer_shapes(target, input_shapes)
  model.alloc_shapes(target, shapes)
  
  model.epoch += 1
  for batch_id in 0..<batch_count:
    stdout.write($batch_id & "/" & $batch_count & "\r")
    stdout.flush_file()
    let offset = batch_size * batch_id
    for it, (name, arg) in args:
      let id = model.program.inputs[name]
      model.tensors[id] = arg.view_first(offset, batch_size)
    
    discard model.call_jit(target)
    
    for tensor in model.program.targets[target].tensors.items:
      if model.program.tensors[tensor].kind == TensorResult:
        model.tensors[tensor].fill_zero()
  
  stdout.write($batch_count & "/" & $batch_count & "\r")
  stdout.write("\n")
  stdout.flush_file()
