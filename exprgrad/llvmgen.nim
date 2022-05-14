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

proc to_llvm(scalar_type: ScalarType): TypeRef =
  case scalar_type:
    of Scalar32: result = float_type()
    of Scalar64: result = double_type()

type Builtin = object
  scalar_type: ScalarType
  # Tensor
  tensor: ValueRef
  shape: ValueRef
  len: ValueRef
  shape_len: ValueRef
  # Debug
  debug_index: ValueRef
  debug_scalar: ValueRef
  # Values
  epoch: ValueRef
  # Threads
  run_threads: ValueRef
  join_threads: ValueRef
  # Gpu
  run_gpu_kernel: ValueRef
  set_gpu_kernel_arg: ValueRef
  # Intrinsics
  sin: ValueRef
  cos: ValueRef
  exp: ValueRef
  ln: ValueRef
  sqrt: ValueRef
  pow: ValueRef

proc model_ptr_type(): TypeRef = pointer_type(int8_type(), 0)
proc void_ptr_type(): TypeRef = pointer_type(int8_type(), 0)

proc tensor_signature(builtin: Builtin): TypeRef =
  function_type(pointer_type(builtin.scalar_type.to_llvm(), 0), [
    model_ptr_type(), nim_int_type()
  ])

proc shape_signature(builtin: Builtin): TypeRef =
  function_type(nim_int_type(), [
    model_ptr_type(), nim_int_type(), nim_int_type()
  ])

proc len_signature(builtin: Builtin): TypeRef =
  function_type(nim_int_type(), [model_ptr_type(), nim_int_type()])

proc shape_len_signature(builtin: Builtin): TypeRef =
  function_type(nim_int_type(), [model_ptr_type(), nim_int_type()])

proc debug_index_signature(builtin: Builtin): TypeRef =
  function_type(void_type(), [model_ptr_type(), nim_int_type()])

proc debug_scalar_signature(builtin: Builtin): TypeRef =
  function_type(void_type(), [model_ptr_type(), builtin.scalar_type.to_llvm()])

proc epoch_signature(builtin: Builtin): TypeRef =
  function_type(nim_int_type(), [model_ptr_type()])

proc task_proc_signature(): TypeRef =
  function_type(void_type(), [
    model_ptr_type(), nim_int_type(), nim_int_type(), void_ptr_type()
  ])

proc run_threads_signature(builtin: Builtin): TypeRef =
  function_type(void_type(), [
    model_ptr_type(), nim_int_type(), nim_int_type(), void_ptr_type(),
    task_proc_signature().pointer_type(0)
  ])

proc join_threads_signature(builtin: Builtin): TypeRef =
  function_type(void_type(), [model_ptr_type()])

proc set_gpu_kernel_arg_signature(builtin: Builtin): TypeRef =
  function_type(void_type(), [
    model_ptr_type(), nim_int_type(),
    nim_int_type(), void_ptr_type()
  ])

proc run_gpu_kernel_signature(builtin: Builtin): TypeRef =
  function_type(void_type(), [
    model_ptr_type(), nim_int_type(),
    nim_int_type(),
    nim_int_type().pointer_type(0),
    nim_int_type().pointer_type(0)
  ])

proc scalar_unary_intrinsic_signature(builtin: Builtin): TypeRef =
  function_type(builtin.scalar_type.to_llvm(), [builtin.scalar_type.to_llvm()])

proc scalar_binary_intrinsic_signature(builtin: Builtin): TypeRef =
  function_type(builtin.scalar_type.to_llvm(), [
    builtin.scalar_type.to_llvm(), builtin.scalar_type.to_llvm()
  ])

proc init_builtin(module: ModuleRef, program: Program): Builtin =
  result = Builtin(scalar_type: program.scalar_type)
  result.tensor = module.add_function("tensor", result.tensor_signature())
  result.shape = module.add_function("shape", result.shape_signature())
  result.len = module.add_function("len", result.len_signature())
  result.shape_len = module.add_function("shape_len", result.shape_len_signature())
  result.debug_index = module.add_function("debug_index", result.debug_index_signature())
  result.debug_scalar = module.add_function("debug_scalar", result.debug_scalar_signature())
  result.epoch = module.add_function("epoch", result.epoch_signature())
  result.run_threads = module.add_function("run_threads", result.run_threads_signature())
  result.join_threads = module.add_function("join_threads", result.join_threads_signature())
  result.run_gpu_kernel = module.add_function("run_gpu_kernel", result.run_gpu_kernel_signature())
  result.set_gpu_kernel_arg = module.add_function("set_gpu_kernel_arg", result.set_gpu_kernel_arg_signature())
  
  let type_postfix = [Scalar32: "f32", Scalar64: "f64"][result.scalar_type]
  result.sin = module.add_function(cstring("llvm.sin." & type_postfix), result.scalar_unary_intrinsic_signature())
  result.cos = module.add_function(cstring("llvm.cos." & type_postfix), result.scalar_unary_intrinsic_signature())
  result.exp = module.add_function(cstring("llvm.exp." & type_postfix), result.scalar_unary_intrinsic_signature())
  result.ln = module.add_function(cstring("llvm.log." & type_postfix), result.scalar_unary_intrinsic_signature())
  result.sqrt = module.add_function(cstring("llvm.sqrt." & type_postfix), result.scalar_unary_intrinsic_signature())
  result.pow = module.add_function(cstring("llvm.pow." & type_postfix), result.scalar_binary_intrinsic_signature())

type Context = ref object
  program: Program
  target: string
  kernel: Kernel
  kernel_id: KernelId
  module: ModuleRef
  builder: BuilderRef
  fn: ValueRef
  builtin: Builtin
  tensors: seq[ValueRef]
  regs: seq[ValueRef]
  gpu_sources: seq[GpuKernelSource]

proc `[]`(ctx: Context, reg: RegId): ValueRef = ctx.regs[reg]
proc `[]=`(ctx: Context, reg: RegId, val: ValueRef) = ctx.regs[reg] = val
proc `[]`(ctx: Context, tensor: TensorId): ValueRef = ctx.tensors[tensor]
proc `[]=`(ctx: Context, tensor: TensorId, val: ValueRef) = ctx.tensors[tensor] = val

proc scalar_type(ctx: Context): TypeRef =
  ctx.program.scalar_type.to_llvm()

proc to_llvm(typ: Type, ctx: Context): TypeRef =
  case typ.kind:
    of TypeIndex: result = nim_int_type()
    of TypeScalar: result = ctx.scalar_type()
    of TypeBoolean: result = int1_type()
    of TypeArray: result = pointer_type(typ.item.to_llvm(ctx), 0)

proc to_llvm(instrs: seq[Instr], ctx: Context) =
  let builder = ctx.builder
  for instr in instrs:
    var res = ValueRef(nil)
    
    template binop(op) =
      res = builder.op(
        ctx[instr.args[0]], ctx[instr.args[1]], cstring($instr.res)
      )
    
    template unop(op) =
      res = builder.op(ctx[instr.args[0]], cstring($instr.res))
    
    template generic_op(op_kind, index_op, scalar_op) =
      if ctx.kernel.regs[instr.args[0]].typ.kind == TypeScalar:
        op_kind(scalar_op)
      else:
        op_kind(index_op)
    
    case instr.kind:
      of InstrIndex:
        res = const_nim_int(instr.index_lit)
      of InstrScalar:
        res = const_real(ctx.scalar_type(), cdouble(instr.scalar_lit))
      of InstrBoolean:
        res = const_int(int1_type(), culonglong(ord(instr.boolean_lit)), 0)
      of InstrAdd: generic_op(binop, build_nsw_add, build_fadd)
      of InstrSub: generic_op(binop, build_nsw_sub, build_fsub)
      of InstrMul: generic_op(binop, build_nsw_mul, build_fmul)
      of InstrDiv: binop(build_fdiv)
      of InstrIndexDiv: binop(build_sdiv)
      of InstrMod: binop(build_srem)
      of InstrWrap:
        res = builder.build_srem(ctx[instr.args[0]], ctx[instr.args[1]], cstring($instr.res & "_mod"))
        res = builder.build_add(res, ctx[instr.args[1]], cstring($instr.res & "_offset"))
        res = builder.build_srem(res, ctx[instr.args[1]], cstring($instr.res))
      of InstrNegate: generic_op(unop, build_negate, build_fnegate)
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
        res = builder.build_call2(
          ctx.builtin.scalar_unary_intrinsic_signature(),
          fn, [ctx[instr.args[0]]], cstring($instr.res)
        )
      of InstrPow:
        let fn = case instr.kind:
          of InstrPow: ctx.builtin.pow
          else: nil
        res = builder.build_call2(
          ctx.builtin.scalar_binary_intrinsic_signature(),
          fn, [ctx[instr.args[0]], ctx[instr.args[1]]],
          cstring($instr.res)
        )
      of InstrEq: generic_op(binop, build_icmp_eq, build_fcmp_oeq)
      of InstrLt: generic_op(binop, build_icmp_slt, build_fcmp_olt)
      of InstrLe: generic_op(binop, build_icmp_sle, build_fcmp_ole)
      of InstrAnd: binop(build_and)
      of InstrOr: binop(build_or)
      of InstrToScalar, InstrToIndex:
        let
          from_typ = ctx.kernel.regs[instr.args[0]].typ
          to_typ = ctx.kernel.regs[instr.res].typ
        
        template convert(name: untyped) =
          res = builder.name(ctx[instr.args[0]], to_typ.to_llvm(ctx), cstring($instr.res))
        
        if from_typ.kind == TypeIndex and to_typ.kind == TypeScalar:
          convert(build_si_to_fp)
        elif from_typ.kind == TypeScalar and to_typ.kind == TypeIndex:
          convert(build_fp_to_si)
        else:
          raise GeneratorError(msg: "Unable to convert " & $from_typ & " to " & $to_typ)
      of InstrRead, InstrWrite, InstrOverwrite:
        let
          align = cuint(4) # TODO
          value_ptr = builder.build_gep2(
            ctx.scalar_type(), ctx[instr.tensor],
            [ctx[instr.args[0]]], "value_ptr"
          )
        value_ptr.set_is_in_bounds(1)
        
        case instr.kind:
          of InstrWrite, InstrRead:
            let value = builder.build_load2(
              ctx.scalar_type(), value_ptr, cstring($instr.res)
            )
            value.set_alignment(align)
            case instr.kind:
              of InstrRead: res = value
              of InstrWrite:
                builder.build_store(builder.build_fadd(
                  value, ctx[instr.args[1]], "new_value"
                ), value_ptr).set_alignment(align)
              else: discard
          of InstrOverwrite:
            builder.build_store(ctx[instr.args[1]], value_ptr).set_alignment(align)
          else: discard
      of InstrLen:
        res = builder.build_call2(ctx.builtin.len_signature(), ctx.builtin.len, [
          ctx.fn.get_param(0),
          const_nim_int(int(instr.tensor))
        ], cstring($instr.res))
      of InstrShape:
        res = builder.build_call2(ctx.builtin.shape_signature(), ctx.builtin.shape, [
          ctx.fn.get_param(0),
          const_nim_int(int(instr.tensor)),
          const_nim_int(instr.dim)
        ], cstring($instr.res))
      of InstrShapeLen:
        res = builder.build_call2(ctx.builtin.shape_len_signature(), ctx.builtin.shape_len, [
          ctx.fn.get_param(0),
          const_nim_int(int(instr.tensor))
        ], cstring($instr.res))
      of InstrEpoch:
        res = builder.build_call2(ctx.builtin.epoch_signature(),
          ctx.builtin.epoch, [ctx.fn.get_param(0)], cstring($instr.res)
        )
      of InstrLoop:
        let
          header_block = builder.get_insert_block()
          cond_block = ctx.fn.append_basic_block("cond")
          body_block = ctx.fn.append_basic_block("body")
          end_block = ctx.fn.append_basic_block("end")
          incr_block = ctx.fn.append_basic_block("incr")
        discard builder.build_br(cond_block)
        builder.position_builder_at_end(cond_block)
        ctx[instr.loop_iter] = builder.build_phi(
          ctx.kernel.regs[instr.loop_iter].typ.to_llvm(ctx),
          cstring("iter_" & $instr.loop_iter)
        )
        let
          cond = builder.build_icmp_eq(
            ctx[instr.loop_iter],
            ctx[instr.args[1]],
            "exitcond"
          )
        discard builder.build_cond_br(cond, end_block, body_block)
        builder.position_builder_at_end(body_block)
        instr.body.to_llvm(ctx)
        discard builder.build_br(incr_block)
        
        if instr.loop_step <= 0:
          raise GeneratorError(msg: "Loop step size must be a positive integer.")
        
        builder.position_builder_at_end(incr_block)
        let new_iter = builder.build_add(
          ctx[instr.loop_iter],
          const_nim_int(instr.loop_step),
          "incr_iter"
        )
        discard builder.build_br(cond_block)
        
        ctx[instr.loop_iter].add_incoming(
          [ctx[instr.args[0]], new_iter],
          [header_block, incr_block]
        )
        builder.position_builder_at_end(end_block)
      of InstrThreads:
        var closure_fields: seq[TypeRef] = @[]
        for reg in instr.threads_closure.regs:
          closure_fields.add(ctx.kernel.regs[reg].typ.to_llvm(ctx))
        for tensor in instr.threads_closure.tensors:
          closure_fields.add(pointer_type(ctx.scalar_type(), 0))
        let closure_type = struct_type(closure_fields)
        
        let
          current_block = builder.get_insert_block()
          sig = task_proc_signature()
          task = ctx.module.add_function(cstring($ctx.kernel_id & "_task"), sig)
          entry = task.append_basic_block(cstring("entry"))
        
        let task_ctx = Context(
          program: ctx.program,
          target: ctx.target,
          kernel: ctx.kernel,
          kernel_id: ctx.kernel_id,
          module: ctx.module,
          builder: ctx.builder,
          fn: task,
          builtin: ctx.builtin,
          regs: new_seq[ValueRef](ctx.kernel.regs.len),
          tensors: new_seq[ValueRef](ctx.program.tensors.len)
        )
        task_ctx[instr.threads_begin] = task.get_param(1)
        task_ctx[instr.threads_end] = task.get_param(2)
        
        var offset = 0
        builder.position_builder_at_end(current_block)
        let closure = builder.build_alloca(closure_type, "closure")
        builder.position_builder_at_end(entry)
        let task_closure = builder.build_bit_cast(task.get_param(3), closure_type.pointer_type(0), "closure")
        
        template make_closure(ids) =
          for id in ids:
            block:
              builder.position_builder_at_end(current_block)
              let field_ptr = builder.build_gep2(closure_type, closure, [
                const_int32(0), const_int32(int32(offset))
              ], "field_ptr")
              discard builder.build_store(ctx[id], field_ptr)
            block:
              builder.position_builder_at_end(entry)
              let field_ptr = builder.build_gep2(closure_type, task_closure, [
                const_int32(0), const_int32(int32(offset))
              ], "field_ptr")
              task_ctx[id] = builder.build_load2(closure_fields[offset], field_ptr, cstring($id))
            offset += 1
        
        make_closure(instr.threads_closure.regs)
        make_closure(instr.threads_closure.tensors)
        
        builder.position_builder_at_end(current_block)
        discard builder.build_call2(ctx.builtin.run_threads_signature(), ctx.builtin.run_threads, [
          ctx.fn.get_param(0), ctx[instr.args[0]], ctx[instr.args[1]],
          builder.build_bit_cast(closure, void_ptr_type(), "data"),
          task
        ], cstring(""))
        
        builder.position_builder_at_end(entry)
        instr.body.to_llvm(task_ctx)
        discard builder.build_ret()
        
        builder.position_builder_at_end(current_block)
        discard builder.build_call2(
          ctx.builtin.join_threads_signature(),
          ctx.builtin.join_threads,
          [ctx.fn.get_param(0)],
          cstring("")
        )
      of InstrArray:
        let
          array_type = ctx.kernel.regs[instr.res].typ
          item_type = array_type.item.to_llvm(ctx)
          current_block = builder.get_insert_block()
        builder.position_builder_at_start(ctx.fn.get_entry_basic_block())
        res = builder.build_array_alloca(
          array_type.item.to_llvm(ctx),
          const_nim_int(instr.args.len),
          cstring($instr.res)
        )
        builder.position_builder_at_end(current_block)
        for it, arg in instr.args:
          let value_ptr = builder.build_gep2(
            item_type,
            res,
            [const_nim_int(it)],
            "array_value_ptr"
          )
          discard builder.build_store(ctx[arg], value_ptr)
      of InstrArrayRead:
        let
          array_type = ctx.kernel.regs[instr.args[0]].typ
          item_type = array_type.item.to_llvm(ctx)
          value_ptr = builder.build_gep2(
            item_type,
            ctx[instr.args[0]],
            [ctx[instr.args[1]]],
            "array_value_ptr"
          )
        res = builder.build_load2(item_type, value_ptr, cstring($instr.res))
      of InstrArrayLen:
        res = const_nim_int(ctx.kernel.regs[instr.args[0]].typ.len)
      of InstrGpu:
        when defined(opencl):
          let source = instr.body.to_cl(instr.gpu_closure, instr.gpu_indices, ctx.kernel, ctx.program)
          ctx.gpu_sources.add(GpuKernelSource(name: "cl_kernel", source: source))
      else:
        raise GeneratorError(msg: "Unable to generate LLVM IR for " & $instr.kind)
    
    if not res.is_nil:
      assert instr.res != RegId(0)
      ctx[instr.res] = res

proc to_llvm(kernel: Kernel, kernel_id: KernelId, ctx: Context) =
  let
    builder = ctx.builder
    kernel_block = ctx.fn.append_basic_block(cstring($kernel_id))
  discard builder.build_br(kernel_block)
  builder.position_builder_at_end(kernel_block)
  
  ctx.regs = new_seq[ValueRef](kernel.regs.len)
  kernel.setup.to_llvm(ctx)

proc to_llvm*(program: Program): (ModuleRef, seq[GpuKernelSource]) =
  program.assert_gen("llvm", requires={
    StageTyped, StageGenerated, StageTensors, StageShapes,
    StageLoops, StageTensorInstrs, StageSortedShapes,
    StageConditions
  })

  let
    module = module_create_with_name("module")
    builtin = init_builtin(module, program)
  var gpu_sources: seq[GpuKernelSource] = @[]
  for name, target in program.targets:
    let
      sig = function_type(void_type(), [model_ptr_type()])
      fn = module.add_function(cstring("target_" & name), sig)
      entry = fn.append_basic_block("entry")
      builder = create_builder()
    
    builder.enable_fast_math()
    
    builder.position_builder_at_end(entry)
    var ctx = Context(
      program: program,
      module: module,
      builder: builder,
      fn: fn,
      builtin: builtin,
      target: name,
      tensors: new_seq[ValueRef](program.tensors.len)
    )
    for tensor_id in target.tensors:
      ctx[tensor_id] = builder.build_call2(
        ctx.builtin.tensor_signature(),
        ctx.builtin.tensor,
        [ctx.fn.get_param(0), const_nim_int(int(tensor_id))],
        cstring($tensor_id)
      )
    for it, kernel in target.kernels:
      ctx.kernel = kernel
      ctx.kernel_id = KernelId(it + 1)
      kernel.to_llvm(KernelId(it + 1), ctx)
    discard builder.build_ret()
    dispose_builder(builder)
    gpu_sources.add(ctx.gpu_sources)
  result = (module, gpu_sources)

type
  JitBuiltin* = object
    tensor*: pointer
    shape*: pointer
    len*: pointer
    shape_len*: pointer
    debug_index*: pointer
    debug_scalar*: pointer
    epoch*: pointer
    run_threads*: pointer
    join_threads*: pointer
    run_gpu_kernel*: pointer
    set_gpu_kernel_arg*: pointer
  
  Jit* = ref object
    module: ModuleRef
    engine: ExecutionEngineRef
    builtin: JitBuiltin
    gpu_context: GpuContext

proc finalize*(jit: Jit) =
  if not jit.engine.is_nil:
    dispose_execution_engine(jit.engine)

proc new_jit*(module: ModuleRef, builtin: JitBuiltin): Jit =
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
    if not err.is_nil:
      dispose_message(err)
  
  let 
    triple = get_default_target_triple()
    target_features = get_host_cpu_features()
    target_cpu = get_host_cpu_name()
  module.set_target(triple)
  var target: TargetRef = nil
  if get_target_from_triple(triple, target.addr, err.addr) != 0:
    raise JitError(msg: $err)
  let machine = create_target_machine(
    target, triple, target_cpu, target_features,
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
    pass_err = module.run_passes("default<O3>", machine, opts)
  if not pass_err.is_nil:
    let
      msg = pass_err.get_error_message()
      str = $msg
    dispose_error_message(msg)
    raise JitError(msg: str)
  dispose_pass_builder_options(opts)
  
  if create_jit_compiler_for_module(result.engine.addr, module, 3, err.addr) != 0:
    raise JitError(msg: $err)
  
  for name, value in builtin.field_pairs:
    if value.is_nil:
      raise JitError(msg: "Builtin " & name & " is nil")
    let fn = result.module.get_named_function(cstring(name))
    if not fn.is_nil:
      result.engine.add_global_mapping(fn, value)

proc get_proc*[T: proc](jit: Jit, name: string): T =
  result = cast[T](get_function_address(jit.engine, cstring(name)))

proc save_bitcode*(jit: Jit, path: string) =
  jit.module.save_bitcode(path)
