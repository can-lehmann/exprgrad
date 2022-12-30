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

# Wrapper for the LLVM C-API
# The LLVM C-API is licensed under Apache-2.0 WITH LLVM-exception
# See LLVM_LICENSE.txt (or https://llvm.org/LICENSE.txt) for more details.

{.passc: gorge("llvm-config-13 --cflags").}
{.passl: gorge("llvm-config-13 --libs").}

type
  ModuleRef* {.importc: "LLVMModuleRef", header: "<llvm-c/Core.h>".} = distinct pointer
  TypeRef* {.importc: "LLVMTypeRef", header: "<llvm-c/Core.h>".} = distinct pointer
  ValueRef* {.importc: "LLVMValueRef", header: "<llvm-c/Core.h>".} = distinct pointer
  BasicBlockRef* {.importc: "LLVMBasicBlockRef", header: "<llvm-c/Core.h>".} = distinct pointer
  BuilderRef* {.importc: "LLVMBuilderRef", header: "<llvm-c/Core.h>".} = object
  ExecutionEngineRef* {.importc: "LLVMExecutionEngineRef", header: "<llvm-c/ExecutionEngine.h>".} = distinct pointer
  ContextRef* {.importc: "LLVMContextRef", header: "<llvm-c/Core.h>".} = object
  PassManagerRef* {.importc: "LLVMPassManagerRef", header: "<llvm-c/Core.h>".} = distinct pointer
  PassManagerBuilderRef* {.importc: "LLVMPassManagerBuilderRef", header: "<llvm-c/Types.h>".} = distinct pointer
  TargetDataRef* {.importc: "LLVMTargetDataRef", header: "<llvm-c/Types.h>".} = distinct pointer
  ModuleProviderRef* {.importc: "LLVMModuleProviderRef", header: "<llvm-c/Types.h>".} = distinct pointer
  TargetRef* {.importc: "LLVMTargetRef", header: "<llvm-c/Types.h>".} = distinct pointer
  TargetMachineRef* {.importc: "LLVMTargetMachineRef", header: "<llvm-c/Types.h>".} = distinct pointer
  AttributeRef* {.importc: "LLVMAttributeRef", header: "<llvm-c/Types.h>".} = distinct pointer
  NamedMDNodeRef* {.importc: "LLVMNamedMDNodeRef", header: "<llvm-c/Types.h>".} = distinct pointer
  MetadataRef* {.importc: "LLVMMetadataRef", header: "<llvm-c/Types.h>".} = distinct pointer
  MemoryBufferRef* {.importc: "LLVMMemoryBufferRef", header: "<llvm-c/Types.h>".} = distinct pointer
  PassRegistryRef* {.importc: "LLVMPassRegistryRef", header: "<llvm-c/Types.h>".} = distinct pointer
  PassBuilderOptionsRef{.importc: "LLVMPassBuilderOptionsRef", header: "<llvm-c/Types.h>".} = distinct pointer
  ErrorRef{.importc: "LLVMErrorRef", header: "<llvm-c/Types.h>".} = distinct pointer
  LlvmBool = cint
  
  IntPredicate* {.size: sizeof(cint).} = enum
    IntEq = 32, IntNe,
    IntUGt, IntUGe, IntULt, IntULe,
    IntSGt, IntSGe, IntSLt, IntSLe
  
  RealPredicate* {.size: sizeof(cint).} = enum
    RealPredicateFalse,
    RealOEq, RealOGt, RealOGe, RealOLt, RealOLe, RealONe,
    RealUEq, RealUGt, RealUGe, RealULt, RealULe, RealUNe,
    RealPredicateTrue
  
  CodeGenOptLevel* = enum
    OptNone, OptLess, OptDefault, OptAggressive
  
  RelocMode* = enum
    RelocDefault, RelocStatic, RelocPic, RelocDynamicNoPic,
    RelocRopi, RelocRwpi, RelocRopiRwpi
  
  CodeModel* = enum
    CodeModelDefault, CodeModelJitDefault, CodeModelTiny, CodeModelSmall,
    CodeModelKernel, CodeModelMedium, CodeModelLarge

proc isNil*(x: ValueRef): bool {.borrow.}
proc isNil*(x: TypeRef): bool {.borrow.}
proc isNil*(x: ModuleRef): bool {.borrow.}
proc isNil*(x: ExecutionEngineRef): bool {.borrow.}
proc isNil*(x: PassManagerRef): bool {.borrow.}
proc isNil*(x: MemoryBufferRef): bool {.borrow.}
proc isNil*(x: PassRegistryRef): bool {.borrow.}
proc isNil*(x: ErrorRef): bool {.borrow.}

{.push header: "<llvm-c/Core.h>".}
proc get_global_context*(): ContextRef {.importc: "LLVMGetGlobalContext".}
proc get_global_pass_registry*(): PassRegistryRef {.importc: "LLVMGetGlobalPassRegistry".}
proc is_llvm_multithreaded*(): LlvmBool {.importc: "LLVMIsMultithreaded".}
proc module_create_with_name*(name: cstring): ModuleRef {.importc: "LLVMModuleCreateWithName".}
proc set_data_layout*(module: ModuleRef, layout: cstring) {.importc: "LLVMSetDataLayout".}
proc get_data_layout*(module: ModuleRef): cstring {.importc: "LLVMGetDataLayout".}
proc set_target*(module: ModuleRef, target: cstring) {.importc: "LLVMSetTarget".}
proc add_function*(module: ModuleRef, name: cstring, typ: TypeRef): ValueRef {.importc: "LLVMAddFunction".}
proc get_named_function*(module: ModuleRef, name: cstring): ValueRef {.importc: "LLVMGetNamedFunction".}
proc get_first_function*(module: ModuleRef): ValueRef {.importc: "LLVMGetFirstFunction".}
proc get_last_function*(module: ModuleRef): ValueRef {.importc: "LLVMGetLastFunction".}
proc get_next_function*(fn: ValueRef): ValueRef {.importc: "LLVMGetNextFunction".}
proc get_previous_function*(fn: ValueRef): ValueRef {.importc: "LLVMGetPreviousFunction".}

proc create_memory_buffer_with_contents_of_file*(path: cstring, mem: ptr MemoryBufferRef, msg: ptr cstring): LlvmBool {.importc: "LLVMCreateMemoryBufferWithContentsOfFile".}
proc dispose_memory_buffer*(mem: MemoryBufferRef) {.importc: "LLVMDisposeMemoryBuffer".}

proc add_attribute_at_index*(fn: ValueRef, idx: cuint, attr: AttributeRef) {.importc: "LLVMAddAttributeAtIndex".}
proc create_string_attribute*(ctx: ContextRef, key: cstring, keyLen: cuint, val: cstring, valLen: cuint): AttributeRef {.importc: "LLVMCreateStringAttribute".}

proc md_string_in_context2*(ctx: ContextRef, str: cstring, len: csize_t): MetadataRef {.importc: "LLVMMDStringInContext2".}
proc md_node_in_context2*(ctx: ContextRef, children: ptr MetadataRef, count: csize_t): MetadataRef {.importc: "LLVMMDNodeInContext2".}
proc metadata_as_value*(ctx: ContextRef, md: MetadataRef): ValueRef {.importc: "LLVMMetadataAsValue".}
proc value_as_metadata*(val: ValueRef): MetadataRef {.importc: "LLVMValueAsMetadata".}
proc get_md_string*(val: ValueRef, len: ptr cuint): cstring {.importc: "LLVMGetMDString".}
proc set_metadata*(val: ValueRef, kindId: cuint, node: ValueRef) {.importc: "LLVMSetMetadata".}
proc has_metadata*(val: ValueRef): cint {.importc: "LLVMHasMetadata".}
proc get_md_kind_id_in_context*(ctx: ContextRef, name: cstring, len: cuint): cuint {.importc: "LLVMGetMDKindIDInContext".}
proc get_md_kind_id*(name: cstring, len: cuint): cuint {.importc: "LLVMGetMDKindID".}
proc add_named_metadata_operand*(module: ModuleRef, name: cstring, val: ValueRef) {.importc: "LLVMAddNamedMetadataOperand".}
proc get_or_insert_named_metadata*(module: ModuleRef, name: cstring, len: csize_t): NamedMDNodeRef {.importc: "LLVMGetOrInsertNamedMetadata".}

proc append_basic_block*(function: ValueRef, name: cstring): BasicBlockRef {.importc: "LLVMAppendBasicBlock".}
proc get_first_instruction*(basicBlock: BasicBlockRef): ValueRef {.importc: "LLVMGetFirstInstruction".}
proc get_last_instruction*(basicBlock: BasicBlockRef): ValueRef {.importc: "LLVMGetLastInstruction".}
proc get_entry_basic_block*(fn: ValueRef): BasicBlockRef {.importc: "LLVMGetEntryBasicBlock".}

proc create_module_provider_for_existing_module*(module: ModuleRef): ModuleProviderRef {.importc: "LLVMCreateModuleProviderForExistingModule".}
proc dispose_module_provider*(provider: ModuleProviderRef) {.importc: "LLVMDisposeModuleProvider".}

proc create_builder*(): BuilderRef {.importc: "LLVMCreateBuilder".}
proc position_builder_at_end*(builder: BuilderRef, basicBlock: BasicBlockRef) {.importc: "LLVMPositionBuilderAtEnd".}
proc get_insert_block*(builder: BuilderRef): BasicBlockRef {.importc: "LLVMGetInsertBlock".}
proc position_builder*(builder: BuilderRef, basicBlock: BasicBlockRef, instr: ValueRef) {.importc: "LLVMPositionBuilder".}
proc position_builder_before*(builder: BuilderRef, instr: ValueRef) {.importc: "LLVMPositionBuilderBefore".}
proc dispose_builder*(builder: BuilderRef) {.importc: "LLVMDisposeBuilder".}
proc set_default_fp_math_tag*(builder: BuilderRef, tag: MetadataRef) {.importc: "LLVMBuilderSetDefaultFPMathTag".}
proc get_default_fp_math_tag*(builder: BuilderRef): MetadataRef {.importc: "LLVMBuilderGetDefaultFPMathTag".}
proc dispose_message*(msg: cstring) {.importc: "LLVMDisposeMessage".}
proc get_value_name2*(value: ValueRef, len: ptr csize_t): cstring {.importc: "LLVMGetValueName2".}
proc set_value_name2*(value: ValueRef, name: cstring, len: csize_t) {.importc: "LLVMSetValueName2".}
proc print_value_to_string*(value: ValueRef): cstring {.importc: "LLVMPrintValueToString".}
proc print_type_to_string*(value: TypeRef): cstring {.importc: "LLVMPrintTypeToString".}

proc get_alignment*(val: ValueRef): cuint {.importc: "LLVMGetAlignment".}
proc set_alignment*(val: ValueRef, bytes: cuint) {.importc: "LLVMSetAlignment".}
proc is_in_bounds*(gep: ValueRef): LlvmBool {.importc: "LLVMIsInBounds".}
proc set_is_in_bounds*(gep: ValueRef, val: LlvmBool) {.importc: "LLVMSetIsInBounds".}

proc void_type*(): TypeRef {.importc: "LLVMVoidType".}
proc int1_type*(): TypeRef {.importc: "LLVMInt1Type".}
proc int8_type*(): TypeRef {.importc: "LLVMInt8Type".}
proc int16_type*(): TypeRef {.importc: "LLVMInt16Type".}
proc int32_type*(): TypeRef {.importc: "LLVMInt32Type".}
proc int64_type*(): TypeRef {.importc: "LLVMInt64Type".}
proc int128_type*(): TypeRef {.importc: "LLVMInt128Type".}
proc half_type*(): TypeRef {.importc: "LLVMHalfType".}
proc float_type*(): TypeRef {.importc: "LLVMFloatType".}
proc double_type*(): TypeRef {.importc: "LLVMDoubleType".}
proc float16_type*(): TypeRef {.importc: "LLVMHalfType".}
proc float32_type*(): TypeRef {.importc: "LLVMFloatType".}
proc float64_type*(): TypeRef {.importc: "LLVMDoubleType".}
proc float128_type*(): TypeRef {.importc: "LLVMFP128Type".}
proc pointer_type*(elem: TypeRef, addressSpace: cuint): TypeRef {.importc: "LLVMPointerType".}
proc function_type*(ret: TypeRef, args: ptr TypeRef, argCount: cuint, isVarArg: LlvmBool): TypeRef {.importc: "LLVMFunctionType".}
proc struct_type*(elemTypes: ptr TypeRef, count: cuint, packed: LlvmBool): TypeRef {.importc: "LLVMStructType".}
proc struct_create_named*(ctx: ContextRef, name: cstring): TypeRef {.importc: "LLVMStructCreateNamed".}
proc struct_set_body*(ty: TypeRef, elemTypes: ptr TypeRef, count: cuint, packed: LlvmBool) {.importc: "LLVMStructSetBody".}
proc vector_type*(elemType: TypeRef, count: cuint): TypeRef {.importc: "LLVMVectorType".}
proc llvm_type_of*(x: ValueRef): TypeRef {.importc: "LLVMTypeOf".}

proc get_param*(fn: ValueRef, index: cuint): ValueRef {.importc: "LLVMGetParam".}
proc get_undef*(ty: TypeRef): ValueRef {.importc: "LLVMGetUndef".}

proc const_int*(ty: TypeRef, n: culonglong, signExtend: LlvmBool): ValueRef {.importc: "LLVMConstInt".}
proc const_real*(ty: TypeRef, n: cdouble): ValueRef {.importc: "LLVMConstReal".}
proc const_struct*(vals: ptr ValueRef, count: cuint, packed: LlvmBool): ValueRef {.importc: "LLVMConstStruct".}
proc const_array*(elemType: TypeRef, vals: ptr ValueRef, length: cuint): ValueRef {.importc: "LLVMConstArray".}
proc const_vector*(vals: ptr ValueRef, count: cuint): ValueRef {.importc: "LLVMConstVector".}

proc build_add*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildAdd".}
proc build_nsw_add*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildNSWAdd".}
proc build_nuw_add*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildNUWAdd".}
proc build_fadd*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildFAdd".}
proc build_sub*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildSub".}
proc build_nsw_sub*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildNSWSub".}
proc build_nuw_sub*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildNUWSub".}
proc build_fsub*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildFSub".}
proc build_mul*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildMul".}
proc build_nsw_mul*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildNSWMul".}
proc build_nuw_mul*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildNUWMul".}
proc build_fmul*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildFMul".}
proc build_sdiv*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildSDiv".}
proc build_udiv*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildUDiv".}
proc build_fdiv*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildFDiv".}
proc build_srem*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildSRem".}
proc build_urem*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildURem".}
proc build_frem*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildFRem".}
proc build_icmp*(builder: BuilderRef, op: IntPredicate, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildICmp".}
proc build_fcmp*(builder: BuilderRef, op: RealPredicate, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildFCmp".}
proc build_and*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildAnd".}
proc build_or*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildOr".}
proc build_xor*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildXor".}
proc build_not*(builder: BuilderRef, a: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildNot".}
proc build_select*(builder: BuilderRef, cond, then, otherwise: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildSelect".}

proc build_bit_cast*(builder: BuilderRef, a: ValueRef, dest: TypeRef, name: cstring): ValueRef {.importc: "LLVMBuildBitCast".}
proc build_fp_to_ui*(builder: BuilderRef, val: ValueRef, destTy: TypeRef, name: cstring): ValueRef {.importc: "LLVMBuildFPToUI".}
proc build_fp_to_si*(builder: BuilderRef, val: ValueRef, destTy: TypeRef, name: cstring): ValueRef {.importc: "LLVMBuildFPToSI".}
proc build_ui_to_fp*(builder: BuilderRef, val: ValueRef, destTy: TypeRef, name: cstring): ValueRef {.importc: "LLVMBuildUIToFP".}
proc build_si_to_fp*(builder: BuilderRef, val: ValueRef, destTy: TypeRef, name: cstring): ValueRef {.importc: "LLVMBuildSIToFP".}

proc build_call*(builder: BuilderRef, fn: ValueRef, args: ptr ValueRef, numArgs: cuint, name: cstring): ValueRef {.importc: "LLVMBuildCall".}
proc build_call2*(builder: BuilderRef, ty: TypeRef, fn: ValueRef, args: ptr ValueRef, numArgs: cuint, name: cstring): ValueRef {.importc: "LLVMBuildCall2".}

proc build_alloca*(builder: BuilderRef, ty: TypeRef, name: cstring): ValueRef {.importc: "LLVMBuildAlloca".}
proc build_array_alloca*(builder: BuilderRef, ty: TypeRef, val: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildArrayAlloca".}
proc build_load*(builder: BuilderRef, pointerVal: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildLoad".}
proc build_load2*(builder: BuilderRef, ty: TypeRef, pointerVal: ValueRef, name: cstring): ValueRef {.importc: "LLVMBuildLoad2".}
proc build_store*(builder: BuilderRef, val, pointerVal: ValueRef): ValueRef {.importc: "LLVMBuildStore".}
proc build_gep2*(builder: BuilderRef, ty: TypeRef, pointerVal: ValueRef, indices: ptr ValueRef, numIndices: cuint, name: cstring): ValueRef {.importc: "LLVMBuildGEP2".}

proc build_ret*(builder: BuilderRef, ret: ValueRef): ValueRef {.importc: "LLVMBuildRet".}
proc build_ret_void*(builder: BuilderRef): ValueRef {.importc: "LLVMBuildRetVoid".}
proc build_ret*(builder: BuilderRef): ValueRef {.importc: "LLVMBuildRetVoid".}
proc build_br*(builder: BuilderRef, next: BasicBlockRef): ValueRef {.importc: "LLVMBuildBr".}
proc build_cond_br*(builder: BuilderRef, cond: ValueRef, then, otherwise: BasicBlockRef): ValueRef {.importc: "LLVMBuildCondBr".}
proc build_phi*(builder: BuilderRef, ty: TypeRef, name: cstring): ValueRef {.importc: "LLVMBuildPhi".}
proc add_incoming*(phi: ValueRef, vals: ptr ValueRef, blocks: ptr BasicBlockRef, count: cuint) {.importc: "LLVMAddIncoming".}
{.pop.}

iterator functions*(module: ModuleRef): ValueRef =
  var cur = module.get_first_function()
  let last = module.get_last_function()
  while pointer(cur) != pointer(last):
    yield cur
    cur = cur.get_next_function()

proc name*(value: ValueRef): string =
  var len: csize_t = 0
  result = $value.get_value_name2(len.addr)

proc `name=`*(value: ValueRef, name: cstring) =
  value.set_value_name2(name, csize_t(name.len))

proc `$`*(value: ValueRef): string =
  var str = value.print_value_to_string()
  result = $str
  dispose_message(str)

proc `$`*(value: TypeRef): string =
  var str = value.print_type_to_string()
  result = $str
  dispose_message(str)

proc functionType*(ret: TypeRef,
                    args: openArray[TypeRef],
                    isVarArg: bool = false): TypeRef =
  if args.len > 0:
    result = function_type(ret, args[0].unsafeAddr, cuint(args.len), cint(ord(isVarArg)))
  else:
    result = function_type(ret, nil, cuint(args.len), cint(ord(isVarArg)))

proc structType*(elemTypes: openArray[TypeRef], packed: bool = false): TypeRef =
  if elemTypes.len > 0:
    result = struct_type(elemTypes[0].unsafeAddr, cuint(elemTypes.len), LlvmBool(ord(packed)))
  else:
    result = struct_type(nil, cuint(elemTypes.len), LlvmBool(ord(packed)))

proc addIncoming*(phi: ValueRef,
                   vals: openArray[ValueRef],
                   blocks: openArray[BasicBlockRef]) =
  assert vals.len == blocks.len
  if vals.len > 0:
    phi.add_incoming(vals[0].unsafeAddr, blocks[0].unsafeAddr, cuint(vals.len))

proc buildCall2*(builder: BuilderRef,
                  typ: TypeRef,
                  fn: ValueRef,
                  args: openArray[ValueRef],
                  name: cstring): ValueRef =
  var argsAddr: ptr ValueRef = nil
  if args.len > 0:
    argsAddr = args[0].unsafeAddr
  result = builder.build_call2(typ, fn, argsAddr, cuint(args.len), name)

proc buildGep2*(builder: BuilderRef,
                 typ: TypeRef,
                 pointerVal: ValueRef,
                 indices: openArray[ValueRef],
                 name: cstring): ValueRef =
  assert indices.len > 0
  
  result = builder.build_gep2(typ, pointerVal,
    indices[0].unsafeAddr, cuint(indices.len), name
  )

proc positionBuilderAtStart*(builder: BuilderRef,
                                basicBlock: BasicBlockRef) =
  let instr = basicBlock.get_first_instruction()
  if instr.isNil:
    builder.position_builder_at_end(basicBlock)
  else:
    builder.position_builder_before(instr)

proc constInt32*(x: int32): ValueRef =
  const_int(int32_type(), cast[culonglong](clonglong(x)), 0)

proc nimIntType*(): TypeRef =
  when sizeof(int) == 8:
    return int64_type()
  elif sizeof(int) == 4:
    return int32_type()
  elif sizeof(int) == 2:
    return int16_type()
  elif sizeof(int) == 1:
    return int8_type()
  else:
    error("Unknown nim int size: " & $sizeof(int))

proc constNimInt*(x: int): ValueRef =
  const_int(nimIntType(), cast[culonglong](clonglong(x)), 0)

template defineBuildIcmp(buildName: untyped, pred: static[IntPredicate]) =
  proc build_name*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef =
    result = builder.build_icmp(pred, a, b, name)

defineBuildIcmp(buildIcmpEq, IntEq)
defineBuildIcmp(buildIcmpNe, IntNe)
defineBuildIcmp(buildIcmpSlt, IntSLt)
defineBuildIcmp(buildIcmpSle, IntSLe)

template defineBuildFcmp(buildName: untyped, pred: static[RealPredicate]) =
  proc build_name*(builder: BuilderRef, a, b: ValueRef, name: cstring): ValueRef =
    result = builder.build_fcmp(pred, a, b, name)

defineBuildFcmp(buildFcmpOeq, RealOEq)
defineBuildFcmp(buildFcmpOne, RealONe)
defineBuildFcmp(buildFcmpOlt, RealOLt)
defineBuildFcmp(buildFcmpOle, RealOLe)

proc buildNegate*(builder: BuilderRef, x: ValueRef, name: cstring): ValueRef =
  result = builder.build_sub(
    const_int(llvm_type_of(x), culonglong(0), 0), x, name
  )

proc buildFnegate*(builder: BuilderRef, x: ValueRef, name: cstring): ValueRef =
  result = builder.build_fsub(
    const_real(llvm_type_of(x), 0), x, name
  )

proc getMdKindId*(name: string): cuint =
  get_md_kind_id(cstring(name), cuint(len(name)))

proc mdStringInContext2*(ctx: ContextRef, str: string): MetadataRef =
  result = ctx.md_string_in_context2(cstring(str), csize_t(len(str)))

proc mdNodeInContext2*(ctx: ContextRef, children: openArray[MetadataRef]): MetadataRef =
  if children.len == 0:
    result = ctx.md_node_in_context2(nil, 0)
  else:
    result = ctx.md_node_in_context2(
      children[0].unsafeAddr, csize_t(children.len)
    )

proc createStringAttribute*(key, val: string): AttributeRef =
  result = create_string_attribute(get_global_context(),
    key.cstring, cuint(key.len),
    val.cstring, cuint(val.len)
  )

{.push header: "<llvm-c/BitWriter.h>".}
proc write_bitcode_to_file*(module: ModuleRef, path: cstring) {.importc: "LLVMWriteBitcodeToFile".}
{.pop.}

{.push header: "<llvm-c/BitReader.h>".}
proc parse_bitcode2*(mem: MemoryBufferRef, module: ptr ModuleRef): LlvmBool {.importc: "LLVMParseBitcode2".}
proc parse_bitcode_in_context2*(ctx: ContextRef, mem: MemoryBufferRef, module: ptr ModuleRef): LlvmBool {.importc: "LLVMParseBitcodeInContext2".}
{.pop.}

proc saveBitcode*(module: ModuleRef, path: string) =
  module.write_bitcode_to_file(path)

proc loadBitcode*(ctx: ContextRef, path: string): ModuleRef =
  var
    msg: cstring = nil
    mem = MemoryBufferRef(nil)
  defer:
    if not msg.isNil:
      msg.dispose_message()
    if not mem.isNil:
      mem.dispose_memory_buffer()
  if create_memory_buffer_with_contents_of_file(path.cstring, mem.addr, msg.addr) != 0:
    raise newException(IOError, $msg)
  if parse_bitcode_in_context2(ctx, mem, result.addr) != 0 or result.isNil:
    raise newException(ValueError, "Failed to parse module from " & path)

proc loadBitcode*(path: string): ModuleRef =
  get_global_context().loadBitcode(path)

type VerifierFailureAction* = enum
  AbortProcessAction, PrintMessageAction, ReturnStatusAction

{.push header: "<llvm-c/Analysis.h>".}
proc verify_module*(module: ModuleRef, failureAction: VerifierFailureAction, msg: ptr cstring): LlvmBool {.importc: "LLVMVerifyModule".}
proc verify_function*(module: ModuleRef, failureAction: VerifierFailureAction): LlvmBool {.importc: "LLVMVerifyFunction".}
{.pop.}

{.push header: "<llvm-c/ExecutionEngine.h>".}
proc create_execution_engine_for_module*(engine: ptr ExecutionEngineRef, module: ModuleRef, err: ptr cstring): cint {.importc: "LLVMCreateExecutionEngineForModule".}
proc create_jit_compiler_for_module*(engine: ptr ExecutionEngineRef, module: ModuleRef, optLevel: cuint, err: ptr cstring): cint {.importc: "LLVMCreateJITCompilerForModule".}
proc dispose_execution_engine*(engine: ExecutionEngineRef) {.importc: "LLVMDisposeExecutionEngine".}
proc get_function_address*(engine: ExecutionEngineRef, name: cstring): uint64 {.importc: "LLVMGetFunctionAddress".}
proc add_global_mapping*(engine: ExecutionEngineRef, global: ValueRef, address: pointer) {.importc: "LLVMAddGlobalMapping".}
proc link_in_mcjit*() {.importc: "LLVMLinkInMCJIT".}
proc get_execution_engine_target_data*(engine: ExecutionEngineRef): TargetDataRef {.importc: "LLVMGetExecutionEngineTargetData".}
{.pop.}

proc getFunctionAddress*[T](engine: ExecutionEngineRef, name: string): T =
  cast[T](get_function_address(engine, name.cstring))

{.push header: "<llvm-c/Target.h>".}
proc initialize_native_target*() {.importc: "LLVMInitializeNativeTarget".}
proc initialize_native_asm_printer*() {.importc: "LLVMInitializeNativeAsmPrinter".}
proc set_module_data_layout*(module: ModuleRef, layout: TargetDataRef) {.importc: "LLVMSetModuleDataLayout".}
{.pop.}

{.push header: "<llvm-c/TargetMachine.h>".}
proc get_default_target_triple*(): cstring {.importc: "LLVMGetDefaultTargetTriple".}
proc get_host_cpu_name*(): cstring {.importc: "LLVMGetHostCPUName".}
proc get_host_cpu_features*(): cstring {.importc: "LLVMGetHostCPUFeatures".}
proc get_target_from_triple*(triple: cstring, target: ptr TargetRef, err: ptr cstring): LlvmBool {.importc: "LLVMGetTargetFromTriple".}
proc target_has_jit*(target: TargetRef): LlvmBool {.importc: "LLVMTargetHasJIT".}
proc create_target_machine*(target: TargetRef, triple, cpu, features: cstring, level: CodeGenOptLevel, reloc: RelocMode, model: CodeModel): TargetMachineRef {.importc: "LLVMCreateTargetMachine".}
proc create_target_data_layout*(machine: TargetMachineRef): TargetDataRef {.importc: "LLVMCreateTargetDataLayout".}
proc add_analysis_passes*(machine: TargetMachineRef, manager: PassManagerRef) {.importc: "LLVMAddAnalysisPasses".}
{.pop.}

{.push header: "<llvm-c/Core.h>".}
proc create_pass_manager*(): PassManagerRef {.importc: "LLVMCreatePassManager".}
proc create_function_pass_manager*(module: ModuleRef): PassManagerRef {.importc: "LLVMCreateFunctionPassManager".}
proc run_pass_manager*(manager: PassManagerRef, module: ModuleRef): LlvmBool {.importc: "LLVMRunPassManager".}
proc initialize_function_pass_manager*(manager: PassManagerRef): LlvmBool {.importc: "LLVMInitializeFunctionPassManager".}
proc run_function_pass_manager*(manager: PassManagerRef, fn: ValueRef): LlvmBool {.importc: "LLVMRunFunctionPassManager".}
proc finalize_function_pass_manager*(manager: PassManagerRef): LlvmBool {.importc: "LLVMFinalizeFunctionPassManager".}
proc dispose_pass_manager*(manager: PassManagerRef) {.importc: "LLVMDisposePassManager".}
{.pop.}

{.push header: "<llvm-c/Transforms/PassManagerBuilder.h>".}
proc pass_manager_builder_create*(): PassManagerBuilderRef {.importc: "LLVMPassManagerBuilderCreate".}
proc pass_manager_builder_dispose*(builder: PassManagerBuilderRef) {.importc: "LLVMPassManagerBuilderDispose".}
proc set_opt_level*(builder: PassManagerBuilderRef, level: cuint) {.importc: "LLVMPassManagerBuilderSetOptLevel".}
proc set_size_level*(builder: PassManagerBuilderRef, level: cuint) {.importc: "LLVMPassManagerBuilderSetSizeLevel".}

proc populate_function_pass_manager*(builder: PassManagerBuilderRef, manager: PassManagerRef) {.importc: "LLVMPassManagerBuilderPopulateFunctionPassManager".}
proc populate_module_pass_manager*(builder: PassManagerBuilderRef, manager: PassManagerRef) {.importc: "LLVMPassManagerBuilderPopulateModulePassManager".}
proc populate_lto_pass_manager*(builder: PassManagerBuilderRef, manager: PassManagerRef, internalize, runInliner: LlvmBool) {.importc: "LLVMPassManagerBuilderPopulateLTOPassManager".}
proc create_pass_manager_builder*(): PassManagerBuilderRef {.importc: "LLVMPassManagerBuilderCreate".}
proc dispose_pass_manager_builder*(builder: PassManagerBuilderRef) {.importc: "LLVMPassManagerBuilderDispose".}
{.pop.}

proc `optLevel=`*(builder: PassManagerBuilderRef, level: int) =
  builder.set_opt_level(cuint(level))

proc `sizeLevel=`*(builder: PassManagerBuilderRef, level: int) =
  builder.set_size_level(cuint(level))

{.push header: "<llvm-c/Initialization.h>".}
proc initialize_core*(registry: PassRegistryRef) {.importc: "LLVMInitializeCore".}
proc initialize_transform_utils*(registry: PassRegistryRef) {.importc: "LLVMInitializeTransformUtils".}
proc initialize_scalar_opts*(registry: PassRegistryRef) {.importc: "LLVMInitializeScalarOpts".}
proc initialize_obj_carc_opts*(registry: PassRegistryRef) {.importc: "LLVMInitializeObjCARCOpts".}
proc initialize_vectorization*(registry: PassRegistryRef) {.importc: "LLVMInitializeVectorization".}
proc initialize_inst_combine*(registry: PassRegistryRef) {.importc: "LLVMInitializeInstCombine".}
proc initialize_aggressive_inst_combiner*(registry: PassRegistryRef) {.importc: "LLVMInitializeAggressiveInstCombiner".}
proc initialize_ipo*(registry: PassRegistryRef) {.importc: "LLVMInitializeIPO".}
proc initialize_instrumentation*(registry: PassRegistryRef) {.importc: "LLVMInitializeInstrumentation".}
proc initialize_analysis*(registry: PassRegistryRef) {.importc: "LLVMInitializeAnalysis".}
proc initialize_ipa*(registry: PassRegistryRef) {.importc: "LLVMInitializeIPA".}
proc initialize_code_gen*(registry: PassRegistryRef) {.importc: "LLVMInitializeCodeGen".}
proc initialize_target*(registry: PassRegistryRef) {.importc: "LLVMInitializeTarget".}
{.pop.}

{.push header: "<llvm-c/Transforms/Vectorize.h>".}
proc add_loop_vectorize_pass*(manager: PassManagerRef) {.importc: "LLVMAddLoopVectorizePass".}
proc add_slp_vectorize_pass*(manager: PassManagerRef) {.importc: "LLVMAddSLPVectorizePass".}
{.pop.}

{.push header: "<llvm-c/Transforms/PassBuilder.h>".}
proc create_pass_builder_options*(): PassBuilderOptionsRef {.importc: "LLVMCreatePassBuilderOptions".}
proc dispose_pass_builder_options*(opts: PassBuilderOptionsRef) {.importc: "LLVMDisposePassBuilderOptions".}
proc run_passes*(module: ModuleRef, passes: cstring, machine: TargetMachineRef, opts: PassBuilderOptionsRef): ErrorRef {.importc: "LLVMRunPasses".}
{.pop.}

{.push header: "<llvm-c/Error.h>".}
proc get_error_message*(err: ErrorRef): cstring {.importc: "LLVMGetErrorMessage".}
proc dispose_error_message*(msg: cstring) {.importc: "LLVMDisposeErrorMessage".}
{.pop.}

when defined(exprgrad_fast_math):
  {.link: "llvm_ext.o".}
  proc enable_fast_math*(builder: BuilderRef) {.importc: "LLVMBuilderEnableFastMath".}
else:
  {.warning: "Please compile the file exprgrad/wrappers/llvm_ext.cpp and pass -d:exprgrad_fast_math to enable fast floating point math.".}
  proc enableFastMath*(builder: BuilderRef) = discard

when isMainModule:
  let
    module = module_create_with_name("my_module")
    sum = add_function(module, "sum", functionType(int32_type(), [
      int32_type(), int32_type()
    ]))
    entry = sum.append_basic_block("entry")
    builder = create_builder()
  builder.position_builder_at_end(entry)
  let
    a = builder.build_add(sum.get_param(0), sum.get_param(1), "")
    b = builder.build_add(a, const_int(int32_type(), cast[culonglong](clonglong(-1)), LlvmBool(ord(true))), "")
  builder.build_ret(b)
  
  module.write_bitcode_to_file("my_module.bc")
  
  var err: cstring
  echo module.verify_module(AbortProcessAction, err.addr)
  echo err
  dispose_message(err)
  
  var engine: ExecutionEngineRef
  link_in_mcjit()
  initialize_native_target()
  initialize_native_asm_printer()
  err = nil
  if create_execution_engine_for_module(engine.addr, module, err.addr) != 0:
    discard
  if err != nil:
    echo err
    err.dispose_message()
    quit ""
  
  let myFunc = getFunctionAddress[proc (a, b: cint): cint {.noconv.}](engine, "sum")
  echo myFunc(1, 2)
  
  dispose_builder(builder)
  dispose_execution_engine(engine)
