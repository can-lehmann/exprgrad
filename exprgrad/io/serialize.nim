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

# Save and load models to/from disk

import std/[macros, tables, sets]
import ../tensors, ../ir, ../model
import faststreams

proc store*[T: SomeUnsignedInt](stream: var WriteStream, value: T) =
  var cur = value
  for it in 0..<sizeof(T):
    stream.write(uint8(cur and 0xff))
    cur = cur shr 8

proc store*(stream: var WriteStream, x: bool) = stream.write(uint8(ord(x)))
proc store*(stream: var WriteStream, x: char) = stream.write(x)
proc store*(stream: var WriteStream, x: uint8) = stream.write(x)
proc store*(stream: var WriteStream, x: int8) = stream.write(cast[uint8](x))
proc store*(stream: var WriteStream, x: int32) = stream.store(cast[uint32](x))
proc store*(stream: var WriteStream, x: int64) = stream.store(cast[uint64](x))
proc store*(stream: var WriteStream, x: int16) = stream.store(cast[uint16](x))
proc store*(stream: var WriteStream, x: float32) = stream.store(cast[uint32](x))
proc store*(stream: var WriteStream, x: float64) = stream.store(cast[uint64](x))

proc store*(stream: var WriteStream, value: int) =
  stream.store(cast[uint64](int64(value)))

proc store*(stream: var WriteStream, str: string) =
  stream.store(int64(str.len))
  for chr in str:
    stream.store(chr)

proc store*[T](stream: var WriteStream, sequence: seq[T]) =
  mixin store
  stream.store(int64(sequence.len))
  for item in sequence.items:
    stream.store(item)

proc store*[T](stream: var WriteStream, hash_set: HashSet[T]) =
  mixin store
  stream.store(int64(hash_set.len))
  for item in hash_set.items:
    stream.store(item)

proc store*[T](stream: var WriteStream, values: set[T]) =
  mixin store
  stream.store(int64(values.len))
  for item in values.items:
    stream.store(item)

proc store*[K, V](stream: var WriteStream, tab: Table[K, V]) =
  mixin store
  stream.store(int64(tab.len))
  for key, value in tab.pairs:
    stream.store(key)
    stream.store(value)

proc store*[T](stream: var WriteStream, tensor: Tensor[T]) =
  stream.store(tensor.is_nil)
  if not tensor.is_nil:
    stream.store(tensor.shape)
    for it in 0..<tensor.len:
      stream.store(tensor.data[it])

proc store*[A, B](stream: var WriteStream, slice: HSlice[A, B]) =
  mixin store
  stream.store(slice.a)
  stream.store(slice.b)

proc read_uint[T](stream: var ReadStream): T =
  for it in 0..<sizeof(T):
    result = result xor (T(stream.read_uint8()) shl (it * 8))

proc load*[T: SomeUnsignedInt](stream: var ReadStream, value: var T) =
  value = read_uint[T](stream)

proc load*(stream: var ReadStream, x: var bool) = x = bool(stream.read_uint8())
proc load*(stream: var ReadStream, x: var char) = x = stream.read_char()
proc load*(stream: var ReadStream, x: var uint8) = x = stream.read_uint8()
proc load*(stream: var ReadStream, x: var int8) = x = cast[int8](x)

proc load*(stream: var ReadStream, value: var int16) =
  value = cast[int16](read_uint[uint16](stream))

proc load*(stream: var ReadStream, value: var int32) =
  value = cast[int32](read_uint[uint32](stream))

proc load*(stream: var ReadStream, value: var int64) =
  value = cast[int64](read_uint[uint64](stream))

proc load*(stream: var ReadStream, value: var float32) =
  value = cast[float32](read_uint[uint32](stream))

proc load*(stream: var ReadStream, value: var float64) =
  value = cast[float64](read_uint[uint64](stream))

proc load*(stream: var ReadStream, value: var int) =
  value = int(cast[int64](read_uint[uint64](stream)))

proc load*(stream: var ReadStream, str: var string) =
  var len: int64 = 0
  stream.load(len)
  str = new_string(len)
  for it in 0..<len:
    stream.load(str[it])

proc load*[T](stream: var ReadStream, sequence: var seq[T]) =
  mixin load
  var len: int64 = 0
  stream.load(len)
  sequence = new_seq[T](len)
  for it in 0..<len:
    stream.load(sequence[it])

proc load*[T](stream: var ReadStream, values: var set[T]) =
  mixin load
  var len: int64 = 0
  stream.load(len)
  values = {}
  for it in 0..<len:
    var value: T
    stream.load(value)
    values.incl(value)

proc load*[T](stream: var ReadStream, hash_set: var HashSet[T]) =
  mixin load
  var len: int64 = 0
  stream.load(len)
  hash_set = init_hash_set[T]()
  for it in 0..<len:
    var item: T
    stream.load(item)
    hash_set.incl(item)

proc load*[K, V](stream: var ReadStream, tab: var Table[K, V]) =
  mixin load
  var len: int64 = 0
  stream.load(len)
  tab = init_table[K, V]()
  for it in 0..<len:
    var
      key: K
      val: V
    stream.load(key)
    stream.load(val)
    tab[key] = val

proc load*[T](stream: var ReadStream, tensor: var Tensor[T]) =
  var is_nil = false
  stream.load(is_nil)
  if is_nil:
    tensor = nil
  else:
    var shape: seq[int] = @[]
    stream.load(shape)
    tensor = new_tensor[T](shape)
    for it in 0..<tensor.len:
      stream.load(tensor.data[it])

proc load*[A, B](stream: var ReadStream, slice: var HSlice[A, B]) =
  mixin load
  stream.load(slice.a)
  stream.load(slice.b)

proc is_name(node: NimNode): bool =
  result = node.kind == nnkIdent or node.kind == nnkSym

proc unwrap_name(node: NimNode): NimNode =
  result = node
  while not result.is_name():
    case result.kind:
      of nnkPostfix: result = result[1]
      else: raise new_exception(ValueError, "Unable to unwrap name from " & $node.kind)

proc new_dot_expr(node: NimNode, field: string): NimNode =
  result = new_tree(nnkDotExpr, node, ident(field))

proc collect_discriminants(node: NimNode, into: var seq[NimNode]) =
  if node.kind == nnkRecCase:
    into.add(node[0])
  for child in node:
    child.collect_discriminants(into) 

proc collect_discriminants(node: NimNode): seq[NimNode] =
  node.collect_discriminants(result)

type SerializeKind = enum
  SerializeStore, SerializeLoad

proc gen_serialize(node, decls, stmts, typ: NimNode, kind: SerializeKind) =
  let
    stream = ident("stream")
    value = ident("value")
    kind_name = ident(["store", "load"][ord(kind)])
  case node.kind:
    of nnkTypeDef:
      let body = new_stmt_list()
      node[2].gen_serialize(decls, body, node[0], kind)
      var typ_name = node[0]
      if kind == SerializeLoad:
        typ_name = new_tree(nnkVarTy, typ_name)
      let stream_typ = case kind:
        of SerializeStore: bind_sym("WriteStream")
        of SerializeLoad: bind_sym("ReadStream")
      stmts.add: quote:
        proc `kind_name`*(`stream`: var `stream_typ`, `value`: `typ_name`) =
          `body`
      decls.add: quote:
        proc `kind_name`*(`stream`: var `stream_typ`, `value`: `typ_name`)
      discard # stmts.add returns a NimNode
    of nnkDistinctTy:
      let base_typ = node[0]
      case kind:
        of SerializeStore:
          stmts.add: quote:
            `kind_name`(`stream`, `base_typ`(`value`))
        of SerializeLoad:
          let tmp = gensym(nskVar, "tmp")
          stmts.add(new_tree(nnkVarSection, new_ident_defs(tmp, base_typ)))
          stmts.add(new_call(kind_name, stream, tmp))
          stmts.add(new_assignment(value, new_call(typ, tmp)))
    of nnkEnumTy:
      case kind:
        of SerializeStore:
          stmts.add: quote:
            `kind_name`(`stream`, int64(ord(`value`)))
        of SerializeLoad:
          stmts.add: quote:
            var id: int64
            `kind_name`(`stream`, id)
            `value` = `typ`(id)
    of nnkObjectTy:
      let discrs = node[2].collect_discriminants()
      case kind:
        of SerializeStore:
          for discr in discrs:
            let name = ident(discr[0].unwrap_name().str_val)
            stmts.add: quote:
              `kind_name`(`stream`, `value`.`name`)
        of SerializeLoad:
          let constr = new_tree(nnkObjConstr, typ)
          for discr in discrs:
            let
              load_name = gensym(nskVar, "discr")
              discr_typ = discr[^2]
              name = ident(discr[0].unwrap_name().str_val)
            stmts.add: quote:
              var `load_name`: `discr_typ`
              `kind_name`(`stream`, `load_name`)
            constr.add(new_tree(nnkExprColonExpr, name, load_name))
          stmts.add(new_assignment(value, constr))
      node[2].gen_serialize(decls, stmts, typ, kind)
    of nnkRecList:
      for defs in node:
        defs.gen_serialize(decls, stmts, typ, kind)
    of nnkIdentDefs:
      let field_typ = node[^2]
      for it in 0..<(node.len - 2):
        let name = node[it].unwrap_name().str_val
        stmts.add(new_call(kind_name,
          stream, value.new_dot_expr(name)
        ))
    of nnkRecCase:
      var case_stmt = new_tree(nnkCaseStmt,
        value.new_dot_expr(node[0][0].unwrap_name().str_val)
      )
      for it in 1..<node.len:
        var body = new_stmt_list()
        node[it][^1].gen_serialize(decls, body, typ, kind)
        case node[it].kind:
          of nnkOfBranch:
            case_stmt.add(new_tree(nnkOfBranch, node[it][0], body))
          of nnkElse: 
            case_stmt.add(new_tree(nnkElse, body))
          else: raise new_exception(ValueError, "")
      stmts.add(case_stmt)
    of nnkNilLit: discard
    of nnkRefTy:
      var body = new_stmt_list()
      node[0].gen_serialize(decls, body, typ, kind)
      case kind:
        of SerializeStore:
          stmts.add: quote:
            store(`stream`, is_nil(`value`))
            if not is_nil(`value`):
              `body`
        of SerializeLoad:
          stmts.add: quote:
            var is_nil: bool
            load(`stream`, is_nil)
            if is_nil:
              `value` = nil
            else:
              `body`
    else:
      echo node.tree_repr
      raise new_exception(ValueError, "Unable to generate load for " & $node.kind)

proc gen_serialize(node, decls, impls: NimNode) =
  case node.kind:
    of nnkSym:
      let impl = node.get_impl()
      impl.gen_serialize(decls, impls, nil, SerializeLoad)
      impl.gen_serialize(decls, impls, nil, SerializeStore)
    of nnkBracket:
      for child in node.children:
        child.gen_serialize(decls, impls)
    else:
      raise new_exception(ValueError, "Unable to generate load/store from " & $node.kind)

macro serializable*(args: typed): untyped =
  var
    decls = new_stmt_list()
    impls = new_stmt_list()
  args.gen_serialize(decls, impls)
  result = new_stmt_list([decls, impls])

serializable([KernelId, RegId, TensorId, LoopId])
serializable([TypeKind, Type])
serializable([LoopMode, ParallelClosure, GpuIndex])
serializable([InstrKind, Instr])
serializable([Register, Expr, LinearIndex])
serializable([LoopSchedule, Loop])
serializable([TensorSchedule, Interval, LocalCache, TensorOpKind, TensorOp])
serializable([ShapeConstrKind, ShapeConstraint])
serializable([GenKind, Generator])
serializable([KernelGradient, Kernel, CompileTarget, Target])
serializable([TensorKind, TensorDef])
serializable([ScalarType, Stage, Program])

proc store*[T](stream: var WriteStream, model: Model[T]) =
  stream.store(model.is_nil)
  if not model.is_nil:
    stream.store(model.program)
    stream.store(model.params)
    stream.store(model.caches)

proc load*[T](stream: var ReadStream, model: var Model[T]) =
  var is_nil = false
  stream.load(is_nil)
  if is_nil:
    model = nil
  else:
    var
      program: Program
      params: Table[TensorId, Tensor[T]]
      caches: Table[TensorId, Tensor[T]]
    stream.load(program)
    stream.load(params)
    stream.load(caches)
    model = new_model[T](program, params, caches)

proc save*[T](value: T, path: string) =
  var stream = open_write_stream(path)
  defer: stream.close()
  stream.store(value)

proc load_tensor*[T](path: string): Tensor[T] =
  var stream = open_read_stream(path)
  defer: stream.close()
  stream.load(result)

proc load_model*[T](path: string): Model[T] =
  var stream = open_read_stream(path)
  defer: stream.close()
  stream.load(result)
