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

proc store*[T](stream: var WriteStream, hashSet: HashSet[T]) =
  mixin store
  stream.store(int64(hashSet.len))
  for item in hashSet.items:
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
  stream.store(tensor.isNil)
  if not tensor.isNil:
    stream.store(tensor.shape)
    for it in 0..<tensor.len:
      stream.store(tensor.data[it])

proc store*[A, B](stream: var WriteStream, slice: HSlice[A, B]) =
  mixin store
  stream.store(slice.a)
  stream.store(slice.b)

proc readUint[T](stream: var ReadStream): T =
  for it in 0..<sizeof(T):
    result = result xor (T(stream.readUint8()) shl (it * 8))

proc load*[T: SomeUnsignedInt](stream: var ReadStream, value: var T) =
  value = readUint[T](stream)

proc load*(stream: var ReadStream, x: var bool) = x = bool(stream.readUint8())
proc load*(stream: var ReadStream, x: var char) = x = stream.readChar()
proc load*(stream: var ReadStream, x: var uint8) = x = stream.readUint8()
proc load*(stream: var ReadStream, x: var int8) = x = cast[int8](x)

proc load*(stream: var ReadStream, value: var int16) =
  value = cast[int16](readUint[uint16](stream))

proc load*(stream: var ReadStream, value: var int32) =
  value = cast[int32](readUint[uint32](stream))

proc load*(stream: var ReadStream, value: var int64) =
  value = cast[int64](readUint[uint64](stream))

proc load*(stream: var ReadStream, value: var float32) =
  value = cast[float32](readUint[uint32](stream))

proc load*(stream: var ReadStream, value: var float64) =
  value = cast[float64](readUint[uint64](stream))

proc load*(stream: var ReadStream, value: var int) =
  value = int(cast[int64](readUint[uint64](stream)))

proc load*(stream: var ReadStream, str: var string) =
  var len: int64 = 0
  stream.load(len)
  str = newString(len)
  for it in 0..<len:
    stream.load(str[it])

proc load*[T](stream: var ReadStream, sequence: var seq[T]) =
  mixin load
  var len: int64 = 0
  stream.load(len)
  sequence = newSeq[T](len)
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

proc load*[T](stream: var ReadStream, hashSet: var HashSet[T]) =
  mixin load
  var len: int64 = 0
  stream.load(len)
  hashSet = initHashSet[T]()
  for it in 0..<len:
    var item: T
    stream.load(item)
    hashSet.incl(item)

proc load*[K, V](stream: var ReadStream, tab: var Table[K, V]) =
  mixin load
  var len: int64 = 0
  stream.load(len)
  tab = initTable[K, V]()
  for it in 0..<len:
    var
      key: K
      val: V
    stream.load(key)
    stream.load(val)
    tab[key] = val

proc load*[T](stream: var ReadStream, tensor: var Tensor[T]) =
  var isNil = false
  stream.load(isNil)
  if isNil:
    tensor = nil
  else:
    var shape: seq[int] = @[]
    stream.load(shape)
    tensor = newTensor[T](shape)
    for it in 0..<tensor.len:
      stream.load(tensor.data[it])

proc load*[A, B](stream: var ReadStream, slice: var HSlice[A, B]) =
  mixin load
  stream.load(slice.a)
  stream.load(slice.b)

proc isName(node: NimNode): bool =
  result = node.kind == nnkIdent or node.kind == nnkSym

proc unwrapName(node: NimNode): NimNode =
  result = node
  while not result.isName():
    case result.kind:
      of nnkPostfix: result = result[1]
      else: raise newException(ValueError, "Unable to unwrap name from " & $node.kind)

proc newDotExpr(node: NimNode, field: string): NimNode =
  result = newTree(nnkDotExpr, node, ident(field))

proc collectDiscriminants(node: NimNode, into: var seq[NimNode]) =
  if node.kind == nnkRecCase:
    into.add(node[0])
  for child in node:
    child.collectDiscriminants(into) 

proc collectDiscriminants(node: NimNode): seq[NimNode] =
  node.collectDiscriminants(result)

type SerializeKind = enum
  SerializeStore, SerializeLoad

proc genSerialize(node, decls, stmts, typ: NimNode, kind: SerializeKind) =
  let
    stream = ident("stream")
    value = ident("value")
    kindName = ident(["store", "load"][ord(kind)])
  case node.kind:
    of nnkTypeDef:
      let body = newStmtList()
      node[2].genSerialize(decls, body, node[0], kind)
      var typName = node[0]
      if kind == SerializeLoad:
        typName = newTree(nnkVarTy, typName)
      let streamTyp = case kind:
        of SerializeStore: bindSym("WriteStream")
        of SerializeLoad: bindSym("ReadStream")
      stmts.add: quote:
        proc `kindName`*(`stream`: var `streamTyp`, `value`: `typName`) =
          `body`
      decls.add: quote:
        proc `kindName`*(`stream`: var `streamTyp`, `value`: `typName`)
      discard # stmts.add returns a NimNode
    of nnkDistinctTy:
      let baseTyp = node[0]
      case kind:
        of SerializeStore:
          stmts.add: quote:
            `kindName`(`stream`, `baseTyp`(`value`))
        of SerializeLoad:
          let tmp = genSym(nskVar, "tmp")
          stmts.add(newTree(nnkVarSection, newIdentDefs(tmp, baseTyp)))
          stmts.add(newCall(kindName, stream, tmp))
          stmts.add(newAssignment(value, newCall(typ, tmp)))
    of nnkEnumTy:
      case kind:
        of SerializeStore:
          stmts.add: quote:
            `kindName`(`stream`, int64(ord(`value`)))
        of SerializeLoad:
          stmts.add: quote:
            var id: int64
            `kindName`(`stream`, id)
            `value` = `typ`(id)
    of nnkObjectTy:
      let discrs = node[2].collectDiscriminants()
      case kind:
        of SerializeStore:
          for discr in discrs:
            let name = ident(discr[0].unwrapName().strVal)
            stmts.add: quote:
              `kindName`(`stream`, `value`.`name`)
        of SerializeLoad:
          let constr = newTree(nnkObjConstr, typ)
          for discr in discrs:
            let
              loadName = genSym(nskVar, "discr")
              discrTyp = discr[^2]
              name = ident(discr[0].unwrapName().strVal)
            stmts.add: quote:
              var `loadName`: `discrTyp`
              `kindName`(`stream`, `loadName`)
            constr.add(newTree(nnkExprColonExpr, name, loadName))
          stmts.add(newAssignment(value, constr))
      node[2].genSerialize(decls, stmts, typ, kind)
    of nnkRecList:
      for defs in node:
        defs.genSerialize(decls, stmts, typ, kind)
    of nnkIdentDefs:
      let fieldTyp = node[^2]
      for it in 0..<(node.len - 2):
        let name = node[it].unwrapName().strVal
        stmts.add(newCall(kindName,
          stream, value.newDotExpr(name)
        ))
    of nnkRecCase:
      var caseStmt = newTree(nnkCaseStmt,
        value.newDotExpr(node[0][0].unwrapName().strVal)
      )
      for it in 1..<node.len:
        var body = newStmtList()
        node[it][^1].genSerialize(decls, body, typ, kind)
        case node[it].kind:
          of nnkOfBranch:
            caseStmt.add(newTree(nnkOfBranch, node[it][0], body))
          of nnkElse: 
            caseStmt.add(newTree(nnkElse, body))
          else: raise newException(ValueError, "")
      stmts.add(caseStmt)
    of nnkNilLit: discard
    of nnkRefTy:
      var body = newStmtList()
      node[0].genSerialize(decls, body, typ, kind)
      case kind:
        of SerializeStore:
          stmts.add: quote:
            store(`stream`, isNil(`value`))
            if not isNil(`value`):
              `body`
        of SerializeLoad:
          stmts.add: quote:
            var isNil: bool
            load(`stream`, isNil)
            if isNil:
              `value` = nil
            else:
              `body`
    else:
      echo node.treeRepr
      raise newException(ValueError, "Unable to generate load for " & $node.kind)

proc genSerialize(node, decls, impls: NimNode) =
  case node.kind:
    of nnkSym:
      let impl = node.getImpl()
      impl.genSerialize(decls, impls, nil, SerializeLoad)
      impl.genSerialize(decls, impls, nil, SerializeStore)
    of nnkBracket:
      for child in node.children:
        child.genSerialize(decls, impls)
    else:
      raise newException(ValueError, "Unable to generate load/store from " & $node.kind)

macro serializable*(args: typed): untyped =
  var
    decls = newStmtList()
    impls = newStmtList()
  args.genSerialize(decls, impls)
  result = newStmtList([decls, impls])

serializable([KernelId, RegId, TensorId, LoopId])
serializable([TypeKind, Type])
serializable([LoopMode, ParallelClosure, GpuIndex])
serializable([InstrKind, Instr])
serializable([Register, Expr, LinearIndex])
serializable([LoopSchedule, Loop])
serializable([TensorSchedule, Interval, LocalCache, OffsetInterval])
serializable([TensorOpKind, TensorOp])
serializable([ShapeConstrPriority, ShapeConstrKind, ShapeConstraint])
serializable([GenKind, Generator])
serializable([KernelGradient, Kernel, CompileTarget, Target])
serializable([TensorKind, TensorDef])
serializable([ScalarType, Stage, Program])

proc store*[T](stream: var WriteStream, model: Model[T]) =
  stream.store(model.isNil)
  if not model.isNil:
    stream.store(model.program)
    stream.store(model.params)
    stream.store(model.caches)

proc load*[T](stream: var ReadStream, model: var Model[T]) =
  var isNil = false
  stream.load(isNil)
  if isNil:
    model = nil
  else:
    var
      program: Program
      params: Table[TensorId, Tensor[T]]
      caches: Table[TensorId, Tensor[T]]
    stream.load(program)
    stream.load(params)
    stream.load(caches)
    model = newModel[T](program, params, caches)

proc save*[T](value: T, path: string) =
  var stream = openWriteStream(path)
  defer: stream.close()
  stream.store(value)

proc loadTensor*[T](path: string): Tensor[T] =
  var stream = openReadStream(path)
  defer: stream.close()
  stream.load(result)

proc loadModel*[T](path: string): Model[T] =
  var stream = openReadStream(path)
  defer: stream.close()
  stream.load(result)
