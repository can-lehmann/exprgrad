# Copyright 2022 Can Joshua Lehmann
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

# Domain specific language for constructing the intermediate representation for a kernel

import ir, parser

export Scalar, Index, Boolean, literal, iteratorLiteral

template defineBinop(ArgType, ResultType: typedesc, op, instrKind: untyped) =
  proc op*(a, b: ArgType): ResultType =
    result = ResultType(ExprBuilder(kind: ExprInstr,
      instr: instrKind,
      children: @[ExprBuilder(a), ExprBuilder(b)]
    ))

template defineUnop(ArgType, ResultType: typedesc, op, instrKind: untyped) =
  proc op*(a: ArgType): ResultType =
    result = ResultType(ExprBuilder(kind: ExprInstr,
      instr: instrKind,
      children: @[ExprBuilder(a)]
    ))

template defineType(Type: typedesc) =
  proc literal*(value: Type): Type = value
  defineBinop(Type, Boolean, `==`, InstrEq)

template defineNumber(Type: typedesc) {.dirty.} =
  defineBinop(Type, Type, `+`, InstrAdd)
  defineBinop(Type, Type, `-`, InstrSub)
  defineBinop(Type, Type, `*`, InstrMul)
  defineUnop(Type, Type, `-`, InstrNegate)
  
  defineBinop(Type, Boolean, `<`, InstrLt)
  defineBinop(Type, Boolean, `<=`, InstrLe)

defineType(Boolean)
defineBinop(Boolean, Boolean, `and`, InstrAnd)
defineBinop(Boolean, Boolean, `or`, InstrAnd)

defineType(Scalar)
defineNumber(Scalar)
defineBinop(Scalar, Scalar, `/`, InstrDiv)
defineUnop(Scalar, Scalar, sin, InstrSin)
defineUnop(Scalar, Scalar, cos, InstrCos)
defineUnop(Scalar, Scalar, exp, InstrExp)
defineBinop(Scalar, Scalar, pow, InstrPow)
defineUnop(Scalar, Scalar, sqrt, InstrSqrt)
defineBinop(Scalar, Scalar, log, InstrLog)
defineUnop(Scalar, Scalar, log10, InstrLog10)
defineUnop(Scalar, Scalar, log2, InstrLog2)
defineUnop(Scalar, Scalar, ln, InstrLn)

defineType(Index)
defineNumber(Index)
defineBinop(Index, Index, `div`, InstrIndexDiv)
defineBinop(Index, Index, `mod`, InstrMod)
defineBinop(Index, Index, wrap, InstrWrap)

defineUnop(Scalar, Index, toIndex, InstrToIndex)
defineUnop(Index, Scalar, toScalar, InstrToScalar)

proc epoch*(): Index =
  result = Index(ExprBuilder(kind: ExprInstr, instr: InstrEpoch))

proc select*[T: Index | Scalar | Boolean | Array](cond: Boolean, a, b: T): T =
  result = T(ExprBuilder(kind: ExprInstr,
    instr: InstrSelect,
    children: @[ExprBuilder(cond), ExprBuilder(a), ExprBuilder(b)]
  ))

proc `[]`*[T](arr: Array[T], index: Index): T =
  result = T(ExprBuilder(kind: ExprInstr,
    instr: InstrArrayRead,
    children: @[ExprBuilder(arr), ExprBuilder(index)]
  ))

proc len*[T](arr: Array[T]): Index =
  result = Index(ExprBuilder(kind: ExprInstr,
    instr: InstrArrayLen,
    children: @[ExprBuilder(arr)]
  ))

proc `[]`*(tensor: Fun, indices: varargs[Index]): Scalar =
  let builder = ExprBuilder(kind: ExprRead, tensor: tensor)
  for index in indices:
    builder.children.add(ExprBuilder(index))
  result = Scalar(builder)

proc `{}`*(tensor: Fun, index: Index): Scalar =
  let builder = ExprBuilder(kind: ExprRead,
    isRaw: true,
    tensor: tensor,
    children: @[ExprBuilder(index)]
  )
  result = Scalar(builder)

type TensorShape = object
  tensor: Fun

proc shape*(tensor: Fun): TensorShape =
  result = TensorShape(tensor: tensor)

proc `[]`*(shape: TensorShape, dim: int): Index =
  result = Index(ExprBuilder(
    kind: ExprInstr, instr: InstrShape,
    tensor: shape.tensor,
    dim: dim
  ))

proc `[]`*(shape: TensorShape, dim: BackwardsIndex): Index =
  result = Index(ExprBuilder(
    kind: ExprInstr, instr: InstrShape,
    tensor: shape.tensor,
    dim: -int(dim)
  ))

proc len*(shape: TensorShape): Index =
  result = Index(ExprBuilder(kind: ExprInstr, instr: InstrShapeLen, tensor: shape.tensor))

proc len*(tensor: Fun): Index =
  result = Index(ExprBuilder(kind: ExprInstr, instr: InstrLen, tensor: tensor))

proc sq*[T: Scalar | Index](x: T): T =
  result = x * x

proc max*(x, y: Scalar): Scalar =
  result = select(x > y, x, y)

proc min*(x, y: Scalar): Scalar =
  result = select(x < y, x, y)

converter toBoolean*(x: bool): Boolean = literal(x)
converter toIndex*(x: int): Index = literal(x)
converter toScalar*(x: float): Scalar = literal(x)
