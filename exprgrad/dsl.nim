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

export Scalar, Index, Boolean, literal

template define_binop(ArgType, ResultType: typedesc, op, instr_kind: untyped) =
  proc op*(a, b: ArgType): ResultType =
    result = ResultType(ExprBuilder(kind: ExprInstr,
      instr: instr_kind,
      children: @[ExprBuilder(a), ExprBuilder(b)]
    ))

template define_unop(ArgType, ResultType: typedesc, op, instr_kind: untyped) =
  proc op*(a: ArgType): ResultType =
    result = ResultType(ExprBuilder(kind: ExprInstr,
      instr: instr_kind,
      children: @[ExprBuilder(a)]
    ))

template define_type(Type: typedesc) =
  proc literal*(value: Type): Type = value
  define_binop(Type, Boolean, `==`, InstrEq)

template define_number(Type: typedesc) {.dirty.} =
  define_binop(Type, Type, `+`, InstrAdd)
  define_binop(Type, Type, `-`, InstrSub)
  define_binop(Type, Type, `*`, InstrMul)
  define_unop(Type, Type, `-`, InstrNegate)
  
  define_binop(Type, Boolean, `<`, InstrLt)
  define_binop(Type, Boolean, `<=`, InstrLe)

define_type(Boolean)

define_type(Scalar)
define_number(Scalar)
define_binop(Scalar, Scalar, `/`, InstrDiv)
define_unop(Scalar, Scalar, sin, InstrSin)
define_unop(Scalar, Scalar, cos, InstrCos)
define_unop(Scalar, Scalar, exp, InstrExp)
define_binop(Scalar, Scalar, pow, InstrPow)
define_unop(Scalar, Scalar, sqrt, InstrSqrt)
define_binop(Scalar, Scalar, log, InstrLog)
define_unop(Scalar, Scalar, log10, InstrLog10)
define_unop(Scalar, Scalar, log2, InstrLog2)
define_unop(Scalar, Scalar, ln, InstrLn)

define_type(Index)
define_number(Index)
define_binop(Index, Index, `div`, InstrIndexDiv)
define_binop(Index, Index, `mod`, InstrMod)
define_binop(Index, Index, wrap, InstrWrap)

define_unop(Scalar, Index, to_index, InstrToIndex)
define_unop(Index, Scalar, to_scalar, InstrToScalar)

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
    is_raw: true,
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

converter to_boolean*(x: bool): Boolean = literal(x)
converter to_index*(x: int): Index = literal(x)
converter to_scalar*(x: float): Scalar = literal(x)
