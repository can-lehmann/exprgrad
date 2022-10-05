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

# A simple tensor library

import math, sequtils, random

type
  Tensor*[T] = ref TensorObj[T]
  TensorObj[T] = object
    shape*: seq[int]
    len*: int
    isView*: bool
    data*: ptr UncheckedArray[T]

when defined(gcDestructors):
  proc `=destroy`[T](obj: var TensorObj[T]) =
    if not obj.data.isNil and not obj.isView:
      dealloc(obj.data)
else:
  proc finalizer[T](tensor: Tensor[T]) =
    if not tensor.isNil and not tensor.data.isNil and not tensor.isView:
      dealloc(tensor.data)

proc allocShape*[T](tensor: Tensor[T], shape: openArray[int], fillZero: static[bool] = true) {.inline.} =
  tensor.shape = newSeq[int](shape.len)
  var len = 1
  for it in 0..<shape.len:
    tensor.shape[it] = shape[it]
    len *= shape[it]
  if len != tensor.len or tensor.isView:
    if not tensor.data.isNil and not tensor.isView:
      dealloc(tensor.data)
    tensor.data = cast[ptr UncheckedArray[T]](alloc(sizeof(T) * len))
    assert cast[int](tensor.data) mod sizeof(T) == 0
    tensor.len = len
  if fillZero:
    zeroMem(tensor.data[0].addr, sizeof(T) * len)

proc allocTensor*[T](): Tensor[T] =
  when defined(gcDestructors):
    result = Tensor[T]()
  else:
    new(result, finalizer=finalizer)

proc newTensor*[T](shape: openArray[int]): Tensor[T] =
  result = allocTensor[T]()
  result.allocShape(shape)

proc newTensor*[T](shape: openArray[int], data: seq[T]): Tensor[T] =
  assert shape.prod() == data.len
  result = allocTensor[T]()
  result.allocShape(shape, fillZero = false)
  copyMem(result.data[0].addr, data[0].unsafeAddr, sizeof(T) * data.len)

proc newTensor*[T](shape: openArray[int], value: T): Tensor[T] =
  result = allocTensor[T]()
  result.allocShape(shape, fillZero=false)
  for it in 0..<result.len:
    result.data[it] = value

proc new*[T](_: typedesc[Tensor[T]], shape: openArray[int]): Tensor[T] = newTensor[T](shape)
proc new*[T](_: typedesc[Tensor[T]], shape: openArray[int], data: seq[T]): Tensor[T] = newTensor(shape, data)
proc new*[T](_: typedesc[Tensor[T]], shape: openArray[int], value: T): Tensor[T] = newTensor(shape, value)

proc newRandTensor*[T](shape: openArray[int], slice: HSlice[T, T]): Tensor[T] =
  result = newTensor[T](shape)
  for it in 0..<result.shape.prod:
    result.data[it] = rand(slice)

proc rand*[T](_: typedesc[Tensor[T]], shape: openArray[int], slice: HSlice[T, T]): Tensor[T] =
  result = newRandTensor(shape, slice)

proc clone*[T](tensor: Tensor[T]): Tensor[T] =
  result = Tensor[T]()
  result.allocShape(tensor.shape)
  copyMem(result.data[0].addr, tensor.data[0].addr, sizeof(T) * tensor.len)

proc dataPtr*[T](tensor: Tensor[T]): ptr UncheckedArray[T] {.inline.} = tensor.data

proc `==`*[T](a, b: Tensor[T]): bool =
  if a.len == b.len and a.shape == b.shape:
    for it in 0..<a.len:
      if a.data[it] != b.data[it]:
        return false
    result = true

{.push inline.}
proc `{}`*[T](tensor: Tensor[T], it: int): var T = tensor.data[it]
proc `{}=`*[T](tensor: Tensor[T], it: int, value: T) = tensor.data[it] = value

proc `[]`*[T](tensor: Tensor[T], it: int): var T = tensor.data[it]
proc `[]=`*[T](tensor: Tensor[T], it: int, value: T) = tensor.data[it] = value

proc `[]`*[T](tensor: Tensor[T], y, x: int): var T =
  tensor.data[x + y * tensor.shape[^1]]
proc `[]=`*[T](tensor: Tensor[T], y, x: int, value: T) =
  tensor.data[x + y * tensor.shape[^1]] = value

proc `[]`*[T](tensor: Tensor[T], z, y, x: int): var T =
  tensor.data[x + y * tensor.shape[^1] + z * tensor.shape[^1] * tensor.shape[^2]]
proc `[]=`*[T](tensor: Tensor[T], z, y, x: int, value: T) =
  tensor.data[x + y * tensor.shape[^1] + z * tensor.shape[^1] * tensor.shape[^2]] = value

proc `[]`*[T](tensor: Tensor[T], w, z, y, x: int): var T =
  tensor.data[
    x +
    y * tensor.shape[^1] +
    z * tensor.shape[^2] * tensor.shape[^1] +
    w * tensor.shape[^3] * tensor.shape[^2] * tensor.shape[^1]
  ]
proc `[]=`*[T](tensor: Tensor[T], w, z, y, x: int, value: T) =
  tensor.data[
    x +
    y * tensor.shape[^1] +
    z * tensor.shape[^2] * tensor.shape[^1] +
    w * tensor.shape[^3] * tensor.shape[^2] * tensor.shape[^1]
  ] = value
{.pop.}

proc stringify[T](tensor: Tensor[T], dim: int, index: var int): string =
  if dim >= tensor.shape.len:
    result = $tensor.data[index]
    index += 1
  else:
    result = "["
    for it in 0..<tensor.shape[dim]:
      if it != 0:
        result &= ", "
      result &= tensor.stringify(dim + 1, index)
    result &= "]"

proc `$`*[T](tensor: Tensor[T]): string =
  if tensor.isNil:
    result = "nil"
  else:
    var index = 0
    result = tensor.stringify(0, index)

template defineElementwise(op) =
  proc op*[T](a, b: Tensor[T]): Tensor[T] =
    assert a.shape == b.shape
    result = newTensor[T](a.shape)
    for it in 0..<a.len:
      result.data[it] = op(a.data[it], b.data[it])

defineElementwise(`+`)
defineElementwise(`-`)
defineElementwise(min)
defineElementwise(max)

template defineElementwiseUnary(op) =
  proc op*[T](tensor: Tensor[T]): Tensor[T] =
    result = newTensor[T](tensor.shape)
    for it in 0..<tensor.len:
      result.data[it] = op(tensor.data[it])

defineElementwiseUnary(abs)
defineElementwiseUnary(`-`)

template defineElementwiseMut(op) =
  proc op*[T](a, b: Tensor[T]) =
    assert a.shape == b.shape
    for it in 0..<a.len:
      op(a.data[it], b.data[it])

defineElementwiseMut(`+=`)
defineElementwiseMut(`-=`)

template defineScalarOp(op) =
  proc op*[T](a: Tensor[T], b: T): Tensor[T] =
    result = newTensor[T](a.shape)
    for it in 0..<a.len:
      result.data[it] = op(a.data[it], b)
  
  proc op*[T](a: T, b: Tensor[T]): Tensor[T] =
    result = newTensor[T](b.shape)
    for it in 0..<b.len:
      result.data[it] = op(a, b.data[it])

defineScalarOp(`*`)
defineScalarOp(`/`)
defineScalarOp(`div`)
defineScalarOp(min)
defineScalarOp(max)

template defineReduce(name, op, initial) =
  proc name*[T](tensor: Tensor[T]): T =
    result = T(initial)
    for it in 0..<tensor.len:
      result = op(result, tensor.data[it])

defineReduce(sum, `+`, 0)
defineReduce(prod, `*`, 1)

template defineReduce(name, op) =
  proc name*[T](tensor: Tensor[T]): T =
    assert tensor.len > 0
    result = T(tensor.data[0])
    for it in 1..<tensor.len:
      result = op(result, tensor.data[it])

defineReduce(min, min)
defineReduce(max, max)

proc squares*[T](tensor: Tensor[T]): Tensor[T] =
  result = newTensor[T](tensor.shape)
  for it in 0..<tensor.len:
    result.data[it] = tensor.data[it] * tensor.data[it]

proc remap*[T](tensor: Tensor[T], fromMin, fromMax, toMin, toMax: T): Tensor[T] =
  result = newTensor[T](tensor.shape)
  let
    fromSize = fromMax - fromMin
    toSize = toMax - toMin
  for it in 0..<tensor.len:
    result.data[it] = (tensor.data[it] - fromMin) / fromSize * toSize + toMin

proc isMatrix*[T](tensor: Tensor[T]): bool {.inline.} =
  result = tensor.shape.len == 2

proc `*`*[T](a, b: Tensor[T]): Tensor[T] =
  ## Matrix multiplication
  assert a.isMatrix and b.isMatrix
  assert a.shape[1] == b.shape[0]
  result = newTensor[T]([a.shape[0], b.shape[1]])
  for y in 0..<result.shape[0]:
    for it in 0..<a.shape[1]:
      for x in 0..<result.shape[1]:
        result[y, x] += a[y, it] * b[it, x]

proc transpose*[T](tensor: Tensor[T]): Tensor[T] =
  assert tensor.isMatrix
  result = newTensor[T]([tensor.shape[1], tensor.shape[0]])
  for y in 0..<result.shape[0]:
    for x in 0..<result.shape[1]:
      result[y, x] = tensor[x, y]

proc convert*[A, B](tensor: Tensor[A]): Tensor[B] =
  result = newTensor[B](tensor.shape)
  for it in 0..<tensor.len:
    result[it] = B(tensor.data[it])

template convert*[A](tensor: Tensor[A], B: typedesc): Tensor[B] =
  convert[A, B](tensor)

proc oneHot*[T](indices: Tensor[T], count: int): Tensor[T] =
  result = newTensor[T](indices.shape.toSeq() & @[count])
  for it in 0..<indices.len:
    result.data[it * count + int(indices.data[it])] = T(1)

proc fillZero*[T](tensor: Tensor[T]) {.inline.} =
  if tensor.len > 0:
    zeroMem(tensor.data[0].addr, sizeof(T) * tensor.len)

proc fill*[T](tensor: Tensor[T], value: T) {.inline.} =
  for it in 0..<tensor.len:
    tensor.data[it] = value

proc fillRand*[T](tensor: Tensor[T], slice: HSlice[T, T]) {.inline.} =
  for it in 0..<tensor.len:
    tensor.data[it] = rand(slice)

proc viewFirst*[T](tensor: Tensor[T], offset, size: int): Tensor[T] =
  let stride = tensor.len div tensor.shape[0]
  result = Tensor[T](
    isView: true,
    shape: @[size] & tensor.shape[1..^1],
    len: stride * size,
    data: cast[ptr UncheckedArray[T]](tensor.data[stride * offset].addr)
  )

proc viewFirst*[T](tensor: Tensor[T], slice: HSlice[int, int]): Tensor[T] =
  result = tensor.viewFirst(slice.a, slice.b - slice.a + 1)

proc selectSamples*[T](tensor: Tensor[T], idx: openArray[int]): Tensor[T] =
  result = newTensor[T](@[idx.len] & tensor.shape[1..^1])
  let stride = result.len div idx.len
  for it, id in idx:
    copyMem(
      result.data[it * stride].addr,
      tensor.data[id * stride].addr,
      stride * sizeof(T)
    )

proc selectRandomSamples*[T](tensor: Tensor[T], count: int): Tensor[T] =
  var idx = newSeq[int](count)
  for id in idx.mitems:
    id = rand(0..<tensor.shape[0])
  result = tensor.selectSamples(idx)

proc shuffleXy*[T](x, y: Tensor[T]): tuple[x, y: Tensor[T]] =
  assert x.shape[0] == y.shape[0]
  var idx = newSeq[int](x.shape[0])
  for it in 0..<idx.len:
    idx[it] = it
  shuffle(idx)
  result.x = x.selectSamples(idx)
  result.y = y.selectSamples(idx)

proc shuffleXy*[T](tensors: (Tensor[T], Tensor[T])): tuple[x, y: Tensor[T]] =
  result = shuffleXy(tensors[0], tensors[1])

proc concatFirst*[T](a, b: Tensor[T]): Tensor[T] =
  assert a.shape[1..^1] == b.shape[1..^1]
  result = newTensor[T](@[a.shape[0] + b.shape[0]] & a.shape[1..^1])
  let stride = result.len div result.shape[0]
  for it in 0..<a.shape[0]:
    copyMem(
      result.data[it * stride].addr,
      a.data[it * stride].addr,
      stride * sizeof(T)
    )
  for it in 0..<b.shape[0]:
    copyMem(
      result.data[(it + a.shape[0]) * stride].addr,
      b.data[it * stride].addr,
      stride * sizeof(T)
    )

proc reshape*[T](tensor: Tensor[T], shape: openArray[int]): Tensor[T] =
  var
    len = 1
    targetShape = newSeq[int](shape.len)
    varDim = -1
  for dim, size in shape:
    if size < 0:
      varDim = dim
    else:
      len *= size
      targetShape[dim] = size
  if varDim != -1:
    targetShape[varDim] = tensor.len div len
  
  assert targetShape.prod() == tensor.len
  result = allocTensor[T]()
  result.allocShape(targetShape, fillZero = false)
  copyMem(result.data[0].addr, tensor.data[0].addr, result.len * sizeof(T))
