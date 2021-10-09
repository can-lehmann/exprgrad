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
    is_view*: bool
    data*: ptr UncheckedArray[T]

when defined(gc_destructors):
  proc `=destroy`[T](obj: var TensorObj[T]) =
    if not obj.data.is_nil and not obj.is_view:
      dealloc(obj.data)
else:
  proc finalizer[T](tensor: Tensor[T]) =
    if not tensor.is_nil and not tensor.data.is_nil and not tensor.is_view:
      dealloc(tensor.data)

proc alloc_shape*[T](tensor: Tensor[T], shape: openArray[int], fill_zero: static[bool] = true) {.inline.} =
  tensor.shape = new_seq[int](shape.len)
  var len = 1
  for it in 0..<shape.len:
    tensor.shape[it] = shape[it]
    len *= shape[it]
  if len != tensor.len:
    if not tensor.data.is_nil and not tensor.is_view:
      dealloc(tensor.data)
    tensor.data = cast[ptr UncheckedArray[T]](alloc(sizeof(T) * len))
    assert cast[int](tensor.data) mod sizeof(T) == 0
    tensor.len = len
  if fill_zero:
    zero_mem(tensor.data[0].addr, sizeof(T) * len)

proc alloc_tensor*[T](): Tensor[T] =
  when defined(gc_destructors):
    result = Tensor[T]()
  else:
    new(result, finalizer=finalizer)

proc new_tensor*[T](shape: openArray[int]): Tensor[T] =
  result = alloc_tensor[T]()
  result.alloc_shape(shape)

proc new_tensor*[T](shape: openArray[int], data: seq[T]): Tensor[T] =
  assert shape.prod() == data.len
  result = alloc_tensor[T]()
  result.alloc_shape(shape, fill_zero = false)
  copy_mem(result.data[0].addr, data[0].unsafe_addr, sizeof(T) * data.len)

proc new_tensor*[T](shape: openArray[int], value: T): Tensor[T] =
  result = alloc_tensor[T]()
  result.alloc_shape(shape, fill_zero=false)
  for it in 0..<result.len:
    result.data[it] = value

proc new_rand_tensor*[T](shape: openArray[int], slice: HSlice[T, T]): Tensor[T] =
  result = new_tensor[T](shape)
  for it in 0..<result.shape.prod:
    result.data[it] = rand(slice)

proc data_ptr*[T](tensor: Tensor[T]): ptr UncheckedArray[T] {.inline.} = tensor.data

proc `==`*[T](a, b: Tensor[T]): bool =
  if a.len == b.len and a.shape == b.shape:
    for it in 0..<a.len:
      if a.data[it] != b.data[it]:
        return false
    result = true

proc `{}`*[T](tensor: Tensor[T], it: int): var T {.inline.} = tensor.data[it]
proc `{}=`*[T](tensor: Tensor[T], it: int, value: T) {.inline.} = tensor.data[it] = value

proc `[]`*[T](tensor: Tensor[T], it: int): var T {.inline.} = tensor.data[it]
proc `[]=`*[T](tensor: Tensor[T], it: int, value: T) {.inline.} = tensor.data[it] = value

proc `[]`*[T](tensor: Tensor[T], x, y: int): var T {.inline.} =
  tensor.data[x + y * tensor.shape[0]]
proc `[]=`*[T](tensor: Tensor[T], x, y: int, value: T) {.inline.} =
  tensor.data[x + y * tensor.shape[0]] = value

proc `[]`*[T](tensor: Tensor[T], x, y, z: int): var T {.inline.} =
  tensor.data[x + y * tensor.shape[0] + z * tensor.shape[1] * tensor.shape[0]]
proc `[]=`*[T](tensor: Tensor[T], x, y, z: int, value: T) {.inline.} =
  tensor.data[x + y * tensor.shape[0] + z * tensor.shape[1] * tensor.shape[0]] = value

proc `[]`*[T](tensor: Tensor[T], x, y, z, w: int): var T {.inline.} =
  tensor.data[
    x +
    y * tensor.shape[0] +
    z * tensor.shape[1] * tensor.shape[0] +
    w * tensor.shape[2] * tensor.shape[1] * tensor.shape[0]
  ]
proc `[]=`*[T](tensor: Tensor[T], x, y, z, w: int, value: T) {.inline.} =
  tensor.data[
    x +
    y * tensor.shape[0] +
    z * tensor.shape[1] * tensor.shape[0] +
    w * tensor.shape[2] * tensor.shape[1] * tensor.shape[0]
  ] = value

proc stringify[T](tensor: Tensor[T], dim: int, index: var int): string =
  if dim == 0:
    result = $tensor.data[index]
    index += 1
  else:
    result = "["
    for it in 0..<tensor.shape[dim - 1]:
      if it != 0:
        result &= ", "
      result &= tensor.stringify(dim - 1, index)
    result &= "]"

proc `$`*[T](tensor: Tensor[T]): string =
  if tensor.is_nil:
    result = "nil"
  else:
    var index = 0
    result = tensor.stringify(tensor.shape.len, index)

template define_elementwise(op) =
  proc op*[T](a, b: Tensor[T]): Tensor[T] =
    assert a.shape == b.shape
    result = new_tensor[T](a.shape)
    for it in 0..<a.len:
      result.data[it] = op(a.data[it], b.data[it])

define_elementwise(`+`)
define_elementwise(`-`)
define_elementwise(min)
define_elementwise(max)

template define_elementwise_unary(op) =
  proc op*[T](tensor: Tensor[T]): Tensor[T] =
    result = new_tensor[T](tensor.shape)
    for it in 0..<tensor.len:
      result.data[it] = op(tensor.data[it])

define_elementwise_unary(abs)
define_elementwise_unary(`-`)

template define_elementwise_mut(op) =
  proc op*[T](a, b: Tensor[T]) =
    assert a.shape == b.shape
    for it in 0..<a.len:
      op(a.data[it], b.data[it])

define_elementwise_mut(`+=`)
define_elementwise_mut(`-=`)

template define_scalar_op(op) =
  proc op*[T](a: Tensor[T], b: T): Tensor[T] =
    result = new_tensor[T](a.shape)
    for it in 0..<a.len:
      result.data[it] = op(a.data[it], b)
  
  proc op*[T](a: T, b: Tensor[T]): Tensor[T] =
    result = new_tensor[T](b.shape)
    for it in 0..<b.len:
      result.data[it] = op(a, b.data[it])

define_scalar_op(`*`)
define_scalar_op(`/`)
define_scalar_op(`div`)
define_scalar_op(min)
define_scalar_op(max)

template define_reduce(name, op, initial) =
  proc name*[T](tensor: Tensor[T]): T =
    result = T(initial)
    for it in 0..<tensor.len:
      result = op(result, tensor.data[it])

define_reduce(sum, `+`, 0)
define_reduce(prod, `*`, 1)

template define_reduce(name, op) =
  proc name*[T](tensor: Tensor[T]): T =
    assert tensor.len > 0
    result = T(tensor.data[0])
    for it in 1..<tensor.len:
      result = op(result, tensor.data[it])

define_reduce(min, min)
define_reduce(max, max)

proc squares*[T](tensor: Tensor[T]): Tensor[T] =
  result = new_tensor[T](tensor.shape)
  for it in 0..<tensor.len:
    result.data[it] = tensor.data[it] * tensor.data[it]

proc remap*[T](tensor: Tensor[T], from_min, from_max, to_min, to_max: T): Tensor[T] =
  result = new_tensor[T](tensor.shape)
  let
    from_size = from_max - from_min
    to_size = to_max - to_min
  for it in 0..<tensor.len:
    result.data[it] = (tensor.data[it] - from_min) / from_size * to_size + to_min

proc is_matrix*[T](tensor: Tensor[T]): bool {.inline.} =
  result = tensor.shape.len == 2

proc `*`*[T](a, b: Tensor[T]): Tensor[T] =
  ## Matrix multiplication
  assert a.is_matrix and b.is_matrix
  assert a.shape[0] == b.shape[1]
  result = new_tensor[T]([b.shape[0], a.shape[1]])
  for y in 0..<result.shape[1]:
    for it in 0..<a.shape[0]:
      for x in 0..<result.shape[0]:
        result[x, y] += a[it, y] * b[x, it]

proc transpose*[T](tensor: Tensor[T]): Tensor[T] =
  assert tensor.is_matrix
  result = new_tensor[T]([tensor.shape[1], tensor.shape[0]])
  for y in 0..<result.shape[1]:
    for x in 0..<result.shape[0]:
      result[x, y] = tensor[y, x]

proc convert*[A, B](tensor: Tensor[A]): Tensor[B] =
  result = new_tensor[B](tensor.shape)
  for it in 0..<tensor.len:
    result[it] = B(tensor.data[it])

template convert*[A](tensor: Tensor[A], B: typedesc): Tensor[B] =
  convert[A, B](tensor)

proc one_hot*[T](indices: Tensor[T], count: int): Tensor[T] =
  result = new_tensor[T](@[count] & indices.shape.to_seq())
  for it in 0..<indices.len:
    result.data[it * count + int(indices.data[it])] = T(1)

proc fill_zero*[T](tensor: Tensor[T]) {.inline.} =
  if tensor.len > 0:
    zero_mem(tensor.data[0].addr, sizeof(T) * tensor.len)

proc fill*[T](tensor: Tensor[T], value: T) {.inline.} =
  for it in 0..<tensor.len:
    tensor.data[it] = value

proc fill_rand*[T](tensor: Tensor[T], slice: HSlice[T, T]) {.inline.} =
  for it in 0..<tensor.len:
    tensor.data[it] = rand(slice)

proc view_last*[T](tensor: Tensor[T], offset, size: int): Tensor[T] =
  let stride = tensor.len div tensor.shape[^1]
  result = Tensor[T](
    is_view: true,
    shape: tensor.shape[0..^2] & @[size],
    len: stride * size,
    data: cast[ptr UncheckedArray[T]](tensor.data[stride * offset].addr)
  )

proc view_last*[T](tensor: Tensor[T], slice: HSlice[int, int]): Tensor[T] =
  result = tensor.view_last(slice.a, slice.b - slice.a + 1)

proc select_samples*[T](tensor: Tensor[T], idx: openArray[int]): Tensor[T] =
  result = new_tensor[T](tensor.shape[0..^2] & @[idx.len])
  let stride = result.len div idx.len
  for it, id in idx:
    copy_mem(
      result.data[it * stride].addr,
      tensor.data[id * stride].addr,
      stride * sizeof(T)
    )

proc shuffle_xy*[T](x, y: Tensor[T]): tuple[x, y: Tensor[T]] =
  assert x.shape[^1] == y.shape[^1]
  var idx = new_seq[int](x.shape[^1])
  for it in 0..<idx.len:
    idx[it] = it
  shuffle(idx)
  result.x = x.select_samples(idx)
  result.y = y.select_samples(idx)

proc shuffle_xy*[T](tensors: (Tensor[T], Tensor[T])): tuple[x, y: Tensor[T]] =
  result = shuffle_xy(tensors[0], tensors[1])

proc concat_last*[T](a, b: Tensor[T]): Tensor[T] =
  assert a.shape[0..^2] == b.shape[0..^2]
  result = new_tensor[T](a.shape[0..^2] & @[a.shape[^1] + b.shape[^1]])
  let stride = result.len div result.shape[^1]
  for it in 0..<a.shape[^1]:
    copy_mem(
      result.data[it * stride].addr,
      a.data[it * stride].addr,
      stride * sizeof(T)
    )
  for it in 0..<b.shape[^1]:
    copy_mem(
      result.data[(it + a.shape[^1]) * stride].addr,
      b.data[it * stride].addr,
      stride * sizeof(T)
    )

proc reshape*[T](tensor: Tensor[T], shape: openArray[int]): Tensor[T] =
  var
    len = 1
    target_shape = new_seq[int](shape.len)
    var_dim = -1
  for dim, size in shape:
    if size < 0:
      var_dim = dim
    else:
      len *= size
      target_shape[dim] = size
  if var_dim != -1:
    target_shape[var_dim] = tensor.len div len
  
  result = alloc_tensor[T]()
  result.alloc_shape(target_shape, fill_zero = false)
  copy_mem(result.data[0].addr, tensor.data[0].addr, result.len * sizeof(T))
