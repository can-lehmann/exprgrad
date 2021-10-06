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

# Load files in the IDX format (http://yann.lecun.com/exdb/mnist/)

import faststreams, ../tensors

template type_id(typ: typedesc): uint8 =
  when typ is uint8:
    0x08
  elif typ is int8:
    0x09
  elif typ of SomeInteger and sizeof(typ) == 2:
    0x0b
  elif typ of SomeInteger and sizeof(typ) == 4:
    0x0c
  elif typ is float32:
    0x0d
  elif typ is float64:
    0x0e
  else:
    0x00

proc read_uint[T](stream: var ReadStream): T =
  for it in countdown(sizeof(T) - 1, 0):
    result = result xor (T(stream.read_uint8()) shl (it * 8))

proc read_int32(stream: var ReadStream): int32 =
  cast[int32](read_uint[uint32](stream))

proc parse_idx*[T](stream: var ReadStream): Tensor[T] =
  stream.skip(2)
  if stream.read_uint8() != type_id(T):
    raise new_exception(ValueError, "Invalid tensor type")
  let dim_count = stream.read_uint8()
  var shape = new_seq[int](dim_count)
  for it in countdown(dim_count - 1, 0):
    shape[it] = stream.read_int32()
  result = new_tensor[T](shape)
  for it in 0..<result.len:
    when sizeof(T) == 1:
      let value = stream.read_uint8()
    elif sizeof(T) == 2:
      let value = read_uint[uint16](stream)
    elif sizeof(T) == 4:
      let value = read_uint[uint32](stream)
    elif sizeof(T) == 8:
      let value = read_uint[uint64](stream)
    result.data[it] = cast[T](value)

proc load_idx*[T](path: string): Tensor[T] =
  var stream = open_read_stream(path)
  defer: stream.close()
  result = parse_idx[T](stream)
