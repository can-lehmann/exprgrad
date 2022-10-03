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

template typeId(typ: typedesc): uint8 =
  when typ is uint8:
    0x08
  elif typ is int8:
    0x09
  elif typ is SomeInteger and sizeof(typ) == 2:
    0x0b
  elif typ is SomeInteger and sizeof(typ) == 4:
    0x0c
  elif typ is float32:
    0x0d
  elif typ is float64:
    0x0e
  else:
    0x00

proc readUint[T](stream: var ReadStream): T =
  for it in countdown(sizeof(T) - 1, 0):
    result = result xor (T(stream.readUint8()) shl (it * 8))

proc readInt32(stream: var ReadStream): int32 =
  cast[int32](readUint[uint32](stream))

proc parseIdx*[T](stream: var ReadStream): Tensor[T] =
  stream.skip(2)
  if stream.readUint8() != typeId(T):
    raise newException(ValueError, "Invalid tensor type")
  let dimCount = stream.readUint8()
  var shape = newSeq[int](dimCount)
  for it in 0..<int(dimCount):
    shape[it] = stream.readInt32()
  result = newTensor[T](shape)
  for it in 0..<result.len:
    when sizeof(T) == 1:
      let value = stream.readUint8()
    elif sizeof(T) == 2:
      let value = readUint[uint16](stream)
    elif sizeof(T) == 4:
      let value = readUint[uint32](stream)
    elif sizeof(T) == 8:
      let value = readUint[uint64](stream)
    result.data[it] = cast[T](value)

proc loadIdx*[T](path: string): Tensor[T] =
  var stream = openReadStream(path)
  defer: stream.close()
  result = parseIdx[T](stream)

proc writeUint[T: SomeUnsignedInt](stream: var WriteStream, value: T) =
  for it in countdown(sizeof(T) - 1, 0):
    stream.write(uint8((value shr (8 * it)) and 0xff))

proc writeIdx*[T](stream: var WriteStream, tensor: Tensor[T]) =
  stream.write([uint8(0), uint8(0)])
  stream.write(typeId(T))
  stream.write(uint8(tensor.shape.len))
  for dim in tensor.shape:
    stream.writeUint(uint32(dim))
  for it in 0..<tensor.len:
    let value = tensor.data[it]
    when sizeof(T) == 1:
      stream.write(cast[uint8](value))
    elif sizeof(T) == 2:
      stream.writeUint(cast[uint16](value))
    elif sizeof(T) == 4:
      stream.writeUint(cast[uint32](value))
    elif sizeof(T) == 8:
      stream.writeUint(cast[uint64](value))
    else:
      {.error: "Invalid tensor type".}

proc saveIdx*[T](tensor: Tensor[T], path: string) =
  var stream = openWriteStream(path)
  defer: stream.close()
  stream.writeIdx(tensor)
