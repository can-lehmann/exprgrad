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

# Unit-tests for exprgrad's serialize module

import std/[sets, tables]
import exprgrad, exprgrad/ir
import exprgrad/io/[serialize, faststreams]
import ../tools/test_framework

proc checkSerialize[T](value: T, path: string = "data.bin") =
  value.save(path)
  var
    stream = openReadStream(path)
    loaded: T
  defer: stream.close()
  stream.load(loaded)
  check loaded == value

test "scalar":
  checkSerialize(true)
  checkSerialize('a')
  checkSerialize(1)
  checkSerialize(-100)
  checkSerialize(3.14)

test "composite":
  checkSerialize("Hello, world!")
  checkSerialize(@[1, 2, 3])
  checkSerialize(@[newSeq[int](), @[0], @[1, 2], @[3, 4, 5]])
  checkSerialize({1'u8, 2'u8, 3'u8})
  checkSerialize(toHashSet([1, 2, 3]))
  checkSerialize(toTable({"a": 1, "b": 2}))
  checkSerialize(newSeq[int](1024))

test "tensor":
  checkSerialize(newTensor([3, 2], @[float32 1, 2, 3, 4, 5, 6]))
  checkSerialize(newTensor([1, 3, 2], @[int 1, 2, 3, 4, 5, 6]))

test "ir":
  checkSerialize(TensorId(10))
  checkSerialize(Instr(kind: InstrAdd, args: @[RegId(0), RegId(1)], res: RegId(3)))
  checkSerialize(Instr(kind: InstrLen, tensor: TensorId(2), res: RegId(3)))
  checkSerialize(Instr(kind: InstrIndex, indexLit: 10, res: RegId(3)))

