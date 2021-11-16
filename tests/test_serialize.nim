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

proc check_serialize[T](value: T, path: string = "data.bin") =
  value.save(path)
  var
    stream = open_read_stream(path)
    loaded: T
  defer: stream.close()
  stream.load(loaded)
  check loaded == value

test "scalar":
  check_serialize(true)
  check_serialize('a')
  check_serialize(1)
  check_serialize(-100)
  check_serialize(3.14)

test "composite":
  check_serialize("Hello, world!")
  check_serialize(@[1, 2, 3])
  check_serialize(@[new_seq[int](), @[0], @[1, 2], @[3, 4, 5]])
  check_serialize({1'u8, 2'u8, 3'u8})
  check_serialize(to_hash_set([1, 2, 3]))
  check_serialize(to_table({"a": 1, "b": 2}))
  check_serialize(new_seq[int](1024))

test "tensor":
  check_serialize(new_tensor([3, 2], @[float32 1, 2, 3, 4, 5, 6]))
  check_serialize(new_tensor([1, 3, 2], @[int 1, 2, 3, 4, 5, 6]))

test "ir":
  check_serialize(TensorId(10))
  check_serialize(Instr(kind: InstrAdd, args: @[RegId(0), RegId(1)], res: RegId(3)))
  check_serialize(Instr(kind: InstrLen, tensor: TensorId(2), res: RegId(3)))
  check_serialize(Instr(kind: InstrIndex, index_lit: 10, res: RegId(3)))

