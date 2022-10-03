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

# Unit-tests for exprgrad's idxformat module

import exprgrad, exprgrad/io/idxformat
import ../tools/test_framework

proc checkSaveLoad[T](slice: HSlice[T, T] = low(T)..high(T),
                        path: string = "data.bin") =
  let tensor = newRandTensor([2, 3, 4], slice)
  tensor.saveIdx(path)
  check loadIdx[T](path) == tensor

test "int":
  checkSaveLoad[uint8]()
  checkSaveLoad[int8]()
  checkSaveLoad[int16]()
  checkSaveLoad[int32]()

test "float":
  checkSaveLoad(float32(-10)..float32(10))
  checkSaveLoad(float64(-10)..float64(10))
