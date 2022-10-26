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

# Unit-tests for exprgrad's ppmformat module

import exprgrad, exprgrad/io/ppmformat
import ../tools/test_framework

proc checkSaveLoad(tensor: Tensor[uint8], path: string = "data.ppm", useAscii: bool = false) =
  tensor.savePpm(path, useAscii=useAscii)
  check loadPpm(path) == tensor

test "P2":
  checkSaveLoad(Tensor.new([4, 3, 1], 0'u8), useAscii=true)
  checkSaveLoad(Tensor.new([4, 3, 1], 255'u8), useAscii=true)
  checkSaveLoad(Tensor.rand([4, 3, 1], 0'u8..255'u8), useAscii=true)

test "P3":
  checkSaveLoad(Tensor.new([4, 3, 3], 0'u8), useAscii=true)
  checkSaveLoad(Tensor.new([4, 3, 3], 255'u8), useAscii=true)
  checkSaveLoad(Tensor.rand([4, 3, 3], 0'u8..255'u8), useAscii=true)

test "P5":
  checkSaveLoad(Tensor.new([4, 3, 1], 0'u8))
  checkSaveLoad(Tensor.new([4, 3, 1], 255'u8))
  checkSaveLoad(Tensor.rand([4, 3, 1], 0'u8..255'u8))

test "P6":
  checkSaveLoad(Tensor.new([4, 3, 3], 0'u8))
  checkSaveLoad(Tensor.new([4, 3, 3], 255'u8))
  checkSaveLoad(Tensor.rand([4, 3, 3], 0'u8..255'u8))
