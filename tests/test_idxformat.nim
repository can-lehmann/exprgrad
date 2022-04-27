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

proc check_save_load[T](slice: HSlice[T, T] = T(0)..T(1),
                        path: string = "data.bin") =
  let tensor = new_rand_tensor([2, 3, 4], slice)
  tensor.save_idx(path)
  check load_idx[T](path) == tensor

test "int":
  check_save_load[uint8](uint8(0)..uint8(1))
  check_save_load[int8]()
  check_save_load[int16]()
  check_save_load[int32]()

test "float":
  check_save_load[float32]()
  check_save_load[float64]()
