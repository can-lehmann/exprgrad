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

# Unit-tests for compiling exprgrad kernels to the GPU

import std/tables
import exprgrad/[ir, irprint, model, parser, dsl]
import ../tools/test_framework

test "matmul":
  let
    a = input("a")
    b = input("b")
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it) do:
    schedule:
      parallel(y)
      gpu:
        tile_size(x, 32)
        tile_size(y, 16)
        parallel(x)
        cache(a)
        cache(b)
        tile_size(it, 16)
        tile(it)
  let program = to_program([c.target("c", CompileGpu)])
  program.compile()
  echo program.targets["c"]
