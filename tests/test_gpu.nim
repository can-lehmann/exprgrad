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

import std/[tables, sequtils]
import exprgrad, exprgrad/[ir, irprint, model, parser]
import ../tools/test_framework

test "matmul/passes":
  let
    a = input("a", [1024, -1])
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

test "matmul/compile":
  let
    a = input("a", [1024, -1])
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
  when TARGET_SUPPORTS_GPU:
    let program = compile[float32](c.target("c", CompileGpu), gpu=new_gpu_context())

test "static_shapes":
  for (size, expect_bounds_cond) in [(1024, false), (512, false), (123, true), (8, true)]:
    let
      a = input("a", [size, size])
      b = input("b", [size, size])
    c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it) do:
      schedule:
        tile_size(x, 16)
        tile_size(y, 16)
        tile_size(it, 16)
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    let has_bounds_cond = program.targets["c"].kernels[0].setup
      .any_it(it.kind == InstrGpu and it.body.any_it(it.kind == InstrIf))
    check has_bounds_cond == expect_bounds_cond
