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

import std/[tables, sequtils, os, strutils]
import exprgrad, exprgrad/[ir, irprint, model, parser]
import ../tools/test_framework

proc check_cache(name, data: string, path: string = "cache") =
  if file_exists(path / name):
    check read_file(path / name).strip() == data.strip()
  else:
    write_file(path / name, data)

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

test "matmul/compile/basic":
  let
    a = input("a", [64, 64])
    b = input("b", [64, 64])
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=new_gpu_context())
      a_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      b_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      expected = a_tensor * b_tensor
    check squares(program.call("c", {"a": a_tensor, "b": b_tensor}) - expected).sum() < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_compile_basic.ir", $program)

test "matmul/compile/unknown_shape":
  let
    a = input("a")
    b = input("b")
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=new_gpu_context())
      a_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      b_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      expected = a_tensor * b_tensor
    check squares(program.call("c", {"a": a_tensor, "b": b_tensor}) - expected).sum() < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_compile_unknown_shape.ir", $program)

test "matmul/compile/unknown_dim":
  let
    a = input("a", [64, -1])
    b = input("b")
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=new_gpu_context())
      a_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      b_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      expected = a_tensor * b_tensor
    check squares(program.call("c", {"a": a_tensor, "b": b_tensor}) - expected).sum() < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_compile_unknown_dim.ir", $program)


test "matmul/compile/schedule/tiled16":
  let
    a = input("a", [64, 64])
    b = input("b", [64, 64])
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it) do:
    schedule:
      parallel(y)
      gpu:
        tile_size(x, 16)
        tile_size(y, 16)
        parallel(x)
        cache(a)
        cache(b)
        tile_size(it, 16)
        tile(it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=new_gpu_context())
      a_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      b_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      expected = a_tensor * b_tensor
    check squares(program.call("c", {"a": a_tensor, "b": b_tensor}) - expected).sum() < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_compile_tiled16.ir", $program)

test "matmul/compile/schedule/tiled32x16/known_shapes":
  let
    a = input("a", [64, 64])
    b = input("b", [64, 64])
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
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=new_gpu_context())
      a_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      b_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      expected = a_tensor * b_tensor
    check squares(program.call("c", {"a": a_tensor, "b": b_tensor}) - expected).sum() < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_compile_tiled32x16_known_shapes.ir", $program)

test "matmul/compile/schedule/tiled32x16/unknown_shapes":
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
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=new_gpu_context())
      a_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      b_tensor = new_rand_tensor[float32]([64, 64], float32(0)..float32(1))
      expected = a_tensor * b_tensor
    check squares(program.call("c", {"a": a_tensor, "b": b_tensor}) - expected).sum() < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_compile_tiled32x16_unknown_shapes.ir", $program)

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
