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

proc check_cache(name, data: string, search_paths: openArray[string] = ["cache", "tests/cache"]) =
  let base_path = block:
    var base_path = ""
    for path in search_paths:
      if dir_exists(path):
        base_path = path
        break
    base_path
  
  let path = base_path / name
  if file_exists(path):
    check read_file(path).strip() == data.strip()
  else:
    write_file(path, data)

proc mse[T](a, b: Tensor[T]): T = squares(a - b).sum()

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

test "matmul/basic":
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
    check program.call("c", {"a": a_tensor, "b": b_tensor}).mse(expected) < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_basic.ir", $program)

test "matmul/unknown_shape":
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
    check program.call("c", {"a": a_tensor, "b": b_tensor}).mse(expected) < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_unknown_shape.ir", $program)

test "matmul/unknown_dim":
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
    check program.call("c", {"a": a_tensor, "b": b_tensor}).mse(expected) < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_unknown_dim.ir", $program)


test "matmul/schedule/tiled16":
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
    check program.call("c", {"a": a_tensor, "b": b_tensor}).mse(expected) < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_schedule_tiled16.ir", $program)

test "matmul/schedule/tiled32x16/known_shapes":
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
    check program.call("c", {"a": a_tensor, "b": b_tensor}).mse(expected) < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_schedule_tiled32x16_known_shapes.ir", $program)

test "matmul/schedule/tiled32x16/unknown_shapes":
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
    check program.call("c", {"a": a_tensor, "b": b_tensor}).mse(expected) < 0.1
  else:
    let program = to_program([c.target("c", CompileGpu)])
    program.compile()
    check_cache("matmul_schedule_tiled32x16_unknown_shapes.ir", $program)

proc conv1[T](image, filter: Tensor[T]): Tensor[T] =
  result = new_tensor[T]([image.shape[0] - filter.shape[0] + 1])
  for x in 0..<result.shape[0]:
    for dx in 0..<filter.shape[0]:
      result[x] += image[x + dx] * filter[dx]

test "conv1/basic":
  let
    image = input("image", [68])
    filter = input("filter", [5])
  res*[x] ++= image[x + dx] * filter[dx] | (x, dx)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](res.target("res", CompileGpu), gpu=new_gpu_context())
      image_tensor = new_rand_tensor[float32]([68], float32(0)..float32(1))
      filter_tensor = new_rand_tensor[float32]([5], float32(-1)..float32(1))
      res_tensor = program.call("res", {"image": image_tensor, "filter": filter_tensor})
      expected = conv1(image_tensor, filter_tensor)
    check res_tensor.mse(expected) < 0.1
  else:
    let program = to_program([res.target("res", CompileGpu)])
    program.compile()
    check_cache("conv1_basic.ir", $program)

test "conv1/schedule/tiled16":
  let
    image = input("image", [68])
    filter = input("filter", [5])
  res*[x] ++= image[x + dx] * filter[dx] | (x, dx) do:
    schedule:
      parallel(x)
      gpu:
        tile_size(x, 16)
        share_cache(dx)
        cache(image)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](res.target("res", CompileGpu), gpu=new_gpu_context())
      image_tensor = new_rand_tensor[float32]([68], float32(0)..float32(1))
      filter_tensor = new_rand_tensor[float32]([5], float32(-1)..float32(1))
      res_tensor = program.call("res", {"image": image_tensor, "filter": filter_tensor})
      expected = conv1(image_tensor, filter_tensor)
    check res_tensor.mse(expected) < 0.1
  else:
    let program = to_program([res.target("res", CompileGpu)])
    program.compile()
    check_cache("conv1_schedule_tiled16.ir", $program)

test "relu/basic":
  let x = input("x")
  y*{it} ++= select(x{it} > 0.0, x{it}, 0.01 * x{it}) | it
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](y.target("y", CompileGpu), gpu=new_gpu_context())
      x_tensor = new_tensor([2, 3], @[float32 1, 2, -1, -2, 0, 3])
      y_tensor = new_tensor([2, 3], @[float32 1, 2, -0.01, -0.02, 0, 3])
    check program.call("y", {"x": x_tensor}).mse(y_tensor) < 0.1
  else:
    let program = to_program([y.target("y", CompileGpu)])
    program.compile()
    check_cache("relu_basic.ir", $program)

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
