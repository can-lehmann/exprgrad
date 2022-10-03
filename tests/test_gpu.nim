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

proc checkCache(name, data: string, searchPaths: openArray[string] = ["cache", "tests/cache"]) =
  let basePath = block:
    var basePath = ""
    for path in searchPaths:
      if dirExists(path):
        basePath = path
        break
    basePath
  
  let path = basePath / name
  if fileExists(path):
    check readFile(path).strip() == data.strip()
  else:
    writeFile(path, data)

proc mse[T](a, b: Tensor[T]): T = squares(a - b).sum()

test "matmul/passes":
  let
    a = input("a", [1024, -1])
    b = input("b")
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it) do:
    schedule:
      parallel(y)
      gpu:
        tileSize(x, 32)
        tileSize(y, 16)
        parallel(x)
        cache(a)
        cache(b)
        tileSize(it, 16)
        tile(it)
  let program = toProgram([c.target("c", CompileGpu)])
  program.compile()
  echo program.targets["c"]

test "matmul/basic":
  let
    a = input("a", [64, 64])
    b = input("b", [64, 64])
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=newGpuContext())
      aTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      bTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      expected = aTensor * bTensor
    check program.call("c", {"a": aTensor, "b": bTensor}).mse(expected) < 0.1
  else:
    let program = toProgram([c.target("c", CompileGpu)])
    program.compile()
    checkCache("matmul_basic.ir", $program)

test "matmul/unknown_shape":
  let
    a = input("a")
    b = input("b")
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=newGpuContext())
      aTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      bTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      expected = aTensor * bTensor
    check program.call("c", {"a": aTensor, "b": bTensor}).mse(expected) < 0.1
  else:
    let program = toProgram([c.target("c", CompileGpu)])
    program.compile()
    checkCache("matmul_unknown_shape.ir", $program)

test "matmul/unknown_dim":
  let
    a = input("a", [64, -1])
    b = input("b")
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=newGpuContext())
      aTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      bTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      expected = aTensor * bTensor
    check program.call("c", {"a": aTensor, "b": bTensor}).mse(expected) < 0.1
  else:
    let program = toProgram([c.target("c", CompileGpu)])
    program.compile()
    checkCache("matmul_unknown_dim.ir", $program)


test "matmul/schedule/tiled16":
  let
    a = input("a", [64, 64])
    b = input("b", [64, 64])
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it) do:
    schedule:
      parallel(y)
      gpu:
        tileSize(x, 16)
        tileSize(y, 16)
        parallel(x)
        cache(a)
        cache(b)
        tileSize(it, 16)
        tile(it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=newGpuContext())
      aTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      bTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      expected = aTensor * bTensor
    check program.call("c", {"a": aTensor, "b": bTensor}).mse(expected) < 0.1
  else:
    let program = toProgram([c.target("c", CompileGpu)])
    program.compile()
    checkCache("matmul_schedule_tiled16.ir", $program)

test "matmul/schedule/tiled32x16/known_shapes":
  let
    a = input("a", [64, 64])
    b = input("b", [64, 64])
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it) do:
    schedule:
      parallel(y)
      gpu:
        tileSize(x, 32)
        tileSize(y, 16)
        parallel(x)
        cache(a)
        cache(b)
        tileSize(it, 16)
        tile(it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=newGpuContext())
      aTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      bTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      expected = aTensor * bTensor
    check program.call("c", {"a": aTensor, "b": bTensor}).mse(expected) < 0.1
  else:
    let program = toProgram([c.target("c", CompileGpu)])
    program.compile()
    checkCache("matmul_schedule_tiled32x16_known_shapes.ir", $program)

test "matmul/schedule/tiled32x16/unknown_shapes":
  let
    a = input("a")
    b = input("b")
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it) do:
    schedule:
      parallel(y)
      gpu:
        tileSize(x, 32)
        tileSize(y, 16)
        parallel(x)
        cache(a)
        cache(b)
        tileSize(it, 16)
        tile(it)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](c.target("c", CompileGpu), gpu=newGpuContext())
      aTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      bTensor = newRandTensor[float32]([64, 64], float32(0)..float32(1))
      expected = aTensor * bTensor
    check program.call("c", {"a": aTensor, "b": bTensor}).mse(expected) < 0.1
  else:
    let program = toProgram([c.target("c", CompileGpu)])
    program.compile()
    checkCache("matmul_schedule_tiled32x16_unknown_shapes.ir", $program)

proc conv1[T](image, filter: Tensor[T]): Tensor[T] =
  result = newTensor[T]([image.shape[0] - filter.shape[0] + 1])
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
      program = compile[float32](res.target("res", CompileGpu), gpu=newGpuContext())
      imageTensor = newRandTensor[float32]([68], float32(0)..float32(1))
      filterTensor = newRandTensor[float32]([5], float32(-1)..float32(1))
      resTensor = program.call("res", {"image": imageTensor, "filter": filterTensor})
      expected = conv1(imageTensor, filterTensor)
    check resTensor.mse(expected) < 0.1
  else:
    let program = toProgram([res.target("res", CompileGpu)])
    program.compile()
    checkCache("conv1_basic.ir", $program)

test "conv1/schedule/tiled16":
  let
    image = input("image", [68])
    filter = input("filter", [5])
  res*[x] ++= image[x + dx] * filter[dx] | (x, dx) do:
    schedule:
      parallel(x)
      gpu:
        tileSize(x, 16)
        shareCache(dx)
        cache(image)
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](res.target("res", CompileGpu), gpu=newGpuContext())
      imageTensor = newRandTensor[float32]([68], float32(0)..float32(1))
      filterTensor = newRandTensor[float32]([5], float32(-1)..float32(1))
      resTensor = program.call("res", {"image": imageTensor, "filter": filterTensor})
      expected = conv1(imageTensor, filterTensor)
    check resTensor.mse(expected) < 0.1
  else:
    let program = toProgram([res.target("res", CompileGpu)])
    program.compile()
    checkCache("conv1_schedule_tiled16.ir", $program)

test "relu/basic":
  let x = input("x")
  y*{it} ++= select(x{it} > 0.0, x{it}, 0.01 * x{it}) | it
  when TARGET_SUPPORTS_GPU:
    let
      program = compile[float32](y.target("y", CompileGpu), gpu=newGpuContext())
      xTensor = newTensor([2, 3], @[float32 1, 2, -1, -2, 0, 3])
      yTensor = newTensor([2, 3], @[float32 1, 2, -0.01, -0.02, 0, 3])
    check program.call("y", {"x": xTensor}).mse(yTensor) < 0.1
  else:
    let program = toProgram([y.target("y", CompileGpu)])
    program.compile()
    checkCache("relu_basic.ir", $program)

test "static_shapes":
  for (size, expectBoundsCond) in [(1024, false), (512, false), (123, true), (8, true)]:
    let
      a = input("a", [size, size])
      b = input("b", [size, size])
    c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it) do:
      schedule:
        tileSize(x, 16)
        tileSize(y, 16)
        tileSize(it, 16)
    let program = toProgram([c.target("c", CompileGpu)])
    program.compile()
    let hasBoundsCond = program.targets["c"].kernels[0].setup
      .anyIt(it.kind == InstrGpu and it.body.anyIt(it.kind == InstrIf))
    check hasBoundsCond == expectBoundsCond
