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

# Unit-tests for examples in talks about exprgrad

import std/[random, tables]
import exprgrad, exprgrad/[ir, irprint]
import ../tools/test_framework

test "matmul":
  proc matmul(a, b: Fun): Fun =
    result[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  
  proc nimMatmul[T](a, b: Tensor[T]): Tensor[T] =
    result = Tensor[T].new([a.shape[0], b.shape[1]])
    for y in 0..<result.shape[0]:
      for it in 0..<a.shape[1]:
        for x in 0..<result.shape[1]:
          result[y, x] += a[y, it] * b[it, x]
  
  let
    model = compile[float32](matmul(input("a"), input("b")).target("multiply"))
    a = Tensor.new([2, 2], @[float32 1, 2, 3, 4])
    b = Tensor.new([2, 3], @[float32 1, 2, 3, 4, 5, 6])
  
  check model.call("multiply", {"a": a, "b": b}) == nimMatmul(a, b)
  check model.call("multiply", {"a": a, "b": b}) == a * b

test "transpose":
  proc transpose(matrix: Fun): Fun =
    result[y, x] ++= matrix[x, y] | (y, x)
  
  let
    model = compile[float32](input("matrix").transpose().target("transpose"))
    matrix = Tensor.rand([4, 5], 0.0'f32..1.0'f32)
  
  check model.call("transpose", {"matrix": matrix}) == matrix.transpose()

test "increment":
  proc increment(input: Fun): Fun =
    result{it} ++= input{it} + 1.0 | it
  
  let
    model = compile[float32](input("input").increment().target("increment"))
    tensor = Tensor.new([1, 2, 3], @[float32 1, 2, 3, 4, 5, 6])
  
  check model.call("increment", {"input": tensor}) == tensor + Tensor.new([1, 2, 3], 1'f32)

test "sumPositive":
  proc sumPositive(input: Fun): Fun =
    result[0] ++= select(input{it} > 0.0, input{it}, 0.0) | it
  
  let
    model = compile[float32](input("input").sumPositive().target("sumPositive"))
    tensor = Tensor.new([2, 3], @[float32 1, -2, -3, 4, 5, -6])
  
  check model.call("sumPositive", {"input": tensor}) == Tensor.new([1], @[10'f32])

test "ones":
  subtest:
    proc ones(): Fun =
      result{it} ++= 1.0 | it
    
    checkException ShapeError:
      discard compile[float32](ones().target("ones"))
  
  subtest:
    proc ones(): Fun =
      result{it} ++= 1.0 | it
      result.withShape(2, 3)
    
    let model = compile[float32](ones().target("ones"))
    check model.call("ones") == Tensor.new([2, 3], 1'f32)

test "multiple":
  proc linear(input, weights, biases: Fun): Fun =
    result[y, x] ++= input[y, it] * weights[it, x] | (x, y, it)
    result[y, x] ++= biases[x] | (x, y)
  
  let
    graph = linear(input("input"), input("weights"), input("biases"))
    model = compile[float32](graph.target("predict"))
  
  check model.call("predict", {
    "input": Tensor.new([5, 2], @[float32 0, 0, 1, 0, 0, 1, 1, 1, 1, 2]),
    "weights": Tensor.new([2, 1], @[float32 2, 3]),
    "biases": Tensor.new([1], @[float32 1])
  }) == Tensor.new([5, 1], @[float32 1, 3, 4, 6, 9])

test "multiplyAndSquare":
  let
    a = input("a")
    b = input("b")
  
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  d*{it} ++= c{it} * c{it} | it
  
  let model = compile[float32]([
    c.target("multiply"),
    d.target("multiplyAndSquare")
  ])
  
  check model.call("multiply", {
    "a": Tensor.new([2, 2], @[float32 1, 2, 3, 4]),
    "b": Tensor.new([2, 1], @[float32 1, 2])
  }) == Tensor.new([2, 1], @[float32 5, 11])
  
  check model.call("multiplyAndSquare", {
    "a": Tensor.new([2, 2], @[float32 1, 2, 3, 4]),
    "b": Tensor.new([2, 1], @[float32 1, 2])
  }) == Tensor.new([2, 1], @[float32 25, 121])
