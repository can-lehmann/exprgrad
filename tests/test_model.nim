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

# Unit-tests for exprgrad

import std/random
import exprgrad, exprgrad/irprint
import ../tools/test_framework

test "matmul":
  c*[y, x] ++= input("a")[y, it] * input("b")[it, x] | (x, y, it)
  let model = compile[float32](c.target("c"))
  block:
    let
      a = new_tensor([2, 3], @[float32 1, 2, 3, 4, 5, 6])
      b = new_tensor([3, 2], @[float32 1, 2, 3, 4, 5, 6])
    check model.call("c", {"a": a, "b": b}) == a * b

test "relu":
  var inp = input("inp")
  outp*{it} ++= select(0.0 < inp{it}, inp{it}, 0.0) | it
  let model = compile[float32](outp.target("outp"))
  block:
    let
      inp = new_tensor[float32]([2, 3], @[float32 0, -1, 10, -20, 0.1, -0.1])
      outp = new_tensor[float32]([2, 3], @[float32 0, 0, 10, 0, 0.1, 0])
    check model.call("outp", {"inp": inp}) == outp

test "mean_squared_error":
  loss*[0] ++= sq(input("pred"){it} - input("labels"){it}) | it
  let model = compile[float32](loss.target("loss"))
  block:
    let
      pred = new_tensor[float32]([2, 2], @[float32 1, 2, 3, 4])
      labels = new_tensor[float32]([2, 2], @[float32 4, 3, 2, 1])
    check model.call("loss", {
      "pred": pred, "labels": pred
    }) == new_tensor([1], @[float32 0])
    
    check model.call("loss", {
      "pred": pred, "labels": labels
    }) == new_tensor([1], @[float32 9 + 1 + 1 + 9])

test "transpose":
  b*[y, x] ++= input("a")[x, y] | (x, y)
  block:
    let
      model = compile[float32](b.target("b"))
      a = new_tensor[float32]([2, 3], @[float32 1, 2, 3, 4, 5, 6])
    
    check model.call("b", {"a": a}) == a.transpose()

test "max":
  let x = input("x")
  res*{it} ++= max(x{it}, input("y"){it}) | it
  res.copy_shape(x)
  
  let model = compile[float32](res.target("z"))
  check model.call("z", {
    "x": new_tensor([3, 2], @[float32 1, 0, 3, 4, -10, 6]),
    "y": new_tensor([3, 2], @[float32 1, 2, -3, 2, 5, 5.5])
  }) == new_tensor([3, 2], @[float32 1, 2, 3, 4, 5, 6])

test "conv1":
  res*[x] ++= input("image")[x + dx] * input("filter")[dx] | (x, dx)
  let model = compile[float32](res.target("res"))
  check model.call("res", {
    "image": new_tensor([7], @[float32 1, 2, 3, 2, 1, 0, -1]),
    "filter": new_tensor([3], @[float32 1, 2, 3])
  }) == new_tensor([5], @[float32 14, 14, 10, 4, -2])

test "blur":
  res*[x] ++= (
    let image = input("image");
    (image[x] + image[x + 1] + image[x + 2]) / 3.0
  ) | (x in 0..<res.shape[0]) # TODO: Infer loop bounds
  let model = compile[float32](res.target("res"))
  check model.call("res", {
    "image": new_tensor([7], @[float32 1, 2, 3, 2, 1, 0, -1]),
  }) == new_tensor([5], @[float32 2, float32(7/3), 2, 1, 0])

test "blur_center":
  let image = input("image")
  res*[x - 1] ++= (
    (image[x - 1] + image[x] + image[x + 1]) / 3.0
  ) | (x in 1..<(image.shape[0] - 1)) # TODO: Infer loop bounds
  let model = compile[float32](res.target("res"))
  check model.call("res", {
    "image": new_tensor([7], @[float32 1, 2, 3, 2, 1, 0, -1]),
  }) == new_tensor([5], @[float32 2, float32(7/3), 2, 1, 0])

test "blur_offset":
  let image = input("image")
  res*[x + 1] ++= (
    (image[x] + image[x + 1] + image[x + 2]) / 3.0
  ) | (x in 0..<(image.shape[0] - 2)) # TODO: Infer loop bounds
  res.with_shape([image.shape[0]])
  let model = compile[float32](res.target("res"))
  check model.call("res", {
    "image": new_tensor([7], @[float32 1, 2, 3, 2, 1, 0, -1]),
  }) == new_tensor([7], @[float32 0, 2, float32(7/3), 2, 1, 0, 0])

test "shape":
  let inp = input("x")
  res*[0] ++= to_scalar(inp.shape[0])
  res[1] ++= to_scalar(inp.shape[^2])
  res[2] ++= to_scalar(inp.shape[^1])
  res[3] ++= to_scalar(inp.shape.len)
  res[4] ++= to_scalar(inp.len)
  res.with_shape(5)
  
  let model = compile[float64](res.target("y"))
  check model.call("y", {"x": new_tensor([1, 2, 3, 4], 0.0)}) == new_tensor([5], @[float64 1, 3, 4, 4, 24])
  check model.call("y", {"x": new_tensor([2, 3], 0.0)}) == new_tensor([5], @[float64 2, 2, 3, 2, 6])

test "extern":
  proc `*`(inp: Fun, factor: float64): Fun =
    result{it} ++= inp{it} * factor | it
  
  proc test_with_factor(factor: float64) =
    let
      model = compile[float64](target(input("x") * factor, "y"))
      x = new_tensor[float64]([2, 3], @[float64 1, 2, 3, 4, 5, 6])
    check model.call("y", {"x": x}) == x * factor
  
  for it in -2..2:
    test_with_factor(float64(it))

test "xor":
  randomize(10)
  
  hidden*[y, x] ++= input("x")[y, it] * param([2, 4])[it, x] | (y, x, it)
  hidden[y, x] ++= param([4])[x] | (y, x)
  hidden_relu*{it} ++= select(hidden{it} <= 0.0, 0.1 * hidden{it}, hidden{it}) | it
  output*[y, x] ++= hidden_relu[y, it] * param([4, 1])[it, x] | (y, x, it)
  output[y, x] ++= param([1])[x] | (y, x)
  output_sigmoid*{it} ++= 1.0 / (1.0 + exp(-output{it})) | it
  let pred = output_sigmoid.target("predict")
  
  proc optim(param: var Fun, grad: Fun) =
    param{it} ++= -0.1 * grad{it} | it
  loss*[0] ++= sq(pred{it} - input("y"){it}) | it
  let net = loss.target("loss").backprop(optim).target("train")
  
  let model = compile[float32](net)
  
  let
    train_x = new_tensor([4, 2], @[float32 0, 0, 0, 1, 1, 0, 1, 1])
    train_y = new_tensor([4, 1], @[float32 0, 1, 1, 0])
  
  for epoch in 0..<1000:
    model.apply("train", {"x": train_x, "y": train_y})
  
  check squares(model.call("predict", {"x": train_x}) - train_y).sum() < 0.1

test "custom_grad":
  let inp = input("inp")
  identity*{x} ++= inp{x} | x do:
    custom_grad:
      grad(inp){x} ++= inp{x} * 2.0 * grad(identity){x} | x
  
  let
    graph = identity
      .target("identity")
      .backwards()
      .grad(inp)
      .target("grad")
    model = compile[float32](graph)
  
  block:
    let tensor = new_tensor[float32]([2, 2], @[float32 1, 2, 3, 4])
    check model.call("identity", {"inp": tensor}) == tensor
    check model.call("grad", {"inp": tensor}) == tensor * 2

test "dynamic_ast":
  proc elementwise_pow(fun: Fun, n: int): Fun =
    var prod: Scalar = 1.0
    for _ in 0..<n:
      prod = prod * fun{iterator_literal("it")}
    result{it} ++= prod | it
    result.copy_shape(fun)
  
  let x = new_tensor([3, 2], @[float32 1, 2, 3, 4, 5, 6])
  var expected_y = new_tensor[float32]([3, 2], 1)
  for n in 0..<2:
    let
      model = compile[float32](input("x").elementwise_pow(n).target("y"))
      y = model.call("y", {"x": x})
    check squares(y - expected_y).sum() < 0.001
    for it in 0..<expected_y.len:
      expected_y{it} *= x{it}

test "array":
  res*[x] ++= (
    let arr = literal([1.0, 2.0, 3.0]);
    arr[x] + to_scalar(arr.len)
  ) | x
  res.with_shape(3)
  
  let model = compile[float32](res.target("y"))
  check model.call("y", []) == new_tensor([3], @[float32 4, 5, 6])

test "nested_array":
  res*[y, x] ++= (
    let arr = literal([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0]
    ]);
    arr[y][x]
  ) | (y, x)
  res.with_shape(3, 3)
  
  let model = compile[float32](res.target("y"))
  check model.call("y", []) == new_tensor([3, 3], @[float32 1, 2, 3, 4, 5, 6, 7, 8, 9])

test "loop_bounds":
  res*[x] ++= 1.0 | (x in 2..<4)
  res[x] ++= -1.0 | (x in 0..<1)
  res[x] ++= -2.0 | (x in 1..<1)
  res.with_shape(5)
  let model = compile[float32](res.target("res"))
  check model.call("res") == new_tensor([5], @[float32 -1, 0, 1, 1, 0])

