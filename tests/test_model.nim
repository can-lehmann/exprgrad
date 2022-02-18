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
import exprgrad
import ../tools/test_framework

test "matmul":
  iters y, x, it: c*[y, x] ++= input("a")[y, it] * input("b")[it, x]
  let model = compile[float32](c.target("c"))
  block:
    let
      a = new_tensor([2, 3], @[float32 1, 2, 3, 4, 5, 6])
      b = new_tensor([3, 2], @[float32 1, 2, 3, 4, 5, 6])
    check model.call("c", {"a": a, "b": b}) == a * b

test "relu":
  var inp = input("inp")
  iters it: outp*{it} ++= select(0.0 < inp{it}, inp{it}, 0.0)
  let model = compile[float32](outp.target("outp"))
  block:
    let
      inp = new_tensor[float32]([2, 3], @[float32 0, -1, 10, -20, 0.1, -0.1])
      outp = new_tensor[float32]([2, 3], @[float32 0, 0, 10, 0, 0.1, 0])
    check model.call("outp", {"inp": inp}) == outp

test "mean_squared_error":
  iters it: loss*[0] ++= sq(input("pred"){it} - input("labels"){it})
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
  iters y, x: b*[y, x] ++= input("a")[x, y]
  block:
    let
      model = compile[float32](b.target("b"))
      a = new_tensor[float32]([2, 3], @[float32 1, 2, 3, 4, 5, 6])
    
    check model.call("b", {"a": a}) == a.transpose()

test "extern":
  proc `*`(inp: Fun, factor: float64): Fun =
    iters it:
      result{it} ++= inp{it} * factor
  
  proc test_with_factor(factor: float64) =
    let
      model = compile[float64](target(input("x") * factor, "y"))
      x = new_tensor[float64]([2, 3], @[float64 1, 2, 3, 4, 5, 6])
    check model.call("y", {"x": x}) == x * factor
  
  for it in -2..2:
    test_with_factor(float64(it))

test "xor":
  randomize(10)
  
  iters y, x, it:
    hidden*[y, x] ++= input("x")[y, it] * param([2, 4])[it, x]
    hidden[y, x] ++= param([4])[x]
    hidden_relu*{it} ++= select(hidden{it} <= 0.0, 0.1 * hidden{it}, hidden{it})
    output*[y, x] ++= hidden_relu[y, it] * param([4, 1])[it, x]
    output[y, x] ++= param([1])[x]
    output_sigmoid*{it} ++= 1.0 / (1.0 + exp(-output{it})) 
    let pred = output_sigmoid.target("predict")
    
    proc optim(param: var Fun, grad: Fun) =
      param{it} ++= -0.1 * grad{it}
    loss*[0] ++= sq(pred{it} - input("y"){it})
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
  iters x:
    identity*{x} ++= inp{x} | custom_grad(
      grad(inp){x} ++= inp{x} * 2.0 * grad(identity){x}
    )
  
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
    for it in 0..<n:
      iters it: prod = prod * fun{it}
    iters it: result{it} ++= prod
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
  iters x: res*[x] ++= (
    let arr = literal([1.0, 2.0, 3.0]);
    arr[x] + to_scalar(arr.len)
  )
  res.with_shape([3])
  
  let model = compile[float32](res.target("y"))
  check model.call("y", []) == new_tensor([3], @[float32 4, 5, 6])

test "nested_array":
  iters x, y: res*[y, x] ++= (
    let arr = literal([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0]
    ]);
    arr[y][x]
  )
  res.with_shape([3, 3])
  
  let model = compile[float32](res.target("y"))
  check model.call("y", []) == new_tensor([3, 3], @[float32 1, 2, 3, 4, 5, 6, 7, 8, 9])

