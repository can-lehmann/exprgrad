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

import std/[random, math]
import exprgrad, exprgrad/irprint
import ../tools/test_framework

test "matmul":
  c*[y, x] ++= input("a")[y, it] * input("b")[it, x] | (x, y, it)
  let model = compile[float32](c.target("c"))
  block:
    let
      a = newTensor([2, 3], @[float32 1, 2, 3, 4, 5, 6])
      b = newTensor([3, 2], @[float32 1, 2, 3, 4, 5, 6])
    check model.call("c", {"a": a, "b": b}) == a * b

test "relu":
  var inp = input("inp")
  outp*{it} ++= select(0.0 < inp{it}, inp{it}, 0.0) | it
  let model = compile[float32](outp.target("outp"))
  block:
    let
      inp = newTensor[float32]([2, 3], @[float32 0, -1, 10, -20, 0.1, -0.1])
      outp = newTensor[float32]([2, 3], @[float32 0, 0, 10, 0, 0.1, 0])
    check model.call("outp", {"inp": inp}) == outp

test "meanSquaredError":
  loss*[0] ++= sq(input("pred"){it} - input("labels"){it}) | it
  let model = compile[float32](loss.target("loss"))
  block:
    let
      pred = newTensor[float32]([2, 2], @[float32 1, 2, 3, 4])
      labels = newTensor[float32]([2, 2], @[float32 4, 3, 2, 1])
    check model.call("loss", {
      "pred": pred, "labels": pred
    }) == newTensor([1], @[float32 0])
    
    check model.call("loss", {
      "pred": pred, "labels": labels
    }) == newTensor([1], @[float32 9 + 1 + 1 + 9])

test "transpose":
  b*[y, x] ++= input("a")[x, y] | (x, y)
  block:
    let
      model = compile[float32](b.target("b"))
      a = newTensor[float32]([2, 3], @[float32 1, 2, 3, 4, 5, 6])
    
    check model.call("b", {"a": a}) == a.transpose()

test "max":
  let x = input("x")
  res*{it} ++= max(x{it}, input("y"){it}) | it
  res.copyShape(x)
  
  let model = compile[float32](res.target("z"))
  check model.call("z", {
    "x": newTensor([3, 2], @[float32 1, 0, 3, 4, -10, 6]),
    "y": newTensor([3, 2], @[float32 1, 2, -3, 2, 5, 5.5])
  }) == newTensor([3, 2], @[float32 1, 2, 3, 4, 5, 6])

test "conv1":
  res*[x] ++= input("image")[x + dx] * input("filter")[dx] | (x, dx)
  let model = compile[float32](res.target("res"))
  check model.call("res", {
    "image": newTensor([7], @[float32 1, 2, 3, 2, 1, 0, -1]),
    "filter": newTensor([3], @[float32 1, 2, 3])
  }) == newTensor([5], @[float32 14, 14, 10, 4, -2])

test "blur":
  res*[x] ++= (
    let image = input("image");
    (image[x] + image[x + 1] + image[x + 2]) / 3.0
  ) | (x in 0..<res.shape[0]) # TODO: Infer loop bounds
  let model = compile[float32](res.target("res"))
  check model.call("res", {
    "image": newTensor([7], @[float32 1, 2, 3, 2, 1, 0, -1]),
  }) == newTensor([5], @[float32 2, float32(7/3), 2, 1, 0])

test "blurCenter":
  let image = input("image")
  res*[x - 1] ++= (
    (image[x - 1] + image[x] + image[x + 1]) / 3.0
  ) | (x in 1..<(image.shape[0] - 1)) # TODO: Infer loop bounds
  let model = compile[float32](res.target("res"))
  check model.call("res", {
    "image": newTensor([7], @[float32 1, 2, 3, 2, 1, 0, -1]),
  }) == newTensor([5], @[float32 2, float32(7/3), 2, 1, 0])

test "blurOffset":
  let image = input("image")
  res*[x + 1] ++= (
    (image[x] + image[x + 1] + image[x + 2]) / 3.0
  ) | (x in 0..<(image.shape[0] - 2)) # TODO: Infer loop bounds
  res.withShape([image.shape[0]])
  let model = compile[float32](res.target("res"))
  check model.call("res", {
    "image": newTensor([7], @[float32 1, 2, 3, 2, 1, 0, -1]),
  }) == newTensor([7], @[float32 0, 2, float32(7/3), 2, 1, 0, 0])

test "singleWrite":
  res*[0] ++= 10.0
  
  let model = compile[float64](res.target("y"))
  check model.call("y") == newTensor([1], @[float64 10])

test "shape":
  res*{it} ++= 1.0 | it
  res.withShape(3, 2, 1)
  
  let model = compile[float64](res.target("y"))
  check model.call("y") == newTensor([3, 2, 1], 1.0)

test "dimensions":
  let inp = input("x")
  res*[0] ++= toScalar(inp.shape[0])
  res[1] ++= toScalar(inp.shape[^2])
  res[2] ++= toScalar(inp.shape[^1])
  res[3] ++= toScalar(inp.shape.len)
  res[4] ++= toScalar(inp.len)
  res.withShape(5)
  
  let model = compile[float64](res.target("y"))
  check model.call("y", {"x": newTensor([1, 2, 3, 4], 0.0)}) == newTensor([5], @[float64 1, 3, 4, 4, 24])
  check model.call("y", {"x": newTensor([2, 3], 0.0)}) == newTensor([5], @[float64 2, 2, 3, 2, 6])

test "extern":
  proc `*`(inp: Fun, factor: float64): Fun =
    result{it} ++= inp{it} * factor | it
  
  proc testWithFactor(factor: float64) =
    let
      model = compile[float64](target(input("x") * factor, "y"))
      x = newTensor[float64]([2, 3], @[float64 1, 2, 3, 4, 5, 6])
    check model.call("y", {"x": x}) == x * factor
  
  for it in -2..2:
    testWithFactor(float64(it))

test "xor":
  randomize(10)
  
  hidden*[y, x] ++= input("x")[y, it] * param([2, 4])[it, x] | (y, x, it)
  hidden[y, x] ++= param([4])[x] | (y, x)
  hiddenRelu*{it} ++= select(hidden{it} <= 0.0, 0.1 * hidden{it}, hidden{it}) | it
  output*[y, x] ++= hiddenRelu[y, it] * param([4, 1])[it, x] | (y, x, it)
  output[y, x] ++= param([1])[x] | (y, x)
  outputSigmoid*{it} ++= 1.0 / (1.0 + exp(-output{it})) | it
  let pred = outputSigmoid.target("predict")
  
  proc optim(param: var Fun, grad: Fun) =
    param{it} ++= -0.1 * grad{it} | it
  loss*[0] ++= sq(pred{it} - input("y"){it}) | it
  let net = loss.target("loss").backprop(optim).target("train")
  
  let model = compile[float32](net)
  
  let
    trainX = newTensor([4, 2], @[float32 0, 0, 0, 1, 1, 0, 1, 1])
    trainY = newTensor([4, 1], @[float32 0, 1, 1, 0])
  
  for epoch in 0..<1000:
    model.apply("train", {"x": trainX, "y": trainY})
  
  check squares(model.call("predict", {"x": trainX}) - trainY).sum() < 0.1

test "customGrad":
  let inp = input("inp")
  identity*{x} ++= inp{x} | x do:
    customGrad:
      grad(inp){x} ++= inp{x} * 2.0 * grad(identity){x} | x
  
  let
    graph = identity
      .target("identity")
      .backwards()
      .grad(inp)
      .target("grad")
    model = compile[float32](graph)
  
  block:
    let tensor = newTensor[float32]([2, 2], @[float32 1, 2, 3, 4])
    check model.call("identity", {"inp": tensor}) == tensor
    check model.call("grad", {"inp": tensor}) == tensor * 2

test "dynamicAst":
  proc elementwisePow(fun: Fun, n: int): Fun =
    var prod: Scalar = 1.0
    for _ in 0..<n:
      prod = prod * fun{iteratorLiteral("it")}
    result{it} ++= prod | it
    result.copyShape(fun)
  
  let x = newTensor([3, 2], @[float32 1, 2, 3, 4, 5, 6])
  var expectedY = newTensor[float32]([3, 2], 1)
  for n in 0..<2:
    let
      model = compile[float32](input("x").elementwisePow(n).target("y"))
      y = model.call("y", {"x": x})
    check squares(y - expectedY).sum() < 0.001
    for it in 0..<expectedY.len:
      expectedY{it} *= x{it}

test "array":
  res*[x] ++= (
    let arr = literal([1.0, 2.0, 3.0]);
    arr[x] + toScalar(arr.len)
  ) | x
  res.withShape(3)
  
  let model = compile[float32](res.target("y"))
  check model.call("y", []) == newTensor([3], @[float32 4, 5, 6])

test "nestedArray":
  res*[y, x] ++= (
    let arr = literal([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0]
    ]);
    arr[y][x]
  ) | (y, x)
  res.withShape(3, 3)
  
  let model = compile[float32](res.target("y"))
  check model.call("y", []) == newTensor([3, 3], @[float32 1, 2, 3, 4, 5, 6, 7, 8, 9])

test "loopBounds":
  res*[x] ++= 1.0 | (x in 2..<4)
  res[x] ++= -1.0 | (x in 0..<1)
  res[x] ++= -2.0 | (x in 1..<1)
  res.withShape(5)
  let model = compile[float32](res.target("res"))
  check model.call("res") == newTensor([5], @[float32 -1, 0, 1, 1, 0])

test "derive/polynomial":
  let x = input("x")
  y*{it} ++= sq(x{it}) + 2.0 * x{it} + 1.0 | it
  let model = compile[float32](y.backwards().grad(x).target("x^2+2x+1"))
  
  block:
    let x = Tensor.linspace(-8'f32..8'f32, 17)
    check model.call("x^2+2x+1", {"x": x}) == x * 2.0 + Tensor.new([17], 2'f32)

test "derive/trigonometry":
  let x = input("x")
  a*{it} ++= sin(x{it}) | it
  b*{it} ++= cos(x{it}) | it
  let model = compile[float32]([
    a.backwards().grad(x).target("sin"),
    b.backwards().grad(x).target("cos")
  ])
  
  block:
    let x = Tensor.linspace(-8'f32..8'f32, 17)
    check model.call("sin", {"x": x}) == cos(x)
    check model.call("cos", {"x": x}) == -sin(x)

test "derive/exp":
  let x = input("x")
  a*{it} ++= exp(x{it}) | it
  b*{it} ++= exp(2.0 * x{it}) | it
  c*{it} ++= pow(x{it}, 3.0) | it
  d*{it} ++= pow(2.0, x{it}) | it
  e*{it} ++= pow(x{it}, x{it}) | it
  
  let model = compile[float32]([
    a.backwards().grad(x).target("exp(x)"),
    b.backwards().grad(x).target("exp(2x)"),
    c.backwards().grad(x).target("x^3"),
    d.backwards().grad(x).target("2^x"),
    e.backwards().grad(x).target("x^x")
  ])
  
  block:
    let x = Tensor.linspace(-8'f32..8'f32, 17)
    check model.call("exp(x)", {"x": x}) == exp(x)
    check model.call("exp(2x)", {"x": x}) == exp(2'f32 * x) * 2.0
    check model.call("x^3", {"x": x}) == squares(x) * 3.0
    check model.call("2^x", {"x": x}) == pow(2'f32, x) * ln(2'f32)
    
    let
      x2 = Tensor.linspace(1'f32..8'f32, 5)
      e = x2.mapIt(pow(it, it) * (ln(it) + 1.0))
    check sum(squares(model.call("x^x", {"x": x2}) - e)) < 0.01

test "derive/log":
  let x = input("x")
  a*{it} ++= ln(x{it}) | it
  
  let model = compile[float32]([
    a.backwards().grad(x).target("ln(x)"),
  ])
  
  block:
    let x = Tensor.linspace(1'f32..8'f32, 8)
    check model.call("ln(x)", {"x": x}) == 1'f32 / x
