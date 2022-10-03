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

# Unit-tests for exprgrads built-in dnn layers

import std/random
import exprgrad, exprgrad/layers/[base, dnn]
import ../tools/test_framework

randomize(10)

test "xor":
  let net = input("x")
    .dense(2, 4)
    .leakyRelu()
    .dense(4, 1)
    .sigmoid()
    .target("predict")
    .mse(input("y"))
    .target("loss")
    .backprop(gradientDescent.makeOpt(rate=0.2))
    .target("train")
  
  let
    model = compile[float32](net)
    trainX = newTensor([4, 2], @[float32 0, 0, 0, 1, 1, 0, 1, 1])
    trainY = newTensor([4, 1], @[float32 0, 1, 1, 0])
  
  for it in 0..<2000:
    model.apply("train", {"x": trainX, "y": trainY})
  
  let
    internalLoss = model.call("loss", {"x": trainX, "y": trainY}).sum()
    loss = squares(model.call("predict", {"x": trainX}) - trainY).sum()
  
  check internalLoss < 0.1
  check loss < 0.1
  check abs(loss / float64(trainY.len) - internalLoss) < 0.0001

randomize(10)

test "xor/fit":
  let net = input("x")
    .dense(2, 4)
    .leakyRelu()
    .dense(4, 1)
    .sigmoid()
    .target("predict")
    .mse(input("y"))
    .target("loss")
    .backprop(gradientDescent.makeOpt(rate=0.2))
    .target("train")
  
  let
    model = compile[float32](net)
    trainX = newTensor([4, 2], @[float32 0, 0, 0, 1, 1, 0, 1, 1])
    trainY = newTensor([4, 1], @[float32 0, 1, 1, 0])
  
  for it in 0..<2000:
    model.fit("train", {"x": trainX, "y": trainY}, batchSize=4, logStatus=false)
  
  let
    internalLoss = model.call("loss", {"x": trainX, "y": trainY}).sum()
    loss = squares(model.call("predict", {"x": trainX}) - trainY).sum()
  
  check internalLoss < 0.1
  check loss < 0.1
  check abs(loss / float64(trainY.len) - internalLoss) < 0.0001

