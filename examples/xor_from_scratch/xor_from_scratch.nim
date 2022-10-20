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

import std/random
import exprgrad
randomize(10)

# Layer 1
hidden*[y, x] ++= input("x")[y, it] * param([2, 4])[it, x] | (y, x, it)
hidden[y, x] ++= param([4])[x] | (y, x)
hiddenRelu*{it} ++= select(hidden{it} <= 0.0, 0.1 * hidden{it}, hidden{it}) | it
# Layer 2
output*[y, x] ++= hiddenRelu[y, it] * param([4, 1])[it, x] | (y, x, it)
output[y, x] ++= param([1])[x] | (y, x)
outputSigmoid*{it} ++= 1.0 / (1.0 + exp(-output{it})) | it
let pred = outputSigmoid.target("predict")
loss*[0] ++= sq(pred{it} - input("y"){it}) | it # Loss

proc optim(param: var Fun, grad: Fun) =
  param{it} ++= -0.1 * grad{it} | it

let net = loss.target("loss").backprop(optim).target("train") # Train

let model = compile[float32](net)

let
  trainX = Tensor.new([4, 2], @[float32 0, 0, 0, 1, 1, 0, 1, 1])
  trainY = Tensor.new([4, 1], @[float32 0, 1, 1, 0])

for epoch in 0..<5000:
  model.apply("train", {"x": trainX, "y": trainY})

echo model.call("predict", {"x": trainX})
