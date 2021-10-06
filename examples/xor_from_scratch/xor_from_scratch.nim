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
import exprgrad, exprgrad/layers/[base, dnn]
randomize(10)

hidden*[x, y] ++= input("x")[it, y] * param([4, 2])[x, it]
hidden[x, y] ++= param([4])[x]
hidden_relu*{it} ++= select(hidden{it} <= 0.0, 0.1 * hidden{it}, hidden{it})
output*[x, y] ++= hidden_relu[it, y] * param([1, 4])[x, it]
output[x, y] ++= param([1])[x]
output_sigmoid*{it} ++= 1.0 / (1.0 + exp(-output{it})) 
let pred = output_sigmoid.target("predict")

proc optim(param: var Fun, grad: Fun) =
  param{it} ++= -0.1 * grad{it}
loss*[0] ++= sq(pred{it} - input("y"){it})
let net = loss.target("loss").backprop(optim).target("train")

let model = compile[float32](net)

let
  train_x = new_tensor([2, 4], @[float32 0, 0, 0, 1, 1, 0, 1, 1])
  train_y = new_tensor([1, 4], @[float32 0, 1, 1, 0])

for epoch in 0..<5000:
  model.apply("train", {"x": train_x, "y": train_y})

echo model.call("predict", {"x": train_x})
