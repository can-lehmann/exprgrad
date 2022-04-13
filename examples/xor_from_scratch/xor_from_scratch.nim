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
hidden_relu*{it} ++= select(hidden{it} <= 0.0, 0.1 * hidden{it}, hidden{it}) | it
# Layer 2
output*[y, x] ++= hidden_relu[y, it] * param([4, 1])[it, x] | (y, x, it)
output[y, x] ++= param([1])[x] | (y, x)
output_sigmoid*{it} ++= 1.0 / (1.0 + exp(-output{it})) | it
let pred = output_sigmoid.target("predict")
loss*[0] ++= sq(pred{it} - input("y"){it}) | it # Loss

proc optim(param: var Fun, grad: Fun) =
  param{it} ++= -0.1 * grad{it} | it

let net = loss.target("loss").backprop(optim).target("train") # Train

let model = compile[float32](net)

let
  train_x = new_tensor([4, 2], @[float32 0, 0, 0, 1, 1, 0, 1, 1])
  train_y = new_tensor([4, 1], @[float32 0, 1, 1, 0])

for epoch in 0..<5000:
  model.apply("train", {"x": train_x, "y": train_y})

echo model.call("predict", {"x": train_x})
