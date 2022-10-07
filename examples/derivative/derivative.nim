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

import exprgrad

let x = input("x")
y*[it] ++= x[it] * x[it] | it

let
  graph = y
    .target("y")
    .backwards()
    .grad(x)
    .target("grad_x")
  model = compile[float32](graph)

let xs = Tensor.linspace(-2'f32..2'f32, 9)

echo "x: ", xs
echo "y: ", model.call("y", { "x": xs })
echo "grad_x: ", model.call("grad_x", { "x": xs })


