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

import exprgrad

proc matmul(a, b: Fun): Fun =
  result[y, x] ++= a[y, it] * b[it, x] | (x, y, it)

let model = compile[float32](matmul(input("a"), input("b")).target("matmul"))

let
  a = Tensor.new([3, 2], @[float32 1, 2, 3, 4, 5, 6])
  b = Tensor.new([2, 3], @[float32 1, 2, 3, 4, 5, 6])
  expected = a * b
  computed = model.call("matmul", {"a": a, "b": b})

echo expected
echo computed
echo computed == expected
