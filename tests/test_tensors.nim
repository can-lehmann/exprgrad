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

# Unit-tests for the tensors module

import exprgrad
import ../tools/test_framework

let
  a = Tensor.new([2, 3], @[int 1, 2, 3, 4, 5, 6])
  b = Tensor.new([3, 2], @[int 1, 2, 3, 4, 5, 6])
  c = Tensor.new([2, 2], @[int 22, 28, 49, 64])
  d = Tensor.new([2, 2], @[int 1, 2, 3, 4])
  e = Tensor.new([1, 2, 3], @[int 1, 2, 3, 4, 5, 6])

test "constructors":
  check a == newTensor([2, 3], @[int 1, 2, 3, 4, 5, 6])
  check a == Tensor.new([2, 3], @[int 1, 2, 3, 4, 5, 6])
  check Tensor.new([2, 2], 10) == Tensor.new([2, 2], @[10, 10, 10, 10])
  check Tensor[int].new([2, 2]) == Tensor.new([2, 2], 0)
  check Tensor.rand([3, 2], 10..10) == Tensor.new([3, 2], 10)

test "equality":
  check a == a
  check a != b
  check a != b.transpose()
  check c == c
  check c != a
  check c != b
  check c != d
  check c.shape == d.shape
  check c.shape != a.shape
  check a.shape != b.shape
  check a.shape == b.transpose().shape

test "stringify":
  check $a == "[[1, 2, 3], [4, 5, 6]]"
  check $b == "[[1, 2], [3, 4], [5, 6]]"
  check $c == "[[22, 28], [49, 64]]"
  check $e == "[[[1, 2, 3], [4, 5, 6]]]"

test "access":
  check a[1, 0] == 4
  check a{0} == 1
  check b[1, 1] == 4
  check e[0, 1, 1] == 5
  check e[0, 1, 0] == 4
  check e[0, 0, 1] == 2

test "operators":
  check a * 2 div 2 == a
  check a + a == a * 2
  check a - a == Tensor[int].new(a.shape)
  check max(a, 2 * a) == 2 * a
  check min(a, 2 * a) == a

test "matmul":
  check a * b == c
  check d * a == Tensor.new([2, 3], @[9, 12, 15, 19, 26, 33])

test "transpose":
  check a.transpose().transpose() == a
  check b.transpose().transpose() == b
  check c.transpose().transpose() == c
  check a.transpose() == Tensor.new([3, 2], @[int 1, 4, 2, 5, 3, 6])

test "oneHot":
  check Tensor.new([2], @[0, 1]).oneHot(2) == Tensor.new([2, 2], @[1, 0, 0, 1])
  check Tensor.new([2], @[0, 1]).oneHot(3) == Tensor.new([2, 3], @[1, 0, 0, 0, 1, 0])

test "viewFirst":
  check b.viewFirst(0..0) == Tensor.new([1, 2], @[1, 2])
  check b.viewFirst(1..2) == Tensor.new([2, 2], @[3, 4, 5, 6])
  check b.viewFirst(0..0) != b.viewFirst(1..1)
  check a.viewFirst(0..0) == Tensor.new([1, 3], @[1, 2, 3])
  
  a.viewFirst(0..0).allocShape([1, 3])
  check a == Tensor.new([2, 3], @[1, 2, 3, 4, 5, 6])

test "selectSamples":
  check b.selectSamples([0, 1, 2]) == b
  check b.selectSamples([1]) == Tensor.new([1, 2], @[3, 4])
  check b.selectSamples([1, 0]) == Tensor.new([2, 2], @[3, 4, 1, 2])
  check a.selectSamples([1]) == Tensor.new([1, 3], @[4, 5, 6])

  check b.selectSamples([1, 2]) == b.viewFirst(1..2)
  check b.selectSamples([2, 1]) != b.viewFirst(1..2)

test "concatFirst":
  check concatFirst(
    Tensor.new([1], @[1]),
    Tensor.new([1], @[2])
  ) == Tensor.new([2], @[1, 2])
  check concatFirst(
    Tensor.new([1, 2], @[1, 2]),
    Tensor.new([2, 2], @[3, 4, 5, 6])
  ) == Tensor.new([3, 2], @[1, 2, 3, 4, 5, 6])
  check concatFirst(e, e).viewFirst(0, 1) == e

test "reshape":
  check b.reshape([2, 3]) == a
  check b.reshape([2, -1]) == a
  check b.reshape([-1, 3]) == a
  check a.reshape([-1, 3]) == a
  check a.reshape([-1]) == b.reshape([-1])
  check a.reshape([-1, 2, 3]) == e

