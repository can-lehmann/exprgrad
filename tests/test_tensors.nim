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
  a = newTensor([2, 3], @[int 1, 2, 3, 4, 5, 6])
  b = newTensor([3, 2], @[int 1, 2, 3, 4, 5, 6])
  c = newTensor([2, 2], @[int 22, 28, 49, 64])
  d = newTensor([2, 2], @[int 1, 2, 3, 4])
  e = newTensor([1, 2, 3], @[int 1, 2, 3, 4, 5, 6])

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
  check a - a == newTensor[int](a.shape)
  check max(a, 2 * a) == 2 * a
  check min(a, 2 * a) == a

test "matmul":
  check a * b == c

test "transpose":
  check a.transpose().transpose() == a
  check b.transpose().transpose() == b
  check c.transpose().transpose() == c
  check a.transpose() == newTensor([3, 2], @[int 1, 4, 2, 5, 3, 6])

test "oneHot":
  check newTensor[int]([2], @[0, 1]).oneHot(2) == newTensor[int]([2, 2], @[1, 0, 0, 1])
  check newTensor[int]([2], @[0, 1]).oneHot(3) == newTensor[int]([2, 3], @[1, 0, 0, 0, 1, 0])

test "viewFirst":
  check b.viewFirst(0..0) == newTensor([1, 2], @[1, 2])
  check b.viewFirst(1..2) == newTensor([2, 2], @[3, 4, 5, 6])
  check b.viewFirst(0..0) != b.viewFirst(1..1)
  check a.viewFirst(0..0) == newTensor([1, 3], @[1, 2, 3])

test "selectSamples":
  check b.selectSamples([0, 1, 2]) == b
  check b.selectSamples([1]) == newTensor([1, 2], @[3, 4])
  check b.selectSamples([1, 0]) == newTensor([2, 2], @[3, 4, 1, 2])
  check a.selectSamples([1]) == newTensor([1, 3], @[4, 5, 6])

  check b.selectSamples([1, 2]) == b.viewFirst(1..2)
  check b.selectSamples([2, 1]) != b.viewFirst(1..2)

test "concatFirst":
  check concatFirst(
    newTensor([1], @[1]),
    newTensor([1], @[2])
  ) == newTensor([2], @[1, 2])
  check concatFirst(
    newTensor([1, 2], @[1, 2]),
    newTensor([2, 2], @[3, 4, 5, 6])
  ) == newTensor([3, 2], @[1, 2, 3, 4, 5, 6])
  check concatFirst(e, e).viewFirst(0, 1) == e

test "reshape":
  check b.reshape([2, 3]) == a
  check b.reshape([2, -1]) == a
  check b.reshape([-1, 3]) == a
  check a.reshape([-1, 3]) == a
  check a.reshape([-1]) == b.reshape([-1])
  check a.reshape([-1, 2, 3]) == e

