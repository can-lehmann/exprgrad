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
  a = new_tensor([3, 2], @[int 1, 2, 3, 4, 5, 6])
  b = new_tensor([2, 3], @[int 1, 2, 3, 4, 5, 6])
  c = new_tensor([2, 2], @[int 22, 28, 49, 64])
  d = new_tensor([2, 2], @[int 1, 2, 3, 4])

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

test "access":
  check a[0, 1] == 4
  check a{0} == 1
  check b[1, 1] == 4

test "operators":
  check a * 2 div 2 == a
  check a + a == a * 2
  check a - a == new_tensor[int](a.shape)
  check max(a, 2 * a) == 2 * a
  check min(a, 2 * a) == a

test "matmul":
  check a * b == c

test "transpose":
  check a.transpose().transpose() == a
  check b.transpose().transpose() == b
  check c.transpose().transpose() == c

test "view_last":
  check b.view_last(0..0) == new_tensor([2, 1], @[1, 2])
  check b.view_last(1..2) == new_tensor([2, 2], @[3, 4, 5, 6])
  check b.view_last(0..0) != b.view_last(1..1)
  check a.view_last(0..0) == new_tensor([3, 1], @[1, 2, 3])

test "select_samples":
  check b.select_samples([0, 1, 2]) == b
  check b.select_samples([1]) == new_tensor([2, 1], @[3, 4])
  check b.select_samples([1, 0]) == new_tensor([2, 2], @[3, 4, 1, 2])
  check a.select_samples([1]) == new_tensor([3, 1], @[4, 5, 6])

  check b.select_samples([1, 2]) == b.view_last(1..2)
  check b.select_samples([2, 1]) != b.view_last(1..2)

test "concat_last":
  check concat_last(
    new_tensor([1], @[1]),
    new_tensor([1], @[2])
  ) == new_tensor([2], @[1, 2])
  check concat_last(
    new_tensor([2, 1], @[1, 2]),
    new_tensor([2, 2], @[3, 4, 5, 6])
  ) == new_tensor([2, 3], @[1, 2, 3, 4, 5, 6])

test "reshape":
  check b.reshape([3, 2]) == a
  check b.reshape([-1, 2]) == a
  check b.reshape([3, -1]) == a
  check a.reshape([3, -1]) == a
  check a.reshape([-1]) == b.reshape([-1])

