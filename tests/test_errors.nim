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

# Unit-tests for exprgrad's compiler errors

import std/[random, tables]
import exprgrad, exprgrad/[ir, irprint]
import ../tools/test_framework

test "invalidTarget":
  checkException RuntimeError:
    let model = compile[float32]()
    discard model.call("myTarget")

#[ TODO
test "missingInput":
  checkException RuntimeError:
    res*{x} ++= input("x"){x} | x
    let model = compile[float32](res.target("y"))
    discard model.call("y")
  
  checkException RuntimeError:
    let model = compile[float32](input("x").target("y"))
    discard model.call("y")
]#

test "invalidInput":
  checkException RuntimeError:
    let model = compile[float32](input("x").target("y"))
    discard model.call("y", {
      "x": Tensor[float32].new([2, 3]),
      "abc": Tensor[float32].new([2, 3])
    })

test "staticShapeMismatch":
  checkException ShapeError:
    let model = compile[float32](input("x", [2, 3]).target("y"))
    discard model.call("y", {"x": Tensor[float32].new([10, 10])})

test "underconstrainedShape":
  checkException ShapeError:
    res*{x} ++= 1.0 | x
    discard compile[float32](res.target("y"))
  
  checkException ShapeError:
    res*[x] ++= 1.0 | x
    discard compile[float32](res.target("y"))
  
  checkException ShapeError:
    res*[x] ++= input("inp")[y] | (x, y)
    discard compile[float32](res.target("y"))
  
  checkException ShapeError:
    c*{it} ++= input("a"){it} + input("b"){it} | it
    discard compile[float32](c.target("c"))

test "readDimension":
  checkException ShapeError:
    let inp = input("x")
    a*[0] ++= inp[x] | x
    b*[0] ++= a[0, x] | x
    discard compile[float32](b.target("y"))
  
  checkException ShapeError:
    let inp = input("x", [2, 3])
    res*[0] ++= inp[x] | x
    discard compile[float32](res.target("y"))

test "writeDimension":
  checkException ShapeError:
    res*[0] ++= 1.0
    res[0, 0] ++= 1.0
    discard compile[float32](res.target("y"))
  
  checkException ShapeError:
    res*[0] ++= 1.0
    res.withShape(2, 3)
    discard compile[float32](res.target("y"))
