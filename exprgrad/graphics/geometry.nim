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

# Basic types and procedures for working with geometry

import std/math

type
  Vector2*[T] = object
    x*: T
    y*: T
  
  Vec2* = Vector2[float64]
  Index2* = Vector2[int]

template vector_unop(op) =
  proc op*[T](vec: Vector2[T]): Vector2[T] {.inline.} =
    result = Vector2[T](x: op(vec.x), y: op(vec.y))

vector_unop(`-`)
vector_unop(`abs`)
vector_unop(`floor`)
vector_unop(`ceil`)
vector_unop(`round`)

template vector_binop(op) =
  proc op*[T](a, b: Vector2[T]): Vector2[T] {.inline.} =
    result = Vector2[T](x: op(a.x, b.x), y: op(a.y, b.y))

vector_binop(`+`)
vector_binop(`-`)
vector_binop(`*`)
vector_binop(`/`)
vector_binop(`mod`)
vector_binop(`min`)
vector_binop(`max`)

template vector_binop_scalar(op) =
  proc op*[T](a: T, b: Vector2[T]): Vector2[T] {.inline.} =
    result = Vector2[T](x: op(a, b.x), y: op(a, b.y))
  
  proc op*[T](a: Vector2[T], b: T): Vector2[T] {.inline.} =
    result = Vector2[T](x: op(a.x, b), y: op(a.y, b))

vector_binop_scalar(`*`)
vector_binop_scalar(`/`)
vector_binop_scalar(`mod`)
vector_binop_scalar(min)
vector_binop_scalar(max)

template vector_binop_mut(op) =
  proc op*[T](a: var Vector2[T], b: Vector2[T]) {.inline.} =
    op(a.x, b.x)
    op(a.y, b.y)

vector_binop_mut(`+=`)
vector_binop_mut(`-=`)
vector_binop_mut(`*=`)
vector_binop_mut(`/=`)

template vector_binop_mut_scalar(op) =
  proc op*[T](a: var Vector2[T], b: T) {.inline.} =
    op(a.x, b)
    op(a.y, b)

vector_binop_mut_scalar(`*=`)
vector_binop_mut_scalar(`/=`)

{.push inline.}
proc dot*[T](a, b: Vector2[T]): T =
  result = a.x * b.x + a.y * b.y

proc length*[T](vec: Vector2[T]): float64 =
  result = sqrt(float64(vec.x * vec.x + vec.y * vec.y))

proc normalize*[T](vec: Vector2[T]): Vector2[T] =
  result = vec / T(vec.length())
{.pop.}

type
  BoundingBox*[T] = object
    min*: T
    max*: T
  
  BoundingBox2*[T] = BoundingBox[Vector2[T]]
  Box2* = BoundingBox2[float64]

proc size*[T](box: BoundingBox[T]): T {.inline.} = box.max - box.min

