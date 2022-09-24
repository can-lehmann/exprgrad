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

import std/[math, hashes]

type
  Vector2*[T] = object
    x*: T
    y*: T
  
  Vec2* = Vector2[float64]
  Index2* = Vector2[int]
  
  Vector3*[T] = object
    x*: T
    y*: T
    z*: T
  
  Vec3* = Vector3[float64]
  Index3* = Vector3[int]
  
  Vector4*[T] = object
    x*: T
    y*: T
    z*: T
    w*: T
  
  Vec4* = Vector4[float64]
  Index4* = Vector4[int]

template vector_unop(op) =
  proc op*[T](vec: Vector2[T]): Vector2[T] {.inline.} =
    result = Vector2[T](x: op(vec.x), y: op(vec.y))
  
  proc op*[T](vec: Vector3[T]): Vector3[T] {.inline.} =
    result = Vector3[T](x: op(vec.x), y: op(vec.y), z: op(vec.z))
  
  proc op*[T](vec: Vector4[T]): Vector4[T] {.inline.} =
    result = Vector4[T](x: op(vec.x), y: op(vec.y), z: op(vec.z), w: op(vec.w))

vector_unop(`-`)
vector_unop(`abs`)
vector_unop(`floor`)
vector_unop(`ceil`)
vector_unop(`round`)

template vector_binop(op) =
  proc op*[T](a, b: Vector2[T]): Vector2[T] {.inline.} =
    result = Vector2[T](x: op(a.x, b.x), y: op(a.y, b.y))
  
  proc op*[T](a, b: Vector3[T]): Vector3[T] {.inline.} =
    result = Vector3[T](x: op(a.x, b.x), y: op(a.y, b.y), z: op(a.z, b.z))
  
  proc op*[T](a, b: Vector4[T]): Vector4[T] {.inline.} =
    result = Vector4[T](x: op(a.x, b.x), y: op(a.y, b.y), z: op(a.z, b.z), z: op(a.w, b.w))

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

  proc op*[T](a: T, b: Vector3[T]): Vector3[T] {.inline.} =
    result = Vector3[T](x: op(a, b.x), y: op(a, b.y), z: op(a, b.z))
  
  proc op*[T](a: Vector3[T], b: T): Vector3[T] {.inline.} =
    result = Vector3[T](x: op(a.x, b), y: op(a.y, b), z: op(a.z, b))
  
  proc op*[T](a: T, b: Vector4[T]): Vector4[T] {.inline.} =
    result = Vector4[T](x: op(a, b.x), y: op(a, b.y), z: op(a, b.z), w: op(a, b.w))
  
  proc op*[T](a: Vector4[T], b: T): Vector4[T] {.inline.} =
    result = Vector4[T](x: op(a.x, b), y: op(a.y, b), z: op(a.z, b), w: op(a.w, b))

vector_binop_scalar(`*`)
vector_binop_scalar(`/`)
vector_binop_scalar(`mod`)
vector_binop_scalar(min)
vector_binop_scalar(max)

template vector_binop_mut(op) =
  proc op*[T](a: var Vector2[T], b: Vector2[T]) {.inline.} =
    op(a.x, b.x)
    op(a.y, b.y)
  
  proc op*[T](a: var Vector3[T], b: Vector2[T]) {.inline.} =
    op(a.x, b.x)
    op(a.y, b.y)
    op(a.z, b.z)
  
  proc op*[T](a: var Vector4[T], b: Vector2[T]) {.inline.} =
    op(a.x, b.x)
    op(a.y, b.y)
    op(a.z, b.z)
    op(a.w, b.w)

vector_binop_mut(`+=`)
vector_binop_mut(`-=`)
vector_binop_mut(`*=`)
vector_binop_mut(`/=`)

template vector_binop_mut_scalar(op) =
  proc op*[T](a: var Vector2[T], b: T) {.inline.} =
    op(a.x, b)
    op(a.y, b)
  
  proc op*[T](a: var Vector3[T], b: T) {.inline.} =
    op(a.x, b)
    op(a.y, b)
    op(a.z, b)
  
  proc op*[T](a: var Vector4[T], b: T) {.inline.} =
    op(a.x, b)
    op(a.y, b)
    op(a.z, b)
    op(a.w, b)

vector_binop_mut_scalar(`*=`)
vector_binop_mut_scalar(`/=`)

{.push inline.}
proc dot*[T](a, b: Vector2[T]): T =
  result = a.x * b.x + a.y * b.y

proc dot*[T](a, b: Vector3[T]): T =
  result = a.x * b.x + a.y * b.y + a.z * b.z

proc dot*[T](a, b: Vector4[T]): T =
  result = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w

proc length*[T](vec: Vector2[T]): float64 =
  result = sqrt(float64(vec.x * vec.x + vec.y * vec.y))

proc length*[T](vec: Vector3[T]): float64 =
  result = sqrt(float64(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z))

proc length*[T](vec: Vector4[T]): float64 =
  result = sqrt(float64(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w))

proc normalize*[T](vec: Vector2[T]): Vector2[T] = vec / T(vec.length())
proc normalize*[T](vec: Vector3[T]): Vector3[T] = vec / T(vec.length())
proc normalize*[T](vec: Vector4[T]): Vector4[T] = vec / T(vec.length())

proc cross*[T](a, b: Vector3[T]): Vector3[T] =
  result = Vector3[T](
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  )

proc to_vec2*(index: Index2): Vec2 =
  result = Vec2(x: float64(index.x), y: float64(index.y))

proc to_index2*(vec: Vec2): Index2 =
  result = Index2(x: int(vec.x), y: int(vec.y))
{.pop.}

type Axis* = distinct int

const
  AxisX* = Axis(0)
  AxisY* = Axis(1)
  AxisZ* = Axis(2)
  AxisW* = Axis(3)

{.push inline.}
proc `[]`*[T](vec: Vector2[T], axis: Axis): T = [vec.x, vec.y][int(axis)]
proc `[]`*[T](vec: Vector3[T], axis: Axis): T = [vec.x, vec.y, vec.z][int(axis)]
proc `[]`*[T](vec: Vector4[T], axis: Axis): T = [vec.x, vec.y, vec.z, vec.w][int(axis)]

proc `[]`*[T](vec: var Vector2[T], axis: Axis): var T = [vec.x.addr, vec.y.addr][int(axis)][]
proc `[]`*[T](vec: var Vector3[T], axis: Axis): var T = [vec.x.addr, vec.y.addr, vec.z.addr][int(axis)][]
proc `[]`*[T](vec: var Vector4[T], axis: Axis): var T = [vec.x.addr, vec.y.addr, vec.z.addr, vec.w.addr][int(axis)][]

proc `[]=`*[T](vec: var Vector2[T], axis: Axis, value: T) = [vec.x.addr, vec.y.addr][int(axis)][] = value
proc `[]=`*[T](vec: var Vector3[T], axis: Axis, value: T) = [vec.x.addr, vec.y.addr, vec.z.addr][int(axis)][] = value
proc `[]=`*[T](vec: var Vector4[T], axis: Axis, value: T) = [vec.x.addr, vec.y.addr, vec.z.addr, vec.w.addr][int(axis)][] = value
{.pop.}

type
  BoundingBox*[T] = object
    min*: T
    max*: T
  
  BoundingBox2*[T] = BoundingBox[Vector2[T]]
  Inter* = BoundingBox[float64]
  Box2* = BoundingBox2[float64]
  
  BoundingBox3*[T] = BoundingBox[Vector3[T]]
  Box3* = BoundingBox3[float64]

proc size*[T](box: BoundingBox[T]): T {.inline.} = box.max - box.min
proc center*[T](box: BoundingBox[T]): T {.inline.} = (box.max + box.min) / 2

proc x_inter*[T](box: BoundingBox2[T]): BoundingBox[T] =
  result = BoundingBox[T](min: box.min.x, max: box.max.x)

proc y_inter*[T](box: BoundingBox2[T]): BoundingBox[T] =
  result = BoundingBox[T](min: box.min.y, max: box.max.y)

type
  StaticMatrix*[T; H, W: static[int]] = object
    data*: array[H * W, T]
  
  Matrix2*[T] = StaticMatrix[T, 2, 2]
  Mat2* = StaticMatrix[float64, 2, 2]
  
  Matrix3*[T] = StaticMatrix[T, 3, 3]
  Mat3* = StaticMatrix[float64, 3, 3]
  
  Matrix4*[T] = StaticMatrix[T, 4, 4]
  Mat4* = StaticMatrix[float64, 4, 4]

{.push inline.}
proc `[]`*[T, H, W](mat: StaticMatrix[T, H, W], y, x: int): T = mat.data[x + y * W]
proc `[]`*[T, H, W](mat: var StaticMatrix[T, H, W], y, x: int): var T = mat.data[x + y * W]
proc `[]=`*[T, H, W](mat: var StaticMatrix[T, H, W], y, x: int, value: T) = mat.data[x + y * W] = value
{.pop.}

proc transpose*[T, H, W](matrix: StaticMatrix[T, W, H]): StaticMatrix[T, H, W] =
  for y in 0..<H:
    for x in 0..<W:
      result[y, x] = matrix[x, y]

proc `*`*[T, H, W, N](a: StaticMatrix[T, H, N],
                      b: StaticMatrix[T, N, W]): StaticMatrix[T, H, W] =
  for y in 0..<H:
    for it in 0..<N:
      for x in 0..<W:
        result[y, x] += a[y, it] * b[it, x]

proc `[]`*[T, H, W](mat: StaticMatrix[T, H, W],
                    y_slice: static[HSlice[int, int]],
                    x_slice: static[HSlice[int, int]]): auto =
  result = StaticMatrix[T, y_slice.b - y_slice.a + 1, x_slice.b - x_slice.a + 1]()
  for y in y_slice:
    for x in y_slice:
      result[y - y_slice.a, x - x_slice.a] = mat[y, x]

proc det*[T](mat: Matrix2[T]): T =
  result = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]

proc det*[T](mat: Matrix3[T]): T =
  for x in 0..<3:
    var
      a = T(1)
      b = T(-1)
    for y in 0..<3:
      a *= mat[y, (x + y) mod 3]
      b *= mat[2 - y, (x + y) mod 3]
    result += a + b

proc invert*[T](mat: Matrix3[T]): Matrix3[T] =
  template wrap(x, y: int): int =
    var res = x mod y
    if res < 0:
      res += y
    res
  
  template at(y, x: int): T = mat[wrap(y, 3), wrap(x, 3)]
  
  let d = det(mat)
  for y in 0..<3:
    for x in 0..<3:
      let
        a = at(x + 1, y + 1) * at(x - 1, y - 1)
        b = at(x + 1, y - 1) * at(x - 1, y + 1)
      result[y, x] = (a - b) / d

proc identity*[T, N](_: typedesc[StaticMatrix[T, N, N]]): StaticMatrix[T, N, N] =
  for it in 0..<N:
    result[it, it] = T(1)

proc translate*[T](typ: typedesc[Matrix4[T]], offset: Vector3[T]): Matrix4[T] =
  result = typ.identity()
  result[0, 3] = offset.x
  result[1, 3] = offset.y
  result[2, 3] = offset.z

proc scale*[T](typ: typedesc[Matrix4[T]], factors: Vector3[T]): Matrix4[T] =
  result[0, 0] = factors.x
  result[1, 1] = factors.y
  result[2, 2] = factors.z
  result[3, 3] = T(1)

proc to_matrix*[T](vec: Vector2[T]): StaticMatrix[T, 2, 1] =
  result = StaticMatrix[T, 2, 1](data: [vec.x, vec.y])

proc to_matrix*[T](vec: Vector3[T]): StaticMatrix[T, 3, 1] =
  result = StaticMatrix[T, 3, 1](data: [vec.x, vec.y, vec.z])

proc to_matrix*[T](vec: Vector4[T]): StaticMatrix[T, 4, 1] =
  result = StaticMatrix[T, 4, 1](data: [vec.x, vec.y, vec.z, vec.w])

proc to_vector3*[T](mat: StaticMatrix[T, 3, 1]): Vector3[T] =
  result = Vector3[T](x: mat.data[0], y: mat.data[1], z: mat.data[2])

template define_unit(T, Base: untyped, sym: string) =
  type T* = distinct Base
  
  proc `+`*(a, b: T): T {.borrow.}
  proc `-`*(a, b: T): T {.borrow.}
  proc `-`*(x: T): T {.borrow.}
  proc `/`*(a, b: T): float64 {.borrow.}
  
  proc `*`*(a: float64, b: T): T {.borrow.}
  proc `*`*(a: T, b: float64): T {.borrow.}
  
  proc `/`*(a: T, b: float64): T {.borrow.}
  
  proc `==`*(a, b: T): bool {.borrow.}
  proc `<`*(a, b: T): bool {.borrow.}
  proc `<=`*(a, b: T): bool {.borrow.}
  
  proc `+=`*(a: var T, b: T) {.borrow.}
  proc `-=`*(a: var T, b: T) {.borrow.}
  
  proc min*(a, b: T): T {.borrow.}
  proc max*(a, b: T): T {.borrow.}
  
  proc hash*(a: T): Hash {.borrow.}
  
  proc `$`*(x: T): string =
    result = $float64(x) & sym

define_unit(Deg, float64, "°")
define_unit(Rad, float64, "rad")

converter to_rad*(deg: Deg): Rad =
  result = Rad(float64(deg) / 180 * PI)

converter to_deg*(rad: Rad): Deg =
  result = Deg(float64(rad) / PI * 180)

proc sin*(rad: Rad): float64 = sin(float64(rad))
proc cos*(rad: Rad): float64 = cos(float64(rad))
proc tan*(rad: Rad): float64 = tan(float64(rad))

proc rotate_x*(_: typedesc[Mat3], angle: Rad): Mat3 =
  result = Mat3(data: [
    float64 1, 0, 0,
    0, cos(angle), -sin(angle),
    0, sin(angle), cos(angle)
  ])

proc rotate_y*(_: typedesc[Mat3], angle: Rad): Mat3 =
  result = Mat3(data: [
    cos(angle), 0, sin(angle),
    0, 1, 0,
    -sin(angle), 0, cos(angle)
  ])

proc rotate_z*(_: typedesc[Mat3], angle: Rad): Mat3 =
  result = Mat3(data: [
    cos(angle), -sin(angle), 0,
    sin(angle), cos(angle), 0,
    0, 0, 1
  ])

proc rotate_x*(_: typedesc[Mat4], angle: Rad): Mat4 =
  result = Mat4(data: [
    float64 1, 0, 0, 0,
    0, cos(angle), -sin(angle), 0,
    0, sin(angle), cos(angle), 0,
    0, 0, 0, 1
  ])

proc rotate_y*(_: typedesc[Mat4], angle: Rad): Mat4 =
  result = Mat4(data: [
    cos(angle), 0, sin(angle), 0,
    0, 1, 0, 0,
    -sin(angle), 0, cos(angle), 0,
    0, 0, 0, 1
  ])

proc rotate_z*(_: typedesc[Mat4], angle: Rad): Mat4 =
  result = Mat4(data: [
    cos(angle), -sin(angle), 0, 0,
    sin(angle), cos(angle), 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ])
