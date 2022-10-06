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

import std/tables
import geometrymath
import exprgrad, exprgrad/layers/base, exprgrad/io/ppmformat

# Helper functions

proc toVector3(fun: Fun, indices: varargs[Index]): Vector3[Scalar] =
  result = Vector3[Scalar](
    x: fun[@indices & @[literal(0)]],
    y: fun[@indices & @[literal(1)]],
    z: fun[@indices & @[literal(2)]]
  )

proc length(vec: Vector3[Scalar]): Scalar = sqrt(dot(vec, vec))

proc saveImage(image: Tensor[float32], path: string) =
  (image.clamp(0'f32, 1'f32) * 256).convert(uint8).savePpm(path)

# Raytracer

type
  Sphere = object
    geometry: Fun
    color: Fun
  
  Scene = object
    background: Fun
    spheres: seq[Sphere]
    light: Fun
    camera: Fun

proc raycastSphere(sphere: Sphere, dir: Vector3[Scalar]): (Boolean, Scalar, Vector3[Scalar]) =
  # Derivation for ray/sphere intersection
  #
  # |p + d * t| = r
  # (p.x + d.x * t)^2 + (p.y + d.y * t)^2 + (p.z + d.z * t)^2 = r^2
  #
  # p.x^2 + 2 * p.x * d.x * t + d.x^2 * t^2 + 
  # p.y^2 + 2 * p.y * d.y * t + d.y^2 * t^2 + 
  # p.z^2 + 2 * p.z * d.z * t + d.z^2 * t^2 +  = r^2
  # 
  # p.x^2 + p.y^2 + p.z^2 +
  # (p.x * d.x + p.y * d.y + p.z * d.z) * t * 2 +
  # (d.x^2 + d.y^2 + d.z^2) * t^2 = r^2
  
  let
    pos = sphere.geometry.toVector3()
    radius = sphere.geometry[3]
  
  let
    c = dot(pos, pos) - sq(radius)
    b = 2.0 * dot(pos, dir)
    a = dot(dir, dir)
    d = sq(b) - 4.0 * a * c
  result[0] = d >= 0.0
  let e = sqrt(d)
  result[1] = min((b + e) / (2.0 * a), (b - e) / (2.0 * a))
  result[2] = normalize(dir * result[1] - pos)

proc raycast(scene: Scene,
             dir: Vector3[Scalar],
             lightDir: Vector3[Scalar],
             comp: Index,
             viewDistance: float64 = 100.0): Scalar =
  result = scene.background[comp]
  var minDist = literal(viewDistance)
  for sphere in scene.spheres:
    let
      (hit, t, normal) = raycastSphere(sphere, dir)
      isCloser = hit and t > 0.0 and t < minDist
      # TODO: There seems to be some error in the gradient computation here
      # if we swap the arguments of max
      intensity = dot(normal, lightDir).max(0.0)
      color = intensity * sphere.color[comp]
    result = select(isCloser, color, result)
    minDist = select(isCloser, t, minDist)

proc render(scene: Scene, size: Index2): Fun =
  result[y, x, c] ++= (
    let dir = Vector3[Scalar](
      x: toScalar(x) / float64(size.x) - 0.5,
      y: -(toScalar(y) / float64(size.y) - 0.5),
      z: scene.camera[0]
    );
    let lightDir = scene.light.toVector3().normalize();
    raycast(scene, dir, lightDir, c)
  ) | (y, x, c)
  result.withShape(size.y, size.x, 3)

# Inverse Rendering

proc renderTargetImage(): Tensor[float32] =
  let scene = Scene(
    background: input("background", [3]),
    spheres: @[
      Sphere(
        geometry: input("sphere0.geom", [4]),
        color: input("sphere0.color", [3])
      ),
      Sphere(
        geometry: input("sphere1.geom", [4]),
        color: input("sphere1.color", [3])
      )
    ],
    light: input("light", [3]),
    camera: input("camera")
  )
  
  let
    graph = scene
      .render(Index2(x: 128, y: 128))
      .target("render")
    model = compile[float32](graph)
  
  result = model.call("render", {
    "camera": Tensor.new([1], 1'f32),
    "background": Tensor.new([3], @[float32 0.5, 0.5, 0.5]),
    "sphere0.geom": Tensor.new([4], @[float32 0.5, 0.2, 4, 0.5]),
    "sphere0.color": Tensor.new([3], @[float32 1, 0, 0]),
    "sphere1.geom": Tensor.new([4], @[float32 -0.6, -0.35, 3, 0.5]),
    "sphere1.color": Tensor.new([3], @[float32 0, 0, 1]),
    "light": Tensor.new([3], @[float32 1, 1, -0.5])
  }).clamp(0'f32, 1'f32)

proc main() =
  let targetImage = renderTargetImage()
  targetImage.saveImage("target.ppm")
  
  let scene = Scene(
    background: input("background", [3]),
    spheres: @[
      Sphere(
        geometry: input("sphere0.geom", [4]),
        #color: input("sphere0.color", [3])
        color: param([3], initRange=0'f64..1'f64)
      ),
      Sphere(
        geometry: input("sphere1.geom", [4]),
        #color: input("sphere0.color", [3])
        color: param([3], initRange=0'f64..1'f64)
      )
    ],
    light: input("light", [3]),
    #light: param([2], initRange = -1'f64..1'f64).makeLight(),
    camera: input("camera")
  )
  
  let
    graph = scene
      .render(Index2(x: 128, y: 128))
      .target("render")
      .mse(input("target"))
      .target("loss")
      .backprop(gradientDescent.makeOpt(rate=0.01))
      .target("train")
    model = compile[float32](graph)
  
  for it in 0..<100:
    let args = {
      "camera": Tensor.new([1], 1'f32),
      "background": Tensor.new([3], @[float32 0.5, 0.5, 0.5]),
      "sphere0.geom": Tensor.new([4], @[float32 0.5, 0.2, 4, 0.5]),
      "sphere1.geom": Tensor.new([4], @[float32 -0.6, -0.35, 3, 0.5]),
      "light": Tensor.new([3], @[float32 1, 1, -0.5]),
      "target": targetImage
    }
    if it mod 1 == 0:
      echo model.call("loss", args)
      model.call("render", args).saveImage("train/image_" & $it & ".ppm")
    model.apply("train", args)
  echo model.params

when isMainModule:
  main()
