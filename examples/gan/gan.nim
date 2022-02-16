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

import std/[random, sets, os]
import exprgrad, exprgrad/layers/[base, dnn]
import exprgrad/io/[idxformat, ppmformat]
randomize()

proc load_mnist*[T](path: string):
    tuple[train_x, train_y, test_x, test_y: Tensor[T]] =
  let
    test_x_path = path / "t10k-images-idx3-ubyte"
    test_y_path = path / "t10k-labels-idx1-ubyte"
    train_x_path = path / "train-images-idx3-ubyte"
    train_y_path = path / "train-labels-idx1-ubyte"
  result.test_x = load_idx[uint8](test_x_path).reshape([-1, 28 * 28]).convert(T).remap(0, 255, 0.1, 0.9)
  result.test_y = load_idx[uint8](test_y_path).one_hot(10).convert(T)
  result.train_x = load_idx[uint8](train_x_path).reshape([-1, 28 * 28]).convert(T).remap(0, 255, 0.1, 0.9)
  result.train_y = load_idx[uint8](train_y_path).one_hot(10).convert(T)

let (train_x, train_y, test_x, test_y) = load_mnist[float32]("data")

proc gen_loss(labels: Fun): Fun =
  result[0] ++= sq(labels{it}) / to_scalar(labels.shape[^1])

let
  gen = input("seed")
    .dense(32, 64).leaky_relu(0.01)
    .dense(64, 128).leaky_relu(0.01)
    .dense(128, 28 * 28)
    .sigmoid()
    .target("gen")
  discr = cond({"fit.gen": gen, "loss.gen": gen}, input("samples"))
    .dense(28 * 28, 128).leaky_relu(0.01)
    .dense(128, 64).leaky_relu(0.01)
    .dense(64, 1)
    .sigmoid()
    .target("discr")
  fit_gen = discr.gen_loss().target("loss.gen")
    .backwards()
    .optimize(gen.params, gradient_descent.make_opt(0.1))
    .target("fit.gen")
  fit_discr = discr.mse(input("labels")).target("loss.discr")
    .backwards()
    .optimize(
      discr.params.difference(gen.params),
      gradient_descent.make_opt(0.1)
    )
    .target("fit.discr")
  model = compile[float32](gen, discr, fit_gen, fit_discr)

const
  SEED_RANGE = float32(0.0)..float32(1.0)
  LOG_TIME = 10

var epoch = 0
while true:
  if epoch mod LOG_TIME == 0:
    model.call("gen", {
      "seed": new_rand_tensor([1, 32], SEED_RANGE)
    }).reshape([-1, 28, 1]).remap(0, 1, 0, 255).convert(uint8).save_ppm("sample.ppm")
  
  model.epoch += 1
  block train_discr:
    const COUNT = 32
    let
      seed = new_rand_tensor([COUNT, 32], SEED_RANGE)
      samples = concat_first(
        model.call("gen", {"seed": seed}),
        train_x.select_random_samples(COUNT)
      )
      labels = concat_first(
        new_tensor([COUNT, 1], float32(1)),
        new_tensor([COUNT, 1], float32(0))
      )
    model.apply("fit.discr", {"samples": samples, "labels": labels})
    if epoch mod LOG_TIME == 0:
      echo "Discr Loss: ", model.call("loss.discr", {"samples": samples, "labels": labels})
  
  block train_gen:
    const COUNT = 64
    let seed = new_rand_tensor([COUNT, 32], SEED_RANGE)
    model.apply("fit.gen", {"seed": seed})
    if epoch mod LOG_TIME == 0:
      echo "Generator Loss: ", model.call("loss.gen", {"seed": seed})
  
  epoch += 1
