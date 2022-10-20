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

proc loadMnist[T](path: string):
    tuple[trainX, trainY, testX, testY: Tensor[T]] =
  let
    testXPath = path / "t10k-images-idx3-ubyte"
    testYPath = path / "t10k-labels-idx1-ubyte"
    trainXPath = path / "train-images-idx3-ubyte"
    trainYPath = path / "train-labels-idx1-ubyte"
  result.testX = loadIdx[uint8](testXPath).reshape([-1, 28 * 28]).convert(T).remap(0, 255, 0.1, 0.9)
  result.testY = loadIdx[uint8](testYPath).oneHot(10).convert(T)
  result.trainX = loadIdx[uint8](trainXPath).reshape([-1, 28 * 28]).convert(T).remap(0, 255, 0.1, 0.9)
  result.trainY = loadIdx[uint8](trainYPath).oneHot(10).convert(T)

let (trainX, trainY, testX, testY) = loadMnist[float32]("data")

proc genLoss(labels: Fun): Fun =
  result[0] ++= sq(labels{it}) / toScalar(labels.shape[0]) | it

let
  gen = input("seed")
    .dense(32, 64).leakyRelu(0.01)
    .dense(64, 128).leakyRelu(0.01)
    .dense(128, 28 * 28)
    .sigmoid()
    .target("gen")
  discr = cond({"fit.gen": gen, "loss.gen": gen}, input("samples"))
    .dense(28 * 28, 128).leakyRelu(0.01)
    .dense(128, 64).leakyRelu(0.01)
    .dense(64, 1)
    .sigmoid()
    .target("discr")
  fitGen = discr.genLoss().target("loss.gen")
    .backwards()
    .optimize(gen.params, gradientDescent.makeOpt(0.1))
    .target("fit.gen")
  fitDiscr = discr.mse(input("labels")).target("loss.discr")
    .backwards()
    .optimize(
      discr.params.difference(gen.params),
      gradientDescent.makeOpt(0.1)
    )
    .target("fit.discr")
  model = compile[float32]([gen, discr, fitGen, fitDiscr])

const
  SEED_RANGE = float32(0.0)..float32(1.0)
  LOG_TIME = 10

var epoch = 0
while true:
  if epoch mod LOG_TIME == 0:
    model.call("gen", {
      "seed": Tensor.rand([1, 32], SEED_RANGE)
    }).reshape([-1, 28, 1]).remap(0, 1, 0, 255).convert(uint8).savePpm("sample.ppm")
  
  model.epoch += 1
  block trainDiscr:
    const COUNT = 32
    let
      seed = Tensor.rand([COUNT, 32], SEED_RANGE)
      samples = concatFirst(
        model.call("gen", {"seed": seed}),
        trainX.selectRandomSamples(COUNT)
      )
      labels = concatFirst(
        Tensor.new([COUNT, 1], float32(1)),
        Tensor.new([COUNT, 1], float32(0))
      )
    model.apply("fit.discr", {"samples": samples, "labels": labels})
    if epoch mod LOG_TIME == 0:
      echo "Discr Loss: ", model.call("loss.discr", {"samples": samples, "labels": labels})
  
  block trainGen:
    const COUNT = 64
    let seed = Tensor.rand([COUNT, 32], SEED_RANGE)
    model.apply("fit.gen", {"seed": seed})
    if epoch mod LOG_TIME == 0:
      echo "Generator Loss: ", model.call("loss.gen", {"seed": seed})
  
  epoch += 1
