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


import std/[os, sugar, times]
import exprgrad, exprgrad/[io/idxformat, io/ppmformat, layers/base, layers/dnn, graphics/dotgraph, io/serialize]

proc loadMnist[T](path: string):
    tuple[trainX, trainY, testX, testY: Tensor[T]] =
  let
    testXPath = path / "t10k-images-idx3-ubyte"
    testYPath = path / "t10k-labels-idx1-ubyte"
    trainXPath = path / "train-images-idx3-ubyte"
    trainYPath = path / "train-labels-idx1-ubyte"
  result.testX = loadIdx[uint8](testXPath).convert(T) / T(255)
  result.testY = loadIdx[uint8](testYPath).oneHot(10).convert(T)
  result.trainX = loadIdx[uint8](trainXPath).convert(T) / T(255)
  result.trainY = loadIdx[uint8](trainYPath).oneHot(10).convert(T)

let (trainX, trainY, testX, testY) = loadMnist[float32]("data")

let
  net = input("x")
    .reshape([-1, 28, 28, 1])
    .conv2(1, 5, 5, 8)
    .leakyRelu() # Move after maxpool2?
    .maxpool2()
    .conv2(8, 3, 3, 16)
    .leakyRelu()
    .maxpool2()
    .reshape([-1, 16 * 5 * 5])
    .dense(16 * 5 * 5, 10)
    .softmax()
    .target("predict")
    .crossEntropy(input("y"))
    .target("loss")
    .backwards()
    .optimize(adam.makeOpt(eta=0.01))
    .target("fit")
  model = compile[float32](net)
  #model = loadModel[float32]("model.bin")

writeFile("model.gv", model.source.toDotGraph("loss"))

echo testX.shape
echo testY.shape

var epoch = 0
while true:
  let
    start = now()
    testLoss = model.call("loss", {"x": testX, "y": testY})
    stop = now()
  echo stop - start
  echo "Epoch: ", epoch, " Test Loss: ", testLoss
  model.fit("fit", {"x": trainX, "y": trainY})
  model.save("model.bin")
  epoch += 1

