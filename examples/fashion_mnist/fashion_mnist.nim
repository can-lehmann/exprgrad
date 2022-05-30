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

proc load_mnist[T](path: string):
    tuple[train_x, train_y, test_x, test_y: Tensor[T]] =
  let
    test_x_path = path / "t10k-images-idx3-ubyte"
    test_y_path = path / "t10k-labels-idx1-ubyte"
    train_x_path = path / "train-images-idx3-ubyte"
    train_y_path = path / "train-labels-idx1-ubyte"
  result.test_x = load_idx[uint8](test_x_path).convert(T) / T(255)
  result.test_y = load_idx[uint8](test_y_path).one_hot(10).convert(T)
  result.train_x = load_idx[uint8](train_x_path).convert(T) / T(255)
  result.train_y = load_idx[uint8](train_y_path).one_hot(10).convert(T)

let (train_x, train_y, test_x, test_y) = load_mnist[float32]("data")

let
  net = input("x")
    .reshape([-1, 28, 28, 1])
    .conv2(1, 5, 5, 8)
    .leaky_relu() # Move after maxpool2?
    .maxpool2()
    .conv2(8, 3, 3, 16)
    .leaky_relu()
    .maxpool2()
    .reshape([-1, 16 * 5 * 5])
    .dense(16 * 5 * 5, 10)
    .softmax()
    .target("predict")
    .cross_entropy(input("y"))
    .target("loss")
    .backwards()
    .optimize(adam.make_opt(eta=0.01))
    .target("fit")
  model = compile[float32](net)
  #model = load_model[float32]("model.bin")

write_file("model.gv", model.source.to_dot_graph("loss"))

echo test_x.shape
echo test_y.shape

var epoch = 0
while true:
  let
    start = now()
    test_loss = model.call("loss", {"x": test_x, "y": test_y})
    stop = now()
  echo stop - start
  echo "Epoch: ", epoch, " Test Loss: ", test_loss
  model.fit("fit", {"x": train_x, "y": train_y})
  model.save("model.bin")
  epoch += 1

