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

# Layers for building neural networks

import ../parser, ../dsl

proc dense*(values: Fun, inp, outp: int, hasBias: bool = true): Fun {.layer.} =
  let weights = param([inp, outp])
  result[y, x] ++= values[y, it] * weights[it, x] | (x, y, it)
  if hasBias:
    let bias = param([outp])
    result[y, x] ++= bias[x] | (y, x)

proc relu*(inp: Fun): Fun {.layer.} =
  result{it} ++= select(inp{it} >= 0.0, inp{it}, 0.0) | it

proc leakyRelu*(inp: Fun, leak: float64 = 0.01): Fun {.layer.} =
  result{it} ++= select(inp{it} >= 0.0, 1.0, leak) * inp{it} | it

proc sigmoid*(inp: Fun): Fun {.layer.} =
  result{it} ++= 1.0 / (1.0 + exp(-inp{it})) | it

proc tanh*(inp: Fun): Fun {.layer.} =
  result{it} ++= (
    let a = exp(inp{it});
    let b = exp(-inp{it});
    (a - b) / (a + b)
  ) | it

proc sin*(inp: Fun): Fun {.layer.} =
  result{it} ++= sin(inp{it}) | it

proc conv2*(images, filters: Fun): Fun {.layer.} =
  result[image, y, x, filter] ++= (
    images[image, y + dy, x + dx, chan] *
    filters[filter, dy, dx, chan]
  ) | (image, y, x, filter, dx, dy, chan)

proc conv2*(images: Fun, chans, w, h, filters: int): Fun =
  let filters = param([filters, h, w, chans])
  result = conv2(images, filters)

proc max(x, y, z, w: Scalar): Scalar =
  result = max(max(x, y), max(z, w))

proc maxpool2*(images: Fun): Fun {.layer.} =
  result[image, y, x, chan] ++= max(
    images[image, y * 2, x * 2, chan],
    images[image, y * 2 + 1, x * 2, chan],
    images[image, y * 2, x * 2 + 1, chan],
    images[image, y * 2 + 1, x * 2 + 1, chan]
  ) | (image, y, x, chan) do:
    customGrad:
      grad(images)[image, y, x, chan] ++= select(
        images[image, y, x, chan] == result[image, y div 2, x div 2, chan],
        grad(result)[image, y div 2, x div 2, chan],
        0.0
      ) | (image, y, x, chan)
  result.lock()

proc avgpool2*(images: Fun): Fun {.layer.} =
  result[image, y, x, chan] ++= (
    images[image, y * 2, x * 2, chan] +
    images[image, y * 2 + 1, x * 2, chan] +
    images[image, y * 2, x * 2 + 1, chan] +
    images[image, y * 2 + 1, x * 2 + 1, chan]
  ) / 4.0 | (image, y, x, chan)

proc upsample2*(images: Fun): Fun {.layer.} =
  result[image, y, x, chan] ++= images[image, x div 2, y div 2, chan] | (image, y, x, chan)
  result.withShape([
    images.shape[0],
    images.shape[1] * 2,
    images.shape[2] * 2,
    images.shape[3]
  ])

proc softmax*(inp: Fun): Fun {.layer.} =
  var sums: Fun
  sums[y] ++= exp(inp[y, x]) | (y, x)
  sums.name = "softmax.sums"
  result[y, x] ++= exp(inp[y, x]) / sums[y] | (y, x)

proc dropout*(inp: Fun, prob: float64): Fun {.layer.} =
  let rand = rand(inp, 0.0..1.0)
  rand.name = "dropout.rand"
  result{it} ++= select(prob <= rand{it}, inp{it} / (1.0 - prob), 0.0) | it
  result.copyShape(inp)
