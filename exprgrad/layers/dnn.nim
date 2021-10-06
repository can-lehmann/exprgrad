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

import ../parser

proc dense*(values: Fun, inp, outp: int, has_bias: bool = true): Fun {.layer.} =
  let weights = param([inp, outp])
  result[x, y] ++= values[it, y] * weights[it, x]
  if has_bias:
    let bias = param([outp])
    result[x, y] ++= bias[x]

proc relu*(inp: Fun): Fun {.layer.} =
  result{it} ++= select(inp{it} >= 0.0, inp{it}, 0.0)

proc leaky_relu*(inp: Fun, leak: float64 = 0.01): Fun {.layer.} =
  result{it} ++= select(inp{it} >= 0.0, 1.0, @leak) * inp{it}

proc sigmoid*(inp: Fun): Fun {.layer.} =
  result{it} ++= 1.0 / (1.0 + exp(-inp{it}))

proc tanh*(inp: Fun): Fun {.layer.} =
  result{it} ++= (
    let a = exp(inp{it});
    let b = exp(-inp{it});
    (a - b) / (a + b)
  )

proc sin*(inp: Fun): Fun {.layer.} =
  result{it} ++= sin(inp{it})

proc conv2*(images, filters: Fun): Fun {.layer.} =
  result[filter, x, y, image] ++=
    images[chan, x + dx, y + dy, image] *
    filters[chan, dx, dy, filter]

proc conv2*(images: Fun, chans, w, h, filters: int): Fun =
  let filters = param([chans, w, h, filters])
  result = conv2(images, filters)

proc maxpool2*(images: Fun): Fun {.layer.} =
  result[chan, x, y, image] ++= max(
    max(
      images[chan, 2 * x, 2 * y, image],
      images[chan, 2 * x + 1, 2 * y, image]
    ),
    max(
      images[chan, 2 * x, 2 * y + 1, image],
      images[chan, 2 * x + 1, 2 * y + 1, image]
    )
  )

proc avgpool2*(images: Fun): Fun {.layer.} =
  result[chan, x, y, image] ++= (
    images[chan, 2 * x, 2 * y, image] +
    images[chan, 2 * x + 1, 2 * y, image] +
    images[chan, 2 * x, 2 * y + 1, image] +
    images[chan, 2 * x + 1, 2 * y + 1, image]
  ) / 4.0

proc upsample2*(images: Fun): Fun {.layer.} =
  # TODO
  result[chan, x, y, image] ++= images[chan, x / 2, y / 2, image]
  result.with_shape([
    images.shape[0],
    images.shape[1] * 2,
    images.shape[2] * 2,
    images.shape[3]
  ])

proc softmax*(inp: Fun): Fun {.layer.} =
  var sums: Fun
  sums[y] ++= exp(inp[x, y])
  sums.name = "softmax.sums"
  result[x, y] ++= exp(inp[x, y]) / sums[y]

proc dropout*(inp: Fun, prob: float64): Fun {.layer.} =
  let rand = rand(inp, 0.0..1.0)
  rand.name = "dropout.rand"
  result{it} ++= select(@prob <= rand{it}, inp{it} / (1.0 - @prob), 0.0)
  result.copy_shape(inp)
