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

proc dense*(values: Fun, inp, outp: int, has_bias: bool = true): Fun {.layer.} =
  iters x, y, it:
    let weights = param([inp, outp])
    result[y, x] ++= values[y, it] * weights[it, x]
    if has_bias:
      let bias = param([outp])
      result[y, x] ++= bias[x]

proc relu*(inp: Fun): Fun {.layer.} =
  iters it: result{it} ++= select(inp{it} >= 0.0, inp{it}, 0.0)

proc leaky_relu*(inp: Fun, leak: float64 = 0.01): Fun {.layer.} =
  iters it: result{it} ++= select(inp{it} >= 0.0, 1.0, leak) * inp{it}

proc sigmoid*(inp: Fun): Fun {.layer.} =
  iters it: result{it} ++= 1.0 / (1.0 + exp(-inp{it}))

proc tanh*(inp: Fun): Fun {.layer.} =
  iters it: result{it} ++= (
    let a = exp(inp{it});
    let b = exp(-inp{it});
    (a - b) / (a + b)
  )

proc sin*(inp: Fun): Fun {.layer.} =
  iters it: result{it} ++= sin(inp{it})

proc conv2*(images, filters: Fun): Fun {.layer.} =
  iters image, y, x, filter, dx, dy, chan:
    result[image, y, x, filter] ++=
      images[image, y + dy, x + dx, chan] *
      filters[filter, dy, dx, chan]

proc conv2*(images: Fun, chans, w, h, filters: int): Fun =
  let filters = param([filters, h, w, chans])
  result = conv2(images, filters)

proc max(x, y, z, w: Scalar): Scalar =
  result = max(max(x, y), max(z, w))

proc maxpool2*(images: Fun): Fun {.layer.} =
  iters image, y, x, chan:
    result[image, y, x, chan] ++= max(
      images[image, y * 2, x * 2, chan],
      images[image, y * 2 + 1, x * 2, chan],
      images[image, y * 2, x * 2 + 1, chan],
      images[image, y * 2 + 1, x * 2 + 1, chan]
    ) | custom_grad(
      grad(images)[image, y, x, chan] ++= select(
        images[image, y, x, chan] == result[image, y / 2, x / 2, chan],
        grad(result)[image, y / 2, x / 2, chan],
        0.0
      )
    )
    result.lock()

proc avgpool2*(images: Fun): Fun {.layer.} =
  iters image, y, x, chan:
    result[image, y, x, chan] ++= (
      images[image, y * 2, x * 2, chan] +
      images[image, y * 2 + 1, x * 2, chan] +
      images[image, y * 2, x * 2 + 1, chan] +
      images[image, y * 2 + 1, x * 2 + 1, chan]
    ) / 4.0

proc upsample2*(images: Fun): Fun {.layer.} =
  # TODO
  iters image, y, x, chan:
    result[image, y, x, chan] ++= images[image, x / 2, y / 2, chan]
    result.with_shape([
      images.shape[0],
      images.shape[1] * 2,
      images.shape[2] * 2,
      images.shape[3]
    ])

proc softmax*(inp: Fun): Fun {.layer.} =
  iters y, x:
    var sums: Fun
    sums[y] ++= exp(inp[y, x])
    sums.name = "softmax.sums"
    result[y, x] ++= exp(inp[y, x]) / sums[y]

proc dropout*(inp: Fun, prob: float64): Fun {.layer.} =
  let rand = rand(inp, 0.0..1.0)
  rand.name = "dropout.rand"
  iters it:
    result{it} ++= select(prob <= rand{it}, inp{it} / (1.0 - prob), 0.0)
  result.copy_shape(inp)
