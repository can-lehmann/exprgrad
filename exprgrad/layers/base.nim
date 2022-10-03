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

# Layers used across many different disciplines

import ../parser, ../dsl

proc `+`*(a, b: Fun): Fun {.layer.} = result{it} ++= a{it} + b{it} | it
proc `-`*(a, b: Fun): Fun {.layer.} = result{it} ++= a{it} - b{it} | it
proc min*(a, b: Fun): Fun {.layer.} = result{it} ++= min(a{it}, b{it}) | it
proc max*(a, b: Fun): Fun {.layer.} = result{it} ++= max(a{it}, b{it}) | it

proc `*`*(a: Fun, factor: float64): Fun {.layer.} = result{it} ++= a{it} * factor | it
proc `/`*(a: Fun, factor: float64): Fun {.layer.} = result{it} ++= a{it} / factor | it

proc matmul*(a, b: Fun): Fun {.layer.} =
  result[y, x] ++= a[y, it] * b[it, x] | (y, x, it)

proc `*`*(a, b: Fun): Fun = matmul(a, b)

proc transpose*(mat: Fun): Fun {.layer.} =
  result[y, x] ++= mat[x, y] | (y, x)

# Optimizers

proc gradientDescent*(param: var Fun, grad: Fun, rate: float64 = 0.01) =
  param{it} ++= -grad{it} * rate | it

proc adam*(param: var Fun, grad: Fun,
           eta: float64 = 0.01,
           beta1: float64 = 0.9,
           beta2: float64 = 0.999,
           eps: float64 = 1e-8) =
  ## Diederik P. Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization", 2014
  var (m, v) = (cache(param, "adam.m"), cache(param, "adam.v"))
  m{it} ++= m{it} * (beta1 - 1.0) + (1.0 - beta1) * grad{it} | it
  v{it} ++= v{it} * (beta2 - 1.0) + (1.0 - beta2) * sq(grad{it}) | it
  param{it} ++= (
    let mHat = m{it} / (1.0 - pow(beta1, toScalar(epoch())));
    let vHat = v{it} / (1.0 - pow(beta2, toScalar(epoch())));
    -eta * mHat / (sqrt(vHat) + eps)
  ) | it

# Losses

proc mse*(a, b: Fun): Fun {.layer.} =
  result[0] ++= sq(a{it} - b{it}) / toScalar(a.shape[0]) | it

proc binaryCrossEntropy*(pred, labels: Fun): Fun {.layer.} =
  result[0] ++= -(
    labels{it} * ln(pred{it}) +
    (1.0 - labels{it}) * ln(1.0 - pred{it})
  ) / toScalar(pred.shape[0]) | it

proc crossEntropy*(pred, labels: Fun): Fun {.layer.} =
  result[0] ++= -(labels{it} * ln(pred{it})) / toScalar(pred.shape[0]) | it
