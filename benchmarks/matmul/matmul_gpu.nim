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

import std/[times, monotimes]
import exprgrad

template measure(name: string, body: untyped) =
  let sample_count = 16
  var sum = init_duration()
  for sample in 0..<sample_count:
    let start = get_mono_time()
    body
    let stop = get_mono_time()
    sum += stop - start
  echo name, ": ", init_duration(microseconds = sum.in_microseconds().int div sample_count)

proc measure_cpu(gpu: GpuContext, a_tensor, b_tensor: Tensor[float32]) =
  let
    a = input("a", a_tensor.shape)
    b = input("b", b_tensor.shape)
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  let model = compile[float32](c.target("c"))
  
  measure "cpu":
    discard model.call("c", {"a": a_tensor, "b": b_tensor})

proc measure_naive(gpu: GpuContext, a_tensor, b_tensor: Tensor[float32]) =
  let
    a = input("a", a_tensor.shape)
    b = input("b", b_tensor.shape)
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it)
  let model = compile[float32](c.target("c", CompileGpu), gpu=gpu)
  
  measure "naive":
    discard model.call("c", {"a": a_tensor, "b": b_tensor})

proc measure_tiled16(gpu: GpuContext, a_tensor, b_tensor: Tensor[float32]) =
  let
    a = input("a", a_tensor.shape)
    b = input("b", b_tensor.shape)
  c*[y, x] ++= a[y, it] * b[it, x] | (x, y, it) do:
    schedule:
      parallel(y)
      gpu:
        tile_size(x, 16)
        tile_size(y, 16)
        parallel(x)
        cache(a)
        cache(b)
        tile_size(it, 16)
        tile(it)
  let model = compile[float32](c.target("c", CompileGpu), gpu=gpu)
  
  measure "tiled16":
    discard model.call("c", {"a": a_tensor, "b": b_tensor})

let
  a_tensor = new_rand_tensor[float32]([2048, 2048], float32(0)..float32(1))
  b_tensor = new_rand_tensor[float32]([2048, 2048], float32(0)..float32(1))

let ctx = new_gpu_context()
measure_cpu(ctx, a_tensor, b_tensor)
measure_naive(ctx, a_tensor, b_tensor)
measure_tiled16(ctx, a_tensor, b_tensor)
