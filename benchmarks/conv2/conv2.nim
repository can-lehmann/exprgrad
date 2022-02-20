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

import std/[times, monotimes, math, random]
import exprgrad

randomize()

template loop(iter: untyped, start, stop: int, step: static[int], body: untyped) =
  when defined(while_loops):
    var iter = int(start)
    let stop_copy = int(stop)
    while iter < stop_copy:
      body
      iter += step
  elif defined(c_loops):
    {.emit: ["int start_copy = ", start, ";"].}
    {.emit: ["int stop_copy = ", stop, ";"].}
    {.emit: ["int step_copy = ", step, ";"].}
    var iter = 0
    {.emit: ["for (", iter, " = start_copy; ", iter, " < stop_copy; ", iter, " += step_copy) {"].}
    body
    {.emit: "}".}
  else:
    when step == 1:
      for iter in start..<stop:
        body
    else:
      for iter in countup(start, stop - 1, step):
        body

proc conv2_naive[T](image, filters: Tensor[T]): Tensor[T] =
  result = new_tensor[T]([
    image.shape[0] - filters.shape[1] + 1,
    image.shape[1] - filters.shape[2] + 1,
    filters.shape[0]
  ])
  loop(y, 0, result.shape[0], 1):
    loop(filter, 0, filters.shape[0], 1):
      loop(dy, 0, filters.shape[1], 1):
        loop(x, 0, result.shape[1], 1):
          loop(dx, 0, filters.shape[2], 1):
            loop(chan, 0, filters.shape[3], 1):
              result[y, x, filter] += image[y + dy, x + dx, chan] * filters[filter, dy, dx, chan]

proc conv2_naive_specialized[T](image, filters: Tensor[T], fc, fh, fw, chans: static[int]): Tensor[T] =
  result = new_tensor[T]([
    image.shape[0] - fh + 1,
    image.shape[1] - fw + 1,
    fc
  ])
  let
    iw = image.shape[1]
    rw = result.shape[1]
  loop(y, 0, result.shape[0], 1):
    loop(dy, 0, fh, 1):
      loop(x, 0, rw, 1):
        loop(dx, 0, fw, 1):
          loop(filter, 0, fc, 1):
            loop(chan, 0, chans, 1):
              result.data[y * fc * rw + x * fc + filter] +=
                image.data[(y + dy) * chans * iw + (x + dx) * chans + chan] *
                filters.data[filter * chans * fw * fh + dy * chans * fw + dx * chans + chan]

proc conv2_naive_specialized[T](image, filters: Tensor[T]): Tensor[T] =
  if filters.shape == @[16, 3, 3, 8]:
    return conv2_naive_specialized(image, filters, 16, 3, 3, 8)
  elif filters.shape == @[1, 3, 3, 1]:
    return conv2_naive_specialized(image, filters, 1, 3, 3, 1)
  elif filters.shape == @[8, 3, 3, 8]:
    return conv2_naive_specialized(image, filters, 8, 3, 3, 8)
  else:
    raise new_exception(ValueError, "")

proc conv2_tiled_specialized[T](image, filters: Tensor[T], fc, fh, fw, chans: static[int]): Tensor[T] =
  assert filters.shape[0] == fc
  assert filters.shape[1] == fh
  assert filters.shape[2] == fw
  assert filters.shape[3] == chans
  assert image.shape[2] == chans
  
  result = new_tensor[T]([
    image.shape[0] - fh + 1,
    image.shape[1] - fw + 1,
    fc
  ])
  let
    iw = image.shape[1]
    rh = result.shape[0]
    rw = result.shape[1]
  const
    TILE_SIZE_X = 2048
    TILE_SIZE_Y = 32
  
  loop(ty, 0, rh, TILE_SIZE_Y):
    loop(tx, 0, rw, TILE_SIZE_X):
      loop(y, ty, min(ty + TILE_SIZE_Y, rh), 1):
        loop(dy, 0, fh, 1):
          loop(x, tx, min(tx + TILE_SIZE_X, rw), 1):
            loop(dx, 0, fw, 1):
              loop(filter, 0, fc, 1):
                loop(chan, 0, chans, 1):
                  result.data[y * fc * rw + x * fc + filter] +=
                    image.data[(y + dy) * chans * iw + (x + dx) * chans + chan] *
                    filters.data[filter * chans * fw * fh + dy * chans * fw + dx * chans + chan]

proc conv2_tiled_specialized[T](image, filters: Tensor[T]): Tensor[T] =
  if filters.shape == @[16, 3, 3, 8]:
    return conv2_tiled_specialized(image, filters, 16, 3, 3, 8)
  elif filters.shape == @[1, 3, 3, 1]:
    return conv2_tiled_specialized(image, filters, 1, 3, 3, 1)
  elif filters.shape == @[8, 3, 3, 8]:
    return conv2_tiled_specialized(image, filters, 8, 3, 3, 8)
  else:
    raise new_exception(ValueError, "")

proc conv2(image, filters: Fun): Fun =
  iters y, x, filter, chan, dy, dx:
    result[y, x, filter] ++=
      image[y + dy, x + dx, chan] *
      filters[filter, dy, dx, chan]

let model = compile[float64](
  conv2(input("image"), input("filters")).target("conv2"),
  conv2(input("image_8", [-1, -1, 8]), input("filters_8_3_3_8", [8, 3, 3, 8])).target("conv2_8_3_3_8"),
  conv2(input("image_1", [-1, -1, 1]), input("filters_1_3_3_1", [1, 3, 3, 1])).target("conv2_1_3_3_1")
)

proc conv2_exprgrad(image, filters: Tensor[float64]): Tensor[float64] =
  result = model.call("conv2", {"image": image, "filters": filters})

proc conv2_exprgrad_specialized(image, filters: Tensor[float64]): Tensor[float64] =
  if filters.shape == @[1, 3, 3, 1]:
    result = model.call("conv2_1_3_3_1", {"image_1": image, "filters_1_3_3_1": filters})
  elif filters.shape == @[8, 3, 3, 8]:
    result = model.call("conv2_8_3_3_8", {"image_8": image, "filters_8_3_3_8": filters})
  else:
    raise new_exception(ValueError, "")


proc conv2_c_specialized(image, filters: Tensor[float64], fc, fh, fw, chans: static[cint]): Tensor[float64] =
  result = new_tensor[float64]([
    image.shape[0] - int(fh) + 1,
    image.shape[1] - int(fw) + 1,
    int(fc)
  ])
  {.emit: ["double* restrict image_data = ", image.data_ptr, ";"].}
  {.emit: ["double* restrict filters_data = ", filters.data_ptr, ";"].}
  {.emit: ["double* restrict result_data = ", result.data_ptr, ";"].}
  {.emit: ["int iw = ", image.shape[1], ", ih = ", image.shape[0], ";"].}
  {.emit: ["int rw = ", result.shape[1], ", rh = ", result.shape[0], ";"].}
  {.emit: ["""
    for (int y = 0; y < rh; y++) {
      for (int dy = 0; dy < """, fh, """; dy++) {
        for (int x = 0; x < rw; x++) {
          for (int dx = 0; dx < """, fw, """; dx++) {
            for (int filter = 0; filter < """, fc, """; filter++) {
              for (int chan = 0; chan < """, chans, """; chan++) {
                result_data[y * """, fc, """ * rw + x * """, fc, """ + filter] +=
                  image_data[(y + dy) * """, chans, """ * iw + (x + dx) * """, chans, """ + chan] *
                  filters_data[filter * """, chans, """ * """, fw, """ * """, fh, """ + dy * """, chans, """ * """, fw, """ + dx * """, chans, """ + chan];
              }
            }
          }  
        }
      }
    }
  """].}

proc conv2_c_specialized(image, filters: Tensor[float64]): Tensor[float64] =
  if filters.shape == @[16, 3, 3, 8]:
    return conv2_c_specialized(image, filters, 16, 3, 3, 8)
  elif filters.shape == @[1, 3, 3, 1]:
    return conv2_c_specialized(image, filters, 1, 3, 3, 1)
  elif filters.shape == @[8, 3, 3, 8]:
    return conv2_c_specialized(image, filters, 8, 3, 3, 8)
  else:
    raise new_exception(ValueError, "")

proc conv2_c_specialized_reorder(image, filters: Tensor[float64], fc, fh, fw, chans: static[cint]): Tensor[float64] =
  result = new_tensor[float64]([
    image.shape[0] - int(fh) + 1,
    image.shape[1] - int(fw) + 1,
    int(fc)
  ])
  {.emit: ["double* restrict image_data = ", image.data_ptr, ";"].}
  {.emit: ["double* restrict filters_data = ", filters.data_ptr, ";"].}
  {.emit: ["double* restrict result_data = ", result.data_ptr, ";"].}
  {.emit: ["int iw = ", image.shape[1], ", ih = ", image.shape[0], ";"].}
  {.emit: ["int rw = ", result.shape[1], ", rh = ", result.shape[0], ";"].}
  {.emit: ["""
    for (int y = 0; y < rh; y++) {
      for (int x = 0; x < rw; x++) {
        double* restrict region_data = &image_data[y * """, chans, """ * iw + x * """, chans, """];
        for (int filter = 0; filter < """, fc, """; filter++) {
          double acc = 0;
          double* restrict filter_data = &filters_data[filter * """, chans, """ * """, fw, """ * """, fh, """];
          for (int dy = 0; dy < """, fh, """; dy++) {
            for (int dx = 0; dx < """, fw, """; dx++) {
              for (int chan = 0; chan < """, chans, """; chan++) {
                acc +=
                  region_data[dy * """, chans, """ * iw + dx * """, chans, """ + chan] *
                  filter_data[dy * """, chans, """ * """, fw, """ + dx * """, chans, """ + chan];
              }
            }
          }
          result_data[y * """, fc, """ * rw + x * """, fc, """ + filter] = acc;
        }
      }
    }
  """].}

proc conv2_c_specialized_reorder(image, filters: Tensor[float64]): Tensor[float64] =
  if filters.shape == @[16, 3, 3, 8]:
    return conv2_c_specialized_reorder(image, filters, 16, 3, 3, 8)
  elif filters.shape == @[1, 3, 3, 1]:
    return conv2_c_specialized_reorder(image, filters, 1, 3, 3, 1)
  elif filters.shape == @[8, 3, 3, 8]:
    return conv2_c_specialized_reorder(image, filters, 8, 3, 3, 8)
  else:
    raise new_exception(ValueError, "")


proc conv2_c_tiled_specialized(image, filters: Tensor[float64], fc, fh, fw, chans: static[cint]): Tensor[float64] =
  result = new_tensor[float64]([
    image.shape[0] - int(fh) + 1,
    image.shape[1] - int(fw) + 1,
    int(fc)
  ])
  const
    TILE_SIZE_X = cint(2048)
    TILE_SIZE_Y = cint(32)
  {.emit: ["double* restrict image_data = ", image.data_ptr, ";"].}
  {.emit: ["double* restrict filters_data = ", filters.data_ptr, ";"].}
  {.emit: ["double* restrict result_data = ", result.data_ptr, ";"].}
  {.emit: ["int iw = ", image.shape[1], ", ih = ", image.shape[0], ";"].}
  {.emit: ["int rw = ", result.shape[1], ", rh = ", result.shape[0], ";"].}
  {.emit: ["""
    for (int ty = 0; ty < rh; ty += """, TILE_SIZE_Y, """) {
      for (int tx = 0; tx < rw; tx += """, TILE_SIZE_X, """) {
        for (int y = ty; y < rh && y < ty + """, TILE_SIZE_Y, """; y++) {
          for (int dy = 0; dy < """, fh, """; dy++) {
            for (int x = tx; x < rw && x < tx + """, TILE_SIZE_X, """; x++) {
              for (int dx = 0; dx < """, fw, """; dx++) {
                for (int filter = 0; filter < """, fc, """; filter++) {
                  for (int chan = 0; chan < """, chans, """; chan++) {
                    result_data[y * """, fc, """ * rw + x * """, fc, """ + filter] +=
                      image_data[(y + dy) * """, chans, """ * iw + (x + dx) * """, chans, """ + chan] *
                      filters_data[filter * """, chans, """ * """, fw, """ * """, fh, """ + dy * """, chans, """ * """, fw, """ + dx * """, chans, """ + chan];
                  }
                }
              }  
            }
          }
        }
      }
    }
  """].}

proc conv2_c_tiled_specialized(image, filters: Tensor[float64]): Tensor[float64] =
  if filters.shape == @[16, 3, 3, 8]:
    return conv2_c_tiled_specialized(image, filters, 16, 3, 3, 8)
  elif filters.shape == @[1, 3, 3, 1]:
    return conv2_c_tiled_specialized(image, filters, 1, 3, 3, 1)
  elif filters.shape == @[8, 3, 3, 8]:
    return conv2_c_tiled_specialized(image, filters, 8, 3, 3, 8)
  else:
    raise new_exception(ValueError, "")

proc conv2_c_tiled_specialized_reorder(image, filters: Tensor[float64], fc, fh, fw, chans: static[cint]): Tensor[float64] =
  result = new_tensor[float64]([
    image.shape[0] - int(fh) + 1,
    image.shape[1] - int(fw) + 1,
    int(fc)
  ])
  const
    TILE_SIZE_X = cint(2048)
    TILE_SIZE_Y = cint(32)
  {.emit: ["double* restrict image_data = ", image.data_ptr, ";"].}
  {.emit: ["double* restrict filters_data = ", filters.data_ptr, ";"].}
  {.emit: ["double* restrict result_data = ", result.data_ptr, ";"].}
  {.emit: ["int iw = ", image.shape[1], ", ih = ", image.shape[0], ";"].}
  {.emit: ["int rw = ", result.shape[1], ", rh = ", result.shape[0], ";"].}
  {.emit: ["""
    for (int ty = 0; ty < rh; ty += """, TILE_SIZE_Y, """) {
      for (int tx = 0; tx < rw; tx += """, TILE_SIZE_X, """) {
        for (int y = ty; y < rh && y < ty + """, TILE_SIZE_Y, """; y++) {
          for (int x = tx; x < rw && x < tx + """, TILE_SIZE_X, """; x++) {
            for (int filter = 0; filter < """, fc, """; filter++) {
              double acc = 0;
              for (int dy = 0; dy < """, fh, """; dy++) {
                for (int dx = 0; dx < """, fw, """; dx++) {
                  for (int chan = 0; chan < """, chans, """; chan++) {
                    acc +=
                      image_data[(y + dy) * """, chans, """ * iw + (x + dx) * """, chans, """ + chan] *
                      filters_data[filter * """, chans, """ * """, fw, """ * """, fh, """ + dy * """, chans, """ * """, fw, """ + dx * """, chans, """ + chan];
                  }
                }
              }
              result_data[y * """, fc, """ * rw + x * """, fc, """ + filter] = acc;
            }
          }
        }
      }
    }
  """].}

proc conv2_c_tiled_specialized_reorder(image, filters: Tensor[float64]): Tensor[float64] =
  if filters.shape == @[16, 3, 3, 8]:
    return conv2_c_tiled_specialized_reorder(image, filters, 16, 3, 3, 8)
  elif filters.shape == @[1, 3, 3, 1]:
    return conv2_c_tiled_specialized_reorder(image, filters, 1, 3, 3, 1)
  elif filters.shape == @[8, 3, 3, 8]:
    return conv2_c_tiled_specialized_reorder(image, filters, 8, 3, 3, 8)
  else:
    raise new_exception(ValueError, "")


proc measure[T](conv2: proc (images, filters: Tensor[T]): Tensor[T],
                image_shape: openArray[int] = [240 * 4, 320 * 4, 8],
                filters_shape: openArray[int] = [8, 3, 3, 8],
                sample_count: int = 4,
                fail_threshold: float64 = 0.1): Duration =
  for sample in 0..<sample_count:
    let
      image = new_rand_tensor[T](image_shape, T(0)..T(1))
      filters = new_rand_tensor[T](filters_shape, T(-2)..T(2))
    let
      start = get_mono_time()
      res = conv2(image, filters)
      stop = get_mono_time()
      expected = conv2_naive(image, filters)
    if res.shape != expected.shape:
      raise new_exception(ValueError, "Shape mismatch")
    let
      err = squares(res - expected).sum()
      mean_err = float64(err) / res.shape.prod().float64
    if mean_err > fail_threshold:
      raise new_exception(ValueError, $mean_err)
    result += stop - start
  result = init_duration(microseconds = result.in_microseconds().int div sample_count)

echo "conv2_naive: ", measure[float64](conv2_naive)
echo "conv2_naive_specialized: ", measure[float64](conv2_naive_specialized)
echo "conv2_tiled_specialized: ", measure[float64](conv2_tiled_specialized)
echo "conv2_exprgrad: ", measure[float64](conv2_exprgrad)
echo "conv2_exprgrad_specialized: ", measure[float64](conv2_exprgrad_specialized)
echo "conv2_c_specialized: ", measure[float64](conv2_c_specialized)
echo "conv2_c_tiled_specialized: ", measure[float64](conv2_c_tiled_specialized)
echo "conv2_c_tiled_specialized_reorder: ", measure[float64](conv2_c_tiled_specialized_reorder)
echo "conv2_c_specialized_reorder: ", measure[float64](conv2_c_specialized_reorder)

