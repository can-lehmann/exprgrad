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

# Store files in the PPM binary image format

import faststreams, ../tensors

proc write_ppm*(stream: var WriteStream, image: Tensor[uint8]) =
  assert image.shape.len == 3
  case image.shape[0]:
    of 1: stream.write("P5 ")
    of 3: stream.write("P6 ")
    else: raise new_exception(ValueError, "Invalid channel count")
  stream.write($image.shape[1])
  stream.write(' ')
  stream.write($image.shape[2])
  stream.write(" 255\n")
  for it in 0..<image.len:
    stream.write(image[it])

proc save_ppm*(image: Tensor[uint8], path: string) =
  var stream = open_write_stream(path)
  defer: stream.close()
  stream.write_ppm(image)
