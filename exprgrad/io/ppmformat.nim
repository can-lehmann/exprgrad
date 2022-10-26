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

proc writePpm*(stream: var WriteStream, image: Tensor[uint8], useAscii: bool = false) =
  assert image.shape.len == 3
  case image.shape[2]:
    of 1: stream.write(if useAscii: "P2 " else: "P5 ")
    of 3: stream.write(if useAscii: "P3 " else: "P6 ")
    else: raise newException(ValueError, "Invalid channel count")
  stream.write($image.shape[1])
  stream.write(' ')
  stream.write($image.shape[0])
  stream.write(" 255")
  if useAscii:
    for it in 0..<image.len:
      stream.write(' ')
      stream.write($image[it])
  else:
    stream.write('\n')
    for it in 0..<image.len:
      stream.write(image[it])

proc savePpm*(image: Tensor[uint8], path: string, useAscii: bool = false) =
  var stream = openWriteStream(path)
  defer: stream.close()
  stream.writePpm(image, useAscii=useAscii)

const WHITESPACE = {' ', '\n', '\t', '\r'}

proc readWord(stream: var ReadStream): string =
  stream.skip(WHITESPACE)
  while stream.peekChar() notin WHITESPACE + {'\0'}:
    result.add(stream.readChar())

proc readInt(stream: var ReadStream): int {.inline.} =
  stream.skip(WHITESPACE)
  while stream.peekChar() in '0'..'9':
    result *= 10
    result += ord(stream.readChar()) - ord('0')

proc readPpm*(stream: var ReadStream): Tensor[uint8] =
  let
    format = stream.readWord()
    width = stream.readInt()
    height = stream.readInt()
    max = stream.readInt()
    (isAscii, chans) = case format:
      of "P2": (true, 1)
      of "P3": (true, 3)
      of "P5": (false, 1)
      of "P6": (false, 3)
      else: raise newException(IoError, "Unknown format \"" & format & "\"")
  if max > int(high(uint8)):
    raise newException(IoError, "Maximum value must not be larger than 255")
  result = Tensor[uint8].new([height, width, chans])
  if isAscii:
    for it in 0..<result.len:
      result.data[it] = uint8(stream.readInt())
  else:
    if stream.readChar() notin WHITESPACE:
      raise newException(IoError, "Header must end with ASCII whitespace character")
    for it in 0..<result.len:
      result.data[it] = stream.readUint8()

proc loadPpm*(path: string): Tensor[uint8] =
  var stream = openReadStream(path)
  defer: stream.close()
  result = readPpm(stream)
