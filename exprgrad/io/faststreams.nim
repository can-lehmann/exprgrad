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

# Buffered streams for reading large datasets from disk

import std/[math, os]

type ReadStream* = object
  file: File
  buffer: string
  cur: int
  left: int
  pos: int64

proc fillBuffer(stream: var ReadStream) =
  stream.cur = 0
  when nimvm:
    assert stream.file.isNil
    for it in 0..<stream.buffer.len:
      stream.buffer[it] = '\0'
  else:
    if stream.file.isNil:
      zeroMem(stream.buffer[0].addr, stream.buffer.len)
    else:
      stream.left = stream.file.readBuffer(
        stream.buffer[0].addr, stream.buffer.len
      )

proc initReadStream*(str: string): ReadStream =
  result.buffer = str
  result.left = str.len

proc openReadStream*(path: string, bufferSize: int = 2 ^ 14): ReadStream =
  when nimvm:
    result = initReadStream(readFile(path))
  else:
    assert bufferSize > 0
    if not fileExists(path):
      raise newException(IOError, "Unable to open file " & path)
    result.file = open(path, fmRead)
    result.buffer = newString(bufferSize)
    result.fillBuffer()

proc close*(stream: var ReadStream) =
  when nimvm:
    discard
  else:
    if not stream.file.isNil:
      stream.file.close()
      stream.file = nil

proc seek*(stream: var ReadStream, pos: int64) =
  stream.pos = pos
  when nimvm:
    stream.cur = int(pos)
    stream.left = stream.buffer.len - int(pos)
  else:
    if stream.file.isNil:
      stream.cur = int(pos)
      stream.left = stream.buffer.len - int(pos)
    else:
      stream.file.setFilePos(pos)
      stream.fillBuffer()

{.push inline.}
proc position*(stream: var ReadStream): int64 = stream.pos

proc peekChar*(stream: ReadStream): char =
  stream.buffer[stream.cur]

proc readChar*(stream: var ReadStream): char =
  result = stream.buffer[stream.cur]
  stream.cur += 1
  stream.left -= 1
  stream.pos += 1
  if stream.cur >= stream.buffer.len:
    stream.fillBuffer()

proc peekUint8*(stream: ReadStream): uint8 = uint8(stream.peekChar())
proc readUint8*(stream: var ReadStream): uint8 = uint8(stream.readChar())

proc readByte*(stream: var ReadStream): byte = byte(stream.readChar())
proc peekByte*(stream: var ReadStream): byte = byte(stream.peekChar())

proc takeChar*(stream: var ReadStream, chr: char): bool =
  result = stream.peekChar() == chr
  if result:
    discard stream.readChar()

proc atEnd*(stream: ReadStream): bool = stream.left <= 0

proc skip*(stream: var ReadStream, count: int) =
  for it in 0..<count:
    discard stream.readChar()

proc skip*(stream: var ReadStream, chars: set[char]) =
  while stream.peekChar() in chars:
    discard stream.readChar()

proc skipWhitespace*(stream: var ReadStream) =
  stream.skip({' ', '\n', '\t', '\r'})

proc skipUntil*(stream: var ReadStream, stop: set[char]) =
  while not stream.atEnd and stream.peekChar() notin stop:
    discard stream.readChar()

proc readUntil*(stream: var ReadStream, stop: set[char]): string =
  while not stream.atEnd and stream.peekChar() notin stop:
    result.add(stream.readChar())
{.pop.}

type WriteStream* = object
  file: File
  buffer: string
  cur: int

proc openWriteStream*(path: string, bufferSize: int = 2 ^ 14): WriteStream =
  assert bufferSize > 0
  result.file = open(path, fmWrite)
  result.buffer = newString(bufferSize)

proc writeAll(stream: var WriteStream) =
  let written = stream.file.writeBuffer(stream.buffer[0].addr, stream.cur)
  if written != stream.cur:
    raise newException(ValueError, "Failed to write buffer")
  stream.cur = 0

{.push inline.}
proc write*(stream: var WriteStream, x: char) =
  stream.buffer[stream.cur] = x
  stream.cur += 1
  if stream.cur >= stream.buffer.len:
    if stream.file.isNil:
      stream.buffer &= newString(stream.buffer.len)
    else:
      stream.writeAll()

proc write*(stream: var WriteStream, x: uint8) =
  stream.write(cast[char](x))

proc write*(stream: var WriteStream, str: string) =
  # TODO: Optimize
  for chr in str:
    stream.write(chr)

proc write*(stream: var WriteStream, data: openArray[uint8]) =
  for value in data:
    stream.write(value)
{.pop.}

proc flush*(stream: var WriteStream) =
  if not stream.file.isNil:
    stream.writeAll()
    stream.file.flushFile()

proc close*(stream: var WriteStream) =
  if not stream.file.isNil:
    stream.writeAll()
    stream.file.flushFile()
    stream.file.close()
    stream.file = nil
