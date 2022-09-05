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

proc fill_buffer(stream: var ReadStream) =
  stream.cur = 0
  if stream.file.is_nil:
    zero_mem(stream.buffer[0].addr, stream.buffer.len)
  else:
    stream.left = stream.file.read_buffer(
      stream.buffer[0].addr, stream.buffer.len
    )

proc open_read_stream*(path: string, buffer_size: int = 2 ^ 14): ReadStream =
  assert buffer_size > 0
  if not file_exists(path):
    raise new_exception(IoError, "Unable to open file " & path)
  result.file = open(path, fmRead)
  result.buffer = new_string(buffer_size)
  result.fill_buffer()

proc init_read_stream*(str: string): ReadStream =
  result.buffer = str
  result.left = str.len

proc close*(stream: var ReadStream) =
  if not stream.file.is_nil:
    stream.file.close()
    stream.file = nil

proc seek*(stream: var ReadStream, pos: int64) =
  if stream.file.is_nil:
    stream.cur = int(pos)
    stream.left = stream.buffer.len - int(pos)
  else:
    stream.file.set_file_pos(pos)
    stream.fill_buffer()

{.push inline.}
proc peek_char*(stream: ReadStream): char =
  stream.buffer[stream.cur]

proc read_char*(stream: var ReadStream): char =
  result = stream.buffer[stream.cur]
  stream.cur += 1
  stream.left -= 1
  if stream.cur >= stream.buffer.len:
    stream.fill_buffer()

proc peek_uint8*(stream: ReadStream): uint8 = uint8(stream.peek_char())
proc read_uint8*(stream: var ReadStream): uint8 = uint8(stream.read_char())

proc read_byte*(stream: var ReadStream): byte = byte(stream.read_char())
proc peek_byte*(stream: var ReadStream): byte = byte(stream.peek_char())

proc take_char*(stream: var ReadStream, chr: char): bool =
  result = stream.peek_char() == chr
  if result:
    discard stream.read_char()

proc at_end*(stream: ReadStream): bool = stream.left <= 0

proc skip*(stream: var ReadStream, count: int) =
  for it in 0..<count:
    discard stream.read_char()

proc skip*(stream: var ReadStream, chars: set[char]) =
  while stream.peek_char() in chars:
    discard stream.read_char()

proc skip_whitespace*(stream: var ReadStream) =
  stream.skip({' ', '\n', '\t', '\r'})

proc skip_until*(stream: var ReadStream, stop: set[char]) =
  while not stream.at_end and stream.peek_char() notin stop:
    discard stream.read_char()

proc read_until*(stream: var ReadStream, stop: set[char]): string =
  while not stream.at_end and stream.peek_char() notin stop:
    result.add(stream.read_char())
{.pop.}

type WriteStream* = object
  file: File
  buffer: string
  cur: int

proc open_write_stream*(path: string, buffer_size: int = 2 ^ 14): WriteStream =
  assert buffer_size > 0
  result.file = open(path, fmWrite)
  result.buffer = new_string(buffer_size)

proc write_all(stream: var WriteStream) =
  let written = stream.file.write_buffer(stream.buffer[0].addr, stream.cur)
  if written != stream.cur:
    raise new_exception(ValueError, "Failed to write buffer")
  stream.cur = 0

{.push inline.}
proc write*(stream: var WriteStream, x: char) =
  stream.buffer[stream.cur] = x
  stream.cur += 1
  if stream.cur >= stream.buffer.len:
    if stream.file.is_nil:
      stream.buffer &= new_string(stream.buffer.len)
    else:
      stream.write_all()

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
  if not stream.file.is_nil:
    stream.write_all()
    stream.file.flush_file()

proc close*(stream: var WriteStream) =
  if not stream.file.is_nil:
    stream.write_all()
    stream.file.flush_file()
    stream.file.close()
    stream.file = nil
