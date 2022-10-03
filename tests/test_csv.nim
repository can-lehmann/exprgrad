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

# Unit-tests for exprgrad's csv module

import std/sequtils
import exprgrad/io/[csvformat, faststreams]
import ../tools/test_framework

test "base":
  type Row = object
    name: string
    number*: int
  
  let data = "name;number\nTest;1\n\n\n\"Hello, world\";123\n\";\";\"42\""
  var stream = initReadStream(data)
  check toSeq(iterCsv[Row](stream)) == @[
    Row(name: "Test", number: 1),
    Row(name: "Hello, world", number: 123),
    Row(name: ";", number: 42)
  ]

test "named_columns":
  type Row = object
    name* {.csvColumn: "a".}: string
    number {.csvColumn: "b".}: int
  
  let data = "a;b\nTest;1\n\n\n\"Hello, world\";123\n\";\";\"42\""
  var stream = initReadStream(data)
  check toSeq(iterCsv[Row](stream)) == @[
    Row(name: "Test", number: 1),
    Row(name: "Hello, world", number: 123),
    Row(name: ";", number: 42)
  ]

test "ignore_columns":
  type Row {.csvIgnore: ["a"].} = object
    number {.csvColumn: "b".}: int
  
  let data = "a;b\nTest;1\n\n\n\"Hello, world\";123\n\";\";\"42\""
  var stream = initReadStream(data)
  check toSeq(iterCsv[Row](stream)) == @[
    Row(number: 1),
    Row(number: 123),
    Row(number: 42)
  ]

test "custom_parser":
  proc parseName(str: string, dest: var int) =
    dest = len(str)
  
  type Row = object
    nameLen {.csvParser: parseName, csvColumn: "name".}: int
    number*: int
  
  let data = "name;number\nTest;1\n\n\n\"Hello, world\";123\n\";\";\"42\"\n"
  var stream = initReadStream(data)
  check toSeq(iterCsv[Row](stream)) == @[
    Row(nameLen: len("Test"), number: 1),
    Row(nameLen: len("Hello, world"), number: 123),
    Row(nameLen: len(";"), number: 42)
  ]
