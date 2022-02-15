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

# Unit-tests for exprgrad's json module

import std/tables
import exprgrad/io/[jsonformat, faststreams]
import ../tools/test_framework

test "base":
  check parse_json[bool]("false") == false
  check parse_json[bool]("true") == true
  check parse_json[int]("10") == 10
  check parse_json[int]("-10") == -10
  check parse_json[int]("0") == 0
  check parse_json[float]("3.14") == 3.14
  check parse_json[string]("\"Hello, world!\"") == "Hello, world!"
  check parse_json[string]("\"Escape sequences: \\n\\t\\\"\\\\\"") == "Escape sequences: \n\t\"\\"

test "collections":
  check parse_json[seq[int]]("[]") == new_seq[int]()
  check parse_json[seq[int]]("[1, 2, 3]") == @[1, 2, 3]
  check parse_json[seq[seq[int]]]("[[1, 2, 3], [4, 5, 6]]") == @[@[1, 2, 3], @[4, 5, 6]]
  check parse_json[array[3, int]]("[1, 2, 3]") == [1, 2, 3]
  check parse_json[array[3, seq[int]]]("[[], [1], [2, 3]]") == [new_seq[int](), @[1], @[2, 3]]
  check parse_json[Table[string, float]]("{\"x\": 1, \"y\": 2.5}") == to_table({"x": 1.0, "y": 2.5})
  check parse_json[Table[string, seq[int]]]("{\"x\": [], \"y\": [1], \"z\": [2, 3]}") == to_table({
    "x": new_seq[int](), "y": @[1], "z": @[2, 3]
  })
  check parse_json[Table[string, seq[int]]]("{}") == init_table[string, seq[int]]()

test "serialize_json":
  type
    TestObjKind = enum
      TestObjArray, TestObjSeq
    
    TestObj = object
      bool_val: bool
      int_val: int 
      float_val: float
      table_val: Table[string, int]
      case kind: TestObjKind:
        of TestObjArray: array_val: array[3, int]
        of TestObjSeq: seq_val: seq[int]

test "json_node":
  check parse_json[JsonNode]("1") == new_jint(1)
  check parse_json[JsonNode]("123") == new_jint(123)
  check parse_json[JsonNode]("3.14") == new_jfloat(3.14)
  check parse_json[JsonNode]("\"Hello, world!\\n\"") == new_jstring("Hello, world!\n")
  check parse_json[JsonNode]("[1, 2, 3]") == new_jarray([new_jint(1), new_jint(2), new_jint(3)])
  check parse_json[JsonNode]("[[null, 1, 2, 2.5], true, false]") == new_jarray([
    new_jarray([new_jnull(), new_jint(1), new_jint(2), new_jfloat(2.5)]),
    new_jbool(true), new_jbool(false)
  ])
  check parse_json[JsonNode]("{\"x\": 1, \"y\": 2}") == new_jobject({
    "x": new_jint(1), "y": new_jint(2)
  })
