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
  check parseJson[bool]("false") == false
  check parseJson[bool]("true") == true
  check parseJson[int]("10") == 10
  check parseJson[int]("-10") == -10
  check parseJson[int]("0") == 0
  check parseJson[float]("3.14") == 3.14
  check parseJson[string]("\"Hello, world!\"") == "Hello, world!"
  check parseJson[string]("\"Escape sequences: \\n\\t\\\"\\\\\"") == "Escape sequences: \n\t\"\\"

test "collections":
  check parseJson[seq[int]]("[]") == newSeq[int]()
  check parseJson[seq[int]]("[1, 2, 3]") == @[1, 2, 3]
  check parseJson[seq[seq[int]]]("[[1, 2, 3], [4, 5, 6]]") == @[@[1, 2, 3], @[4, 5, 6]]
  check parseJson[array[3, int]]("[1, 2, 3]") == [1, 2, 3]
  check parseJson[array[3, seq[int]]]("[[], [1], [2, 3]]") == [newSeq[int](), @[1], @[2, 3]]
  check parseJson[Table[string, float]]("{\"x\": 1, \"y\": 2.5}") == toTable({"x": 1.0, "y": 2.5})
  check parseJson[Table[string, seq[int]]]("{\"x\": [], \"y\": [1], \"z\": [2, 3]}") == toTable({
    "x": newSeq[int](), "y": @[1], "z": @[2, 3]
  })
  check parseJson[Table[string, seq[int]]]("{}") == initTable[string, seq[int]]()
  check parseJson[Table[string, int]]("{\"one\": 1}") == toTable({"one": 1})
  check parseJson[Table[string, bool]]("{\"true\": true}") == toTable({"true": true})

type
  TestObjKind = enum
    TestObjArray, TestObjSeq
  
  TestObj = object
    x: int
    y: string
    z: Table[string, bool]
  
  TestObj2 = object
    boolVal: bool
    intVal: int 
    floatVal: float
    tableVal: Table[string, int]
    case kind: TestObjKind:
      of TestObjArray: arrayVal: array[3, int]
      of TestObjSeq: seqVal: seq[int]

proc `==`(a, b: TestObj2): bool =
  result = a.boolVal == b.boolVal and
           a.intVal == b.intVal and
           a.floatVal == b.floatVal and
           a.tableVal == b.tableVal
  if result and a.kind == b.kind:
    case a.kind:
      of TestObjArray: result = a.arrayVal == b.arrayVal
      of TestObjSeq: result = a.seqVal == b.seqVal
  else:
    result = false

jsonSerializable(TestObjKind, TestObj, TestObj2)

test "json_serializable":
  check parseJson[TestObjKind]("0") == TestObjArray
  check parseJson[TestObjKind]("1") == TestObjSeq
  check parseJson[TestObj]("{}") == TestObj()
  check parseJson[TestObj]("{\"x\": 1}") == TestObj(x: 1)
  check parseJson[TestObj]("{\"y\": \"Hello\", \"x\": 2}") == TestObj(x: 2, y: "Hello")
  check parseJson[TestObj]("{\"y\": \"Test\", \"x\": 3, \"z\": {\"true\": true}}") == TestObj(
    x: 3, y: "Test", z: toTable({"true": true})
  )
  check parseJson[TestObj2]("{}") == TestObj2()
  check parseJson[TestObj2]("{\"boolVal\": true}") == TestObj2(boolVal: true)
  check parseJson[TestObj2]("{\"boolVal\": true, \"tableVal\": {}}") == TestObj2(boolVal: true)
  check parseJson[TestObj2]("{\"boolVal\": true, \"tableVal\": {\"zero\": 0}}") == TestObj2(
    boolVal: true, tableVal: toTable({"zero": 0})
  )
  check parseJson[TestObj2]("{\"boolVal\": true, \"tableVal\": {\"zero\": 0}, \"kind\": 1}") == TestObj2(
    boolVal: true, tableVal: toTable({"zero": 0}), kind: TestObjSeq
  )
  check parseJson[TestObj2]("{\"seqVal\": [1, 2, 3, 4], \"kind\": 1}") == TestObj2(
    kind: TestObjSeq, seqVal: @[1, 2, 3, 4]
  )
  check parseJson[TestObj2]("{\"seqVal\": [1, 2, 3, 4], \"kind\": 0}") == TestObj2(kind: TestObjArray)
  check parseJson[TestObj2]("{\"arrayVal\": [1, 2, 3], \"kind\": 0}") == TestObj2(
    kind: TestObjArray, arrayVal: [1, 2, 3]
  )

test "json_node":
  check parseJson[JsonNode]("1") == newJInt(1)
  check parseJson[JsonNode]("123") == newJInt(123)
  check parseJson[JsonNode]("3.14") == newJFloat(3.14)
  check parseJson[JsonNode]("\"Hello, world!\\n\"") == newJString("Hello, world!\n")
  check parseJson[JsonNode]("[1, 2, 3]") == newJarray([newJInt(1), newJInt(2), newJInt(3)])
  check parseJson[JsonNode]("[[null, 1, 2, 2.5], true, false]") == newJarray([
    newJarray([newJNull(), newJInt(1), newJInt(2), newJFloat(2.5)]),
    newJBool(true), newJBool(false)
  ])
  check parseJson[JsonNode]("{\"x\": 1, \"y\": 2}") == newJobject({
    "x": newJInt(1), "y": newJInt(2)
  })
