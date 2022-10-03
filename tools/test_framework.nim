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

# A simple unit-testing framework

import std/[terminal, macros, sets, exitprocs]

var quitCode = QuitSuccess
addExitProc(proc() {.closure.} =
  quit(quitCode)
)

type TestError = ref object of CatchableError
  env: seq[(string, string)]
  line, column: int

template test*(name: string, body: untyped) =
  var error: TestError = nil
  try:
    body
  except TestError as err:
    error = err

  if error.isNil:
    stdout.setForegroundColor(fgGreen)
    stdout.write("[✓] ")
    stdout.resetAttributes()
    stdout.write(name)
    stdout.write("\n")
  else:
    quitCode = QuitFailure
    stdout.write("\n")
    stdout.setForegroundColor(fgRed)
    stdout.write("Test Failed: ")
    stdout.resetAttributes()
    stdout.write(name)
    stdout.write(" (" & $error.line & ", " & $error.column & ")")
    stdout.write("\n")
    for (varName, value) in error.env:
      stdout.setForegroundColor(fgRed)
      stdout.write(varName & ": ")
      stdout.resetAttributes()
      stdout.write(value)
      stdout.write("\n")
    stdout.write("\n")
  stdout.flushFile()

template subtest*(body: untyped) =
  block:
    body

proc collectEnv(node: NimNode): HashSet[string] =
  case node.kind:
    of nnkIdent, nnkSym:
      result.incl(node.strVal)
    of nnkCallKinds, nnkObjConstr:
      for it in 1..<node.len:
        result = union(result, node[it].collectEnv())
    of nnkDotExpr:
      result = node[0].collectEnv()
    of nnkExprColonExpr:
      result = node[1].collectEnv()
    else:
      for child in node:
        result = union(result, child.collectEnv())

proc stringifyEnvVar[T](x: T): string =
  when compiles($x):
    result = $x
  else:
    result = "..."

macro check*(cond: untyped): untyped =
  let condStr = repr(cond)
  var env = newNimNode(nnkBracket)
  for name in cond.collectEnv():
    env.add(newTree(nnkTupleConstr, [
      newLit(name), newCall(bindSym("stringifyEnvVar"), ident(name))
    ]))
  env = newCall(bindSym("@"), env)
  let
    line = cond.lineInfoObj.line
    column = cond.lineInfoObj.column
  result = quote:
    if not `cond`:
      raise TestError(msg: `condStr`, env: `env`, line: `line`, column: `column`)

macro checkException*(exception, body: untyped): untyped =
  let
    msg = "Expected block to raise " & repr(exception)
    line = body.lineInfoObj.line
    column = body.lineInfoObj.column
  
  result = quote:
    block:
      var raised = false
      try:
        `body`
      except `exception`:
        raised = true
      if not raised:
        raise TestError(msg: `msg`, line: `line`, column: `column`)
