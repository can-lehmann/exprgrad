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

import std/[terminal, macros, sets]

type TestError = ref object of CatchableError
  env: seq[(string, string)]

template test*(name: string, body: untyped) =
  var
    success = true
    env: seq[(string, string)]
  try:
    body
  except TestError as err:
    success = false
    env = err.env
  
  if success:
    stdout.set_foreground_color(fgGreen)
    stdout.write("[✓] ")
    stdout.reset_attributes()
    stdout.write(name)
    stdout.write("\n")
  else:
    stdout.write("\n")
    stdout.set_foreground_color(fgRed)
    stdout.write("Test Failed: ")
    stdout.reset_attributes()
    stdout.write(name)
    stdout.write("\n")
    for (var_name, value) in env:
      stdout.set_foreground_color(fgRed)
      stdout.write(var_name & ": ")
      stdout.reset_attributes()
      stdout.write(value)
      stdout.write("\n")
    stdout.write("\n")
  stdout.flush_file()

template subtest*(body: untyped) =
  block:
    body

proc collect_env(node: NimNode): HashSet[string] =
  case node.kind:
    of nnkIdent, nnkSym:
      result.incl(node.str_val)
    of nnkCallKinds:
      for it in 1..<node.len:
        result = union(result, node[it].collect_env())
    of nnkDotExpr:
      result = node[0].collect_env()
    else:
      for child in node:
        result = union(result, child.collect_env())

proc stringify_env_var[T](x: T): string =
  when compiles($x):
    result = $x
  else:
    result = "..."

macro check*(cond: untyped): untyped =
  let cond_str = repr(cond)
  var env = new_nim_node(nnkBracket)
  for name in cond.collect_env():
    env.add(new_tree(nnkTupleConstr, [
      new_lit(name), new_call(bind_sym("stringify_env_var"), ident(name))
    ]))
  env = new_call(bind_sym("@"), env)
  result = quote:
    if not `cond`:
      raise TestError(msg: `cond_str`, env: `env`)
