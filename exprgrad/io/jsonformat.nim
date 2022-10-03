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

# Fast JSON parser

import std/[macros, strutils, tables]
import std/json except parseJson
import faststreams

proc readName(stream: var ReadStream): string =
  result = stream.readUntil({' ', '\n', '\t', '\r', '{', '}', '[', ']', '\"', ',', ':'})

proc parseJson*(stream: var ReadStream, value: var bool) =
  stream.skipWhitespace()
  let name = stream.readName()
  case name:
    of "true": value = true
    of "false": value = false
    else: raise newException(ValueError, name & " is not a valid bool value")

proc parseJson*(stream: var ReadStream, value: var int) =
  stream.skipWhitespace()
  var isNegated = false
  if stream.takeChar('-'):
    isNegated = true
  else:
    discard stream.takeChar('+')
  value = 0
  while stream.peekChar() in '0'..'9':
    value *= 10
    value += ord(stream.readChar()) - ord('0')
  if isNegated:
    value *= -1

proc parseJson*(stream: var ReadStream, value: var float) =
  stream.skipWhitespace()
  value = stream.readName().parseFloat()

proc parseJson*(stream: var ReadStream, value: var string) =
  value = ""
  stream.skipWhitespace()
  if not stream.takeChar('\"'):
    raise newException(ValueError, "Expected \" at start of string")
  
  while not stream.takeChar('\"'):
    if stream.atEnd:
      raise newException(ValueError, "Expected \" at end of string")
    
    let chr = stream.readChar()
    if chr == '\\':
      case stream.readChar():
        of '\"': value.add('\"')
        of '\\': value.add('\\')
        of '/': value.add('/')
        of 'b': value.add(char(8))
        of 'f': value.add(char(12))
        of 'n': value.add('\n')
        of 'r': value.add('\r')
        of 't': value.add('\t')
        of 'u': raise newException(ValueError, "Not implemented")
        else:
          raise newException(ValueError, "Invalid escape code")
    else:
      value.add(chr)

iterator iterJsonArray*(stream: var ReadStream): int =
  stream.skipWhitespace()
  if not stream.takeChar('['):
    raise newException(ValueError, "Expected [ at start of json array")
  stream.skipWhitespace()
  var it = 0
  while true:
    if stream.peekChar() == ']':
      break
    yield it
    stream.skipWhitespace()
    if not stream.takeChar(','):
      break
    stream.skipWhitespace()
    it += 1
  if not stream.takeChar(']'):
    raise newException(ValueError, "Expected ] at end of json array")

iterator iterJsonObject*(stream: var ReadStream): string =
  stream.skipWhitespace()
  if not stream.takeChar('{'):
    raise newException(ValueError, "Expected { at start of json array")
  
  stream.skipWhitespace()
  while true:
    if stream.peekChar() == '}':
      break
    var name = ""
    stream.parseJson(name)
    stream.skipWhitespace()
    if not stream.takeChar(':'):
      raise newException(ValueError, "Expected colon after object key")
    yield name
    stream.skipWhitespace()
    if not stream.takeChar(','):
      break
    stream.skipWhitespace()
  
  if not stream.takeChar('}'):
    raise newException(ValueError, "Expected } at end of json object")

proc parseJson*[T](stream: var ReadStream, value: var seq[T]) =
  mixin parseJson
  value = newSeq[T]()
  for it in stream.iterJsonArray():
    var item: T
    stream.parseJson(item)
    value.add(item)

proc parseJson*[L, T](stream: var ReadStream, value: var array[L, T]) =
  mixin parseJson
  var count = 0
  for it in stream.iterJsonArray():
    stream.parseJson(value[it])
    count += 1
  if count != value.len:
    raise newException(ValueError, "Expected exactly " & $value.len & " items in array")

proc parseJson*[T](stream: var ReadStream, tab: var Table[string, T]) =
  mixin parseJson
  tab = initTable[string, T]()
  for name in stream.iterJsonObject():
    var value: T
    stream.parseJson(value)
    tab[name] = value

proc parseJson*(stream: var ReadStream, node: var JsonNode) =
  stream.skipWhitespace()
  case stream.peekChar():
    of '[':
      node = newJArray()
      for it in stream.iterJsonArray():
        var child: JsonNode = nil
        stream.parseJson(child)
        node.add(child)
    of '{':
      node = newJObject()
      for name in stream.iterJsonObject():
        var child: JsonNode = nil
        stream.parseJson(child)
        node[name] = child
    of '\"':
      var str = ""
      stream.parseJson(str)
      node = newJString(str)
    else:
      let name = stream.readName()
      case name:
        of "null":
          node = newJNull()
        of "true", "false":
          node = newJBool(name == "true")
        else:
          if '.' in name:
            node = newJFloat(parseFloat(name))
          else:
            node = newJInt(parseInt(name))

type
  FieldKind = enum
    FieldNone, FieldDiscr, FieldCase
  
  Field = object
    name: string
    typ: NimNode
    kind: FieldKind

proc unwrapName(node: NimNode): NimNode =
  result = node
  while result.kind notin {nnkIdent, nnkSym}:
    case result.kind:
      of nnkPostfix: result = result[1]
      else: error("Unable to unwrap name from " & $result.kind)

proc collectFields(node: NimNode, kind: FieldKind = FieldNone): seq[Field] =
  case node.kind:
    of nnkRecList, nnkElse:
      for def in node:
        result.add(def.collectFields(kind))
    of nnkIdentDefs:
      for it in 0..<(node.len - 2):
        result.add(Field(
          name: node[it].unwrapName().strVal,
          typ: node[^2],
          kind: kind
        ))
    of nnkRecCase:
      result.add(node[0].collectFields(FieldDiscr))
      for it in 1..<node.len:
        result.add(node[it].collectFields(FieldCase))
    of nnkOfBranch:
      for it in 1..<node.len:
        result.add(node[it].collectFields(kind))
    else:
      discard

proc genFieldAssignments(node: NimNode, stmts: NimNode) =
  case node.kind:
    of nnkRecList:
      for child in node:
        child.genFieldAssignments(stmts)
    of nnkIdentDefs:
      let value = ident("value")
      for it in 0..<(node.len - 2):
        let name = node[it].unwrapName().strVal
        stmts.add(newAssignment(
          newTree(nnkDotExpr, value, ident(name)),
          ident(name)
        ))
    of nnkOfBranch:
      let body = newStmtList()
      for it in 1..<node.len:
        node[it].genFieldAssignments(body)
      stmts.add(newTree(nnkOfBranch, node[0], body))
    of nnkElse:
      let body = newStmtList()
      for child in node:
        child.genFieldAssignments(body)
      stmts.add(newTree(nnkElse, body))
    of nnkRecCase:
      let
        discr = node[0][0].unwrapName().strVal
        caseStmt = newTree(nnkCaseStmt, ident(discr))
      for it in 1..<node.len:
        node[it].genFieldAssignments(caseStmt)
      stmts.add(caseStmt)
    else:
      discard

proc genParser(typ, name, stmts: NimNode) =
  let (stream, value) = (ident("stream"), ident("value"))
  case typ.kind:
    of nnkSym, nnkIdent:
      typ.getImpl().genParser(typ, stmts)
    of nnkTypeDef:
      typ[^1].genParser(name, stmts)
    of nnkEnumTy:
      stmts.add: quote:
        var id: int
        parseJson(`stream`, id)
        `value` = `name`(id)
    of nnkObjectTy:
      typ[2].genParser(name, stmts)
    of nnkRecList:
      var
        fields = typ.collectFields()
      
      let varSection = newTree(nnkVarSection)
      for field in fields:
        varSection.add(newTree(nnkIdentDefs,
          ident(field.name), field.typ, newEmptyNode()
        ))
      stmts.add(varSection)
      
      let
        fieldNameSym = genSym(nskForVar, "fieldName")
        caseStmt = newTree(nnkCaseStmt, newCall(
          bindSym("nimIdentNormalize"), fieldNameSym
        ))
      for field in fields:
        var target = ident(field.name)
        caseStmt.add(newTree(nnkOfBranch,
          newLit(nimIdentNormalize(field.name)),
          newStmtList(newCall("parseJson", stream, target))
        ))
      caseStmt.add: newTree(nnkElse): quote:
        raise newException(ValueError, "Unknown field")
      stmts.add(newTree(nnkForStmt,
        fieldNameSym,
        newCall(bindSym("iterJsonObject"), stream),
        newStmtList(caseStmt)
      ))
      
      let constr = newTree(nnkObjConstr, name)
      for field in fields:
        if field.kind == FieldDiscr:
          constr.add(newTree(nnkExprColonExpr,
            ident(field.name), ident(field.name)
          ))
      stmts.add(newAssignment(value, constr))
      typ.genFieldAssignments(stmts)
    else:
      error("Unable to generate json parser for " & $typ.kind)

proc genParser(typ: NimNode): NimNode =
  let
    body = newStmtList()
    (stream, value) = (ident("stream"), ident("value"))
  typ.genParser(nil, body)
  result = quote:
    proc parseJson(`stream`: var ReadStream, `value`: var `typ`) =
      `body`

macro jsonSerializable*(types: varargs[typed]): untyped =
  result = newStmtList()
  for typ in types:
    result.add(typ.genParser())
  echo result.repr

proc parseJson*[T](str: string): T =
  mixin parseJson
  var stream = initReadStream(str)
  defer: stream.close()
  stream.parseJson(result)

proc loadJson*[T](path: string): T =
  mixin parseJson
  var stream = openReadStream(path)
  defer: stream.close()
  stream.parseJson(result)

proc newJarray*(children: openArray[JsonNode]): JsonNode =
  result = newJArray()
  for child in children:
    result.add(child)

proc newJobject*(children: openArray[(string, JsonNode)]): JsonNode =
  result = newJObject()
  for (name, value) in children:
    result[name] = value

export JsonNode, newJNull, newJBool, newJInt, newJFloat
export newJString, newJArray, newJObject, json.`$`, json.`==`
