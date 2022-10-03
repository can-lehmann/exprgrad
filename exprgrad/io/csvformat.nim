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

# Parse CSV files

import std/[macros, strutils]
import faststreams

template csvColumn*(name: string) {.pragma.}
template csvIgnore*(columns: varargs[string]) {.pragma.}
template csvParser*(parser: proc) {.pragma.}

proc parseCsvCell*(str: string, value: var int) = value = parseInt(str.strip())
proc parseCsvCell*(str: string, value: var float) = value = parseFloat(str.strip())
proc parseCsvCell*(str: string, value: var string) = value = str

type CsvFormat* = object
  delim*: char
  quote*: char
  escape*: char
  newline*: char

const DEFAULT_CSV_FORMAT* = CsvFormat(
  delim: ';',
  quote: '\"',
  escape: '\\',
  newline: '\n'
)

proc readCsvRow(stream: var ReadStream, format = DEFAULT_CSV_FORMAT): seq[string] =
  var
    value = ""
    isStr = false
  while not stream.atEnd():
    let chr = stream.readChar()
    if chr == format.newline and not isStr:
      break
    elif chr == format.delim and not isStr:
      result.add(value)
      value = ""
    elif chr == format.escape:
      let escaped = stream.readChar()
      case escaped:
        of 'n': value.add('\n')
        of 'r': value.add('\r')
        of 't': value.add('\t')
        else:
          value.add(escaped)
    elif chr == format.quote:
      isStr = not isStr
    else:
      value.add(chr)
  result.add(value)

type Field = object
  name: string
  column: string
  parser: NimNode

proc collectFields(typ: NimNode): seq[Field] =
  case typ.kind:
    of nnkSym:
      result = typ.getImpl().collectFields()
    of nnkTypeDef, nnkObjectTy:
      result = typ[^1].collectFields()
    of nnkRecList:
      for child in typ:
        result.add(child.collectFields())
    of nnkRecCase:
      error("Variant types are not allowed in CSV rows.")
    of nnkIdentDefs:
      for it in 0..<(typ.len - 2):
        var
          ident = typ[it]
          field = Field()
        while ident.kind notin {nnkSym, nnkIdent}:
          case ident.kind:
            of nnkPostfix: ident = ident[1]
            of nnkPragmaExpr:
              for pragma in ident[1]:
                if pragma.len < 2 or pragma[0].kind notin {nnkSym, nnkIdent}:
                  continue
                case nimIdentNormalize(pragma[0].strVal):
                  of "csvcolumn": field.column = pragma[1].strVal
                  of "csvparser": field.parser = pragma[1]
                  else: discard
              ident = ident[0]
            else:
              error("")
        field.name = ident.strVal
        if field.column.len == 0:
          field.column = field.name
        if field.parser.isNil:
          field.parser = ident("parseCsvCell")
        result.add(field)
    else:
      error("Unable to collect fields for " & $typ.kind)

proc collectIgnored(typ: NimNode): seq[string] =
  case typ.kind:
    of nnkSym: result = typ.getImpl().collectIgnored()
    of nnkTypeDef:
      if typ[0].kind == nnkPragmaExpr:
        result = typ[0].collectIgnored()
    of nnkPostfix: result = typ[1].collectIgnored()
    of nnkStrLit: result = @[typ.strVal]
    of nnkBracket:
      for child in typ:
        result.add(child.collectIgnored())
    of nnkPragmaExpr:
      for pragma in typ[1]:
        if pragma.len >= 2 and
           pragma[0].kind in {nnkSym, nnkIdent} and
           nimIdentNormalize(pragma[0].strVal) == "csvignore":
          for it in 1..<pragma.len:
            result.add(pragma[it].collectIgnored())
    else: discard

macro buildCellIndices(header, typ: typed): untyped =
  result = newStmtList()
  let
    indices = ident("indices")
    (index, column) = (ident("index"), ident("column"))
  let
    typeInst = getTypeInst(typ)[1]
    fields = typeInst.collectFields()
    ignored = typeInst.collectIgnored()
    tupleConstr = newNimNode(nnkTupleConstr)
    caseStmt = newTree(nnkCaseStmt, column)
  
  for field in fields:
    tupleConstr.add(newTree(nnkExprColonExpr, ident(field.name), newLit(-1)))
    caseStmt.add(newTree(nnkOfBranch, 
      newLit(field.column),
      newStmtList(newAssignment(
        newTree(nnkDotExpr, indices, ident(field.name)), index
      ))
    ))
  
  for column in ignored:
    caseStmt.add(newTree(nnkOfBranch, newLit(column),
      newStmtList(newTree(nnkDiscardStmt, newEmptyNode()))
    ))
  
  let elseBody = quote:
    raise newException(ValueError,
      "\"" & `column` & "\" is not a known column. " &
      "Add a {.csvIgnore: [\"" & `column` & "\"].} pragma to the row type to ignore it."
    )
  caseStmt.add(newTree(nnkElse, elseBody))
  
  result.add(newVarStmt(indices, tupleConstr))
  result.add(newTree(nnkForStmt, index, column, header, newStmtList(caseStmt)))
  result.add(indices)
  result = newTree(nnkBlockStmt, newEmptyNode(), result)

macro parseRow(row, obj, indices: typed): untyped =
  result = newStmtList()
  let fields = getTypeInst(obj).collectFields()
  for field in fields:
    result.add(newCall(field.parser,
      newTree(nnkBracketExpr, row,
        newTree(nnkDotExpr, indices, ident(field.name))
      ),
      newTree(nnkDotExpr, obj, ident(field.name))
    ))

iterator iterCsv*[T](stream: var ReadStream, format = DEFAULT_CSV_FORMAT): T =
  let
    header = stream.readCsvRow(format)
    indices = buildCellIndices(header, T)
  while not stream.atEnd():
    let row = stream.readCsvRow(format)
    if row.len != header.len:
      continue
    var obj = T()
    parseRow(row, obj, indices)
    yield obj

iterator iterCsv*[T](path: string, format = DEFAULT_CSV_FORMAT): T =
  var stream = openReadStream(path)
  defer: stream.close()
  for row in iter_csv[T](stream, format):
    yield row
