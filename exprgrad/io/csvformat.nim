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

template csv_column*(name: string) {.pragma.}
template csv_ignore*(columns: varargs[string]) {.pragma.}
template csv_parser*(parser: proc) {.pragma.}

proc parse_csv_cell*(str: string, value: var int) = value = parse_int(str.strip())
proc parse_csv_cell*(str: string, value: var float) = value = parse_float(str.strip())
proc parse_csv_cell*(str: string, value: var string) = value = str

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

proc read_csv_row(stream: var ReadStream, format = DEFAULT_CSV_FORMAT): seq[string] =
  var
    value = ""
    is_str = false
  while not stream.at_end():
    let chr = stream.read_char()
    if chr == format.newline and not is_str:
      break
    elif chr == format.delim and not is_str:
      result.add(value)
      value = ""
    elif chr == format.escape:
      let escaped = stream.read_char()
      case escaped:
        of 'n': value.add('\n')
        of 'r': value.add('\r')
        of 't': value.add('\t')
        else:
          value.add(escaped)
    elif chr == format.quote:
      is_str = not is_str
    else:
      value.add(chr)
  result.add(value)

type Field = object
  name: string
  column: string
  parser: NimNode

proc collect_fields(typ: NimNode): seq[Field] =
  case typ.kind:
    of nnkSym:
      result = typ.get_impl().collect_fields()
    of nnkTypeDef, nnkObjectTy:
      result = typ[^1].collect_fields()
    of nnkRecList:
      for child in typ:
        result.add(child.collect_fields())
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
                case nim_ident_normalize(pragma[0].str_val):
                  of "csvcolumn": field.column = pragma[1].str_val
                  of "csvparser": field.parser = pragma[1]
                  else: discard
              ident = ident[0]
            else:
              error("")
        field.name = ident.str_val
        if field.column.len == 0:
          field.column = field.name
        if field.parser.is_nil:
          field.parser = ident("parse_csv_cell")
        result.add(field)
    else:
      error("Unable to collect fields for " & $typ.kind)

proc collect_ignored(typ: NimNode): seq[string] =
  case typ.kind:
    of nnkSym: result = typ.get_impl().collect_ignored()
    of nnkTypeDef:
      if typ[0].kind == nnkPragmaExpr:
        result = typ[0].collect_ignored()
    of nnkPostfix: result = typ[1].collect_ignored()
    of nnkStrLit: result = @[typ.str_val]
    of nnkBracket:
      for child in typ:
        result.add(child.collect_ignored())
    of nnkPragmaExpr:
      for pragma in typ[1]:
        if pragma.len >= 2 and
           pragma[0].kind in {nnkSym, nnkIdent} and
           nim_ident_normalize(pragma[0].str_val) == "csvignore":
          for it in 1..<pragma.len:
            result.add(pragma[it].collect_ignored())
    else: discard

macro build_cell_indices(header, typ: typed): untyped =
  result = new_stmt_list()
  let
    indices = ident("indices")
    (index, column) = (ident("index"), ident("column"))
  let
    type_inst = get_type_inst(typ)[1]
    fields = type_inst.collect_fields()
    ignored = type_inst.collect_ignored()
    tuple_constr = new_nim_node(nnkTupleConstr)
    case_stmt = new_tree(nnkCaseStmt, column)
  
  for field in fields:
    tuple_constr.add(new_tree(nnkExprColonExpr, ident(field.name), new_lit(-1)))
    case_stmt.add(new_tree(nnkOfBranch, 
      new_lit(field.column),
      new_stmt_list(new_assignment(
        new_tree(nnkDotExpr, indices, ident(field.name)), index
      ))
    ))
  
  for column in ignored:
    case_stmt.add(new_tree(nnkOfBranch, new_lit(column),
      new_stmt_list(new_tree(nnkDiscardStmt, new_empty_node()))
    ))
  
  let else_body = quote:
    raise new_exception(ValueError,
      "\"" & `column` & "\" is not a known column. " &
      "Add a {.csv_ignore: [\"" & `column` & "\"].} pragma to the row type to ignore it."
    )
  case_stmt.add(new_tree(nnkElse, else_body))
  
  result.add(new_var_stmt(indices, tuple_constr))
  result.add(new_tree(nnkForStmt, index, column, header, new_stmt_list(case_stmt)))
  result.add(indices)
  result = new_tree(nnkBlockStmt, new_empty_node(), result)

macro parse_row(row, obj, indices: typed): untyped =
  result = new_stmt_list()
  let fields = get_type_inst(obj).collect_fields()
  for field in fields:
    result.add(new_call(field.parser,
      new_tree(nnkBracketExpr, row,
        new_tree(nnkDotExpr, indices, ident(field.name))
      ),
      new_tree(nnkDotExpr, obj, ident(field.name))
    ))

iterator iter_csv*[T](stream: var ReadStream, format = DEFAULT_CSV_FORMAT): T =
  let
    header = stream.read_csv_row(format)
    indices = build_cell_indices(header, T)
  while not stream.at_end():
    let row = stream.read_csv_row(format)
    if row.len != header.len:
      continue
    var obj = T()
    parse_row(row, obj, indices)
    yield obj

iterator iter_csv*[T](path: string, format = DEFAULT_CSV_FORMAT): T =
  var stream = open_read_stream(path)
  defer: stream.close()
  for row in iter_csv[T](stream, format):
    yield row
