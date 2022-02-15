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
import std/json except parse_json
import faststreams

proc read_name(stream: var ReadStream): string =
  result = stream.read_until({' ', '\n', '\t', '\r', '{', '}', '[', ']', '\"', ',', ':'})

proc parse_json*(stream: var ReadStream, value: var bool) =
  let name = stream.read_name()
  case name:
    of "true": value = true
    of "false": value = false
    else: raise new_exception(ValueError, name & " is not a valid bool value")

proc parse_json*(stream: var ReadStream, value: var int) =
  stream.skip_whitespace()
  var is_negated = false
  if stream.take_char('-'):
    is_negated = true
  else:
    discard stream.take_char('+')
  value = 0
  while stream.peek_char() in '0'..'9':
    value *= 10
    value += ord(stream.read_char()) - ord('0')
  if is_negated:
    value *= -1

proc parse_json*(stream: var ReadStream, value: var float) =
  stream.skip_whitespace()
  value = stream.read_name().parse_float()

proc parse_json*(stream: var ReadStream, value: var string) =
  value = ""
  stream.skip_whitespace()
  if not stream.take_char('\"'):
    raise new_exception(ValueError, "Expected \" at start of string")
  
  while not stream.take_char('\"'):
    if stream.at_end:
      raise new_exception(ValueError, "Expected \" at end of string")
    
    let chr = stream.read_char()
    if chr == '\\':
      case stream.read_char():
        of '\"': value.add('\"')
        of '\\': value.add('\\')
        of '/': value.add('/')
        of 'b': value.add(char(8))
        of 'f': value.add(char(12))
        of 'n': value.add('\n')
        of 'r': value.add('\r')
        of 't': value.add('\t')
        of 'u': raise new_exception(ValueError, "Not implemented")
        else:
          raise new_exception(ValueError, "Invalid escape code")
    else:
      value.add(chr)

iterator iter_json_array*(stream: var ReadStream): int =
  stream.skip_whitespace()
  if not stream.take_char('['):
    raise new_exception(ValueError, "Expected [ at start of json array")
  stream.skip_whitespace()
  var it = 0
  while true:
    if stream.peek_char() == ']':
      break
    yield it
    stream.skip_whitespace()
    if not stream.take_char(','):
      break
    stream.skip_whitespace()
    it += 1
  if not stream.take_char(']'):
    raise new_exception(ValueError, "Expected ] at end of json array")

iterator iter_json_object*(stream: var ReadStream): string =
  stream.skip_whitespace()
  if not stream.take_char('{'):
    raise new_exception(ValueError, "Expected { at start of json array")
  
  stream.skip_whitespace()
  while true:
    if stream.peek_char() == '}':
      break
    var name = ""
    stream.parse_json(name)
    stream.skip_whitespace()
    if not stream.take_char(':'):
      raise new_exception(ValueError, "Expected colon after object key")
    yield name
    stream.skip_whitespace()
    if not stream.take_char(','):
      break
    stream.skip_whitespace()
  
  if not stream.take_char('}'):
    raise new_exception(ValueError, "Expected } at end of json object")

proc parse_json*[T](stream: var ReadStream, value: var seq[T]) =
  mixin parse_json
  value = new_seq[T]()
  for it in stream.iter_json_array():
    var item: T
    stream.parse_json(item)
    value.add(item)

proc parse_json*[L, T](stream: var ReadStream, value: var array[L, T]) =
  mixin parse_json
  var count = 0
  for it in stream.iter_json_array():
    stream.parse_json(value[it])
    count += 1
  if count != value.len:
    raise new_exception(ValueError, "Expected exactly " & $value.len & " items in array")

proc parse_json*[T](stream: var ReadStream, tab: var Table[string, T]) =
  mixin parse_json
  tab = init_table[string, T]()
  for name in stream.iter_json_object():
    var value: T
    stream.parse_json(value)
    tab[name] = value

proc parse_json*(stream: var ReadStream, node: var JsonNode) =
  stream.skip_whitespace()
  case stream.peek_char():
    of '[':
      node = new_jarray()
      for it in stream.iter_json_array():
        var child: JsonNode = nil
        stream.parse_json(child)
        node.add(child)
    of '{':
      node = new_jobject()
      for name in stream.iter_json_object():
        var child: JsonNode = nil
        stream.parse_json(child)
        node[name] = child
    of '\"':
      var str = ""
      stream.parse_json(str)
      node = new_jstring(str)
    else:
      let name = stream.read_name()
      case name:
        of "null":
          node = new_jnull()
        of "true", "false":
          node = new_jbool(name == "true")
        else:
          if '.' in name:
            node = new_jfloat(parse_float(name))
          else:
            node = new_jint(parse_int(name))

macro serialize_json*(typ: typed) =
  discard # TODO

proc parse_json*[T](str: string): T =
  mixin parse_json
  var stream = init_read_stream(str)
  defer: stream.close()
  stream.parse_json(result)

proc load_json*[T](path: string): T =
  mixin parse_json
  var stream = open_read_stream(path)
  defer: stream.close()
  stream.parse_json(result)

proc new_jarray*(children: openArray[JsonNode]): JsonNode =
  result = new_jarray()
  for child in children:
    result.add(child)

proc new_jobject*(children: openArray[(string, JsonNode)]): JsonNode =
  result = new_jobject()
  for (name, value) in children:
    result[name] = value

export JsonNode, new_jnull, new_jbool, new_jint, new_jfloat
export new_jstring, new_jarray, new_jobject, json.`$`, json.`==`
