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

# Render programs as DOT graphs 

import std/[tables, sets, strutils]
import ../ir

type
  Node = object
    name: string
    attrs: seq[(string, string)]
  
  Edge = object
    a, b: string
    attrs: seq[(string, string)]
  
  DotGraph = object
    nodes: seq[Node]
    edges: seq[Edge]

proc escape_value(val: string): string =
  result = new_string_of_cap(val.len)
  for chr in val:
    case chr:
      of '\"': result.add("\\\"")
      else: result.add(chr)

proc format_attrs(attrs: openArray[(string, string)]): string =
  result = "["
  for it, (name, value) in attrs:
    if it != 0:
      result &= ", "
    result &= name & "=\"" & value.escape_value() & "\""
  result &= "]"

proc `$`(graph: DotGraph): string =
  result = "digraph {"
  for node in graph.nodes:
    result &= "\n\t" & node.name
    if node.attrs.len > 0:
      result &= " " & format_attrs(node.attrs)
    result &= ";"
  for edge in graph.edges:
    result &= "\n\t" & edge.a & " -> " & edge.b
    if edge.attrs.len > 0:
      result &= " " & format_attrs(edge.attrs)
    result &= ";"
  result &= "\n}"

proc to_dot_graph*(program: Program, target: string): string =
  var
    graph = DotGraph()
    deps = init_table[TensorId, HashSet[TensorId]]()
  for tensor in program.targets[target].tensors:
    let def = program.tensors[tensor]
    var label = def.name
    if label.len == 0:
      label = $tensor
    if def.kind != TensorResult:
      label = ($def.kind)[len("Tensor")..^1].to_lower_ascii() & " " & label
    if def.shape.len > 0:
      label &= " " & $def.shape
    graph.nodes.add(Node(name: $tensor, attrs: @{
      "label": label,
      "shape": "box"
    }))
    deps[tensor] = init_hash_set[TensorId]()
  
  for kernel in program.targets[target].kernels:
    var
      inputs = init_hash_set[TensorId]()
      outputs = init_hash_set[TensorId]()
    for read in kernel.reads:
      inputs.incl(read.tensor)
    if kernel.write.tensor != TensorId(0):
      outputs.incl(kernel.write.tensor)
    for instr in kernel.expr.instrs:
      case instr.kind:
        of InstrRead: inputs.incl(instr.tensor)
        of InstrWrite: outputs.incl(instr.tensor)
        else: discard
    
    for outp in outputs:
      for inp in inputs:
        deps[outp].incl(inp)
  
  for output, inputs in deps:
    for input in inputs:
      graph.edges.add(Edge(a: $input, b: $output))
  
  result = $graph

