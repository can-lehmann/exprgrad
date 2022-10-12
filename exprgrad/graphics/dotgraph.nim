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

import std/[tables, sets, strutils, sequtils]
import ../ir, ../irprint

type
  Node = object
    name: string
    attrs: seq[(string, string)]
  
  Edge = object
    a, b: string
    attrs: seq[(string, string)]
  
  DotGraph = object
    attrs: seq[(string, string)]
    
    nodes: seq[Node]
    edges: seq[Edge]

proc escapeValue(val: string): string =
  result = newStringOfCap(val.len)
  for chr in val:
    case chr:
      of '\"': result.add("\\\"")
      else: result.add(chr)

proc formatAttrs(attrs: openArray[(string, string)]): string =
  result = "["
  for it, (name, value) in attrs:
    if it != 0:
      result &= ", "
    result &= name & "=\"" & value.escapeValue() & "\""
  result &= "]"

proc `$`(graph: DotGraph): string =
  result = "digraph {"
  if graph.attrs.len > 0:
    result &= "\n\tgraph " & graph.attrs.formatAttrs()
  for node in graph.nodes:
    result &= "\n\t" & node.name
    if node.attrs.len > 0:
      result &= " " & formatAttrs(node.attrs)
    result &= ";"
  for edge in graph.edges:
    result &= "\n\t" & edge.a & " -> " & edge.b
    if edge.attrs.len > 0:
      result &= " " & formatAttrs(edge.attrs)
    result &= ";"
  result &= "\n}"

proc toDotGraph*(program: Program,
                 target: string,
                 dpi: int = 200,
                 fontName: string = "",
                 background: string = "#f0f0f0"): string =
  program.assertGen("toDotGraph", requires={})
  
  var
    graph = DotGraph(attrs: @{"dpi": $dpi, "bgcolor": background})
    deps = initTable[TensorId, HashSet[TensorId]]()
    tensors = initHashSet[TensorId]()
  
  for kernel in program.targets[target].kernels:
    var
      inputs = initHashSet[TensorId]()
      outputs = initHashSet[TensorId]()
    for read in kernel.reads:
      inputs.incl(read.tensor)
    if kernel.write.tensor != TensorId(0):
      outputs.incl(kernel.write.tensor)
    if kernel.generator.kind == GenReshape:
      inputs.incl(kernel.generator.tensor)
    
    for outp in outputs:
      if outp notin deps:
        deps[outp] = initHashSet[TensorId]()
      for inp in inputs:
        deps[outp].incl(inp)
    tensors.incl(inputs)
    tensors.incl(outputs)
  
  for tensor in tensors:
    let def = program.tensors[tensor]
    var label = def.name
    if label.len == 0:
      label = $tensor
    if def.shape.len > 0:
      label &= " [" & def.shape.mapIt($it).join(", ") & "]"
    
    var attrs = @{
      "label": label,
      "shape": "box"
    }
    case def.kind:
      of TensorInput:
        attrs.add(("penwidth", "4"))
      of TensorParam:
        attrs.add(("shape", "parallelogram"))
      else: discard
    
    if fontName.len > 0:
      attrs.add(("fontname", fontName))
    
    graph.nodes.add(Node(name: $tensor, attrs: attrs))
  
  for output, inputs in deps:
    for input in inputs:
      graph.edges.add(Edge(a: $input, b: $output))
  
  result = $graph

