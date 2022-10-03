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

# Common abstraction over vector graphics backends

import std/[macros, strutils]
import geometry

type Color* = object
  r*: uint8
  g*: uint8
  b*: uint8
  a*: uint8

{.push inline.}
proc rgba*(r, g, b, a: uint8): Color = Color(r: r, g: g, b: b, a: a)
proc rgb*(r, g, b: uint8): Color = Color(r: r, g: g, b: b, a: 255)
proc grey*(value: uint8): Color = Color(r: value, g: value, b: value, a: 255)
{.pop.}

proc toHex*(color: Color): string =
  const DIGITS = "0123456789abcdef"
  template component(value: uint8): string =
    DIGITS[int(value shr 4) and 0xf] & DIGITS[int(value) and 0xf]
  result = "#"
  result &= component(color.r)
  result &= component(color.g)
  result &= component(color.b)
  if color.a != 255:
    result &= component(color.a)

type
  PathPoint* = object
    pos: Vec2
  
  Path* = object
    points: seq[PathPoint]
    closed: bool
  
  ShapeKind* = enum
    ShapeRect, ShapeEllipse, ShapeLine, ShapePath
  
  ShapeStyle* = object
    stroke*: Color
    fill*: Color
    strokeWidth*: float64
  
  Shape* = object
    style*: ShapeStyle
    case kind*: ShapeKind:
      of ShapeRect, ShapeEllipse:
        pos*: Vec2
        size*: Vec2
      of ShapeLine:
        start*: Vec2
        stop*: Vec2
      of ShapePath:
        subpaths*: seq[Path]
  
  Canvas* = object
    size*: Vec2
    background*: Color
    shapes*: seq[Shape]

proc init*(_: typedesc[Canvas], size: Vec2, background: Color = Color()): Canvas =
  result = Canvas(size: size, background: background)

const DEFAULT_SHAPE_STYLE = ShapeStyle(stroke: grey(0), strokeWidth: 1.0)

macro defaultStyle(paramNode: untyped, default: static[ShapeStyle], procNode: untyped): untyped =
  result = procNode.copyNimTree()
  let
    name = nimIdentNormalize(paramNode.strVal)
    newParams = newTree(nnkFormalParams)
    call = newTree(nnkCall, procNode.name)
  for it, param in procNode.params:
    if it > 0 and param[0].eqIdent(name):
      let constr = newTree(nnkObjConstr, bindSym("ShapeStyle"))
      for name, value in default.fieldPairs:
        let def = newTree(nnkIdentDefs, ident(name), newEmptyNode(), newLit(value))
        newParams.add(def)
        constr.add(newTree(nnkExprColonExpr, ident(name), ident(name)))
      call.add(constr)
    else:
      newParams.add(param.copyNimTree())
      for it2 in 0..<(param.len - 2):
        call.add(param[it2])
  result.params = newParams
  result.body = newStmtList(call)
  result = newStmtList(procNode, result)

proc rect*(canvas: var Canvas,
           pos, size: Vec2,
           style: ShapeStyle) {.defaultStyle(style, DEFAULT_SHAPE_STYLE).} =
  canvas.shapes.add(Shape(kind: ShapeRect, pos: pos, size: size, style: style))

proc rect*(canvas: var Canvas,
           box: Box2,
           style: ShapeStyle) {.defaultStyle(style, DEFAULT_SHAPE_STYLE).} =
  canvas.shapes.add(Shape(kind: ShapeRect, pos: box.min, size: box.size, style: style))

proc ellipse*(canvas: var Canvas,
              pos, size: Vec2,
              style: ShapeStyle) {.defaultStyle(style, DEFAULT_SHAPE_STYLE).} =
  canvas.shapes.add(Shape(kind: ShapeEllipse, pos: pos, size: size, style: style))

proc line*(canvas: var Canvas,
           start, stop: Vec2,
           style: ShapeStyle) {.defaultStyle(style, DEFAULT_SHAPE_STYLE).} =
  canvas.shapes.add(Shape(kind: ShapeLine, start: start, stop: stop, style: style))

proc path*(canvas: var Canvas,
           path: Path,
           style: ShapeStyle) {.defaultStyle(style, DEFAULT_SHAPE_STYLE).} =
  canvas.shapes.add(Shape(kind: ShapePath, subpaths: @[path], style: style))

type XmlBuilder = object
  data: string

proc emit(builder: var XmlBuilder, text: string) =
  builder.data.add(text)

proc beginTag(builder: var XmlBuilder,
               name: string,
               attrs: openArray[(string, string)]) =
  builder.emit("<")
  builder.emit(name)
  for (name, value) in attrs:
    builder.emit(" ")
    builder.emit(name)
    builder.emit("=\"")
    builder.emit(value)
    builder.emit("\"")
  builder.emit(">")

proc endTag(builder: var XmlBuilder, name: string) =
  builder.emit("</")
  builder.emit(name)
  builder.emit(">")

proc tag(builder: var XmlBuilder,
         name: string,
         attrs: openArray[(string, string)]) =
  builder.beginTag(name, attrs)
  builder.endTag(name)

template tag(builder: var XmlBuilder,
             name: string,
             attrs: openArray[(string, string)],
             body: untyped) =
  block:
    builder.beginTag(name, attrs)
    defer: builder.endTag(name)
    body

proc toSvg(color: Color): string =
  if color == Color():
    result = "none"
  else:
    result = color.toHex()

proc toSvgAttrs(style: ShapeStyle): seq[(string, string)] =
  result = @{
    "fill": style.fill.toSvg(),
    "stroke": style.stroke.toSvg(),
    "stroke-width": $style.strokeWidth
  }

proc genSvg(shape: Shape, builder: var XmlBuilder) =
  let attrs = shape.style.toSvgAttrs()
  case shape.kind:
    of ShapeRect:
      builder.tag("rect", attrs & @{
        "x": $shape.pos.x,
        "y": $shape.pos.y,
        "width": $shape.size.x,
        "height": $shape.size.y
      })
    of ShapeEllipse:
      builder.tag("ellipse", attrs & @{
        "cx": $shape.pos.x,
        "cy": $shape.pos.y,
        "rx": $shape.size.x,
        "ry": $shape.size.y
      })
    of ShapeLine:
      builder.tag("line", attrs & @{
        "x1": $shape.start.x,
        "y1": $shape.start.y,
        "x2": $shape.stop.x,
        "y2": $shape.stop.y
      })
    of ShapePath:
      discard

proc genSvg(canvas: Canvas, builder: var XmlBuilder) =
  let attrs = {
    "xmlns": "http://www.w3.org/2000/svg",
    "width": $canvas.size.x,
    "height": $canvas.size.y,
    "viewBox": "0 0 " & $canvas.size.x & " " & $canvas.size.y
  }
  builder.tag("svg", attrs):
    if canvas.background != Color():
      builder.tag("rect", {
        "x": "0",
        "y": "0",
        "width": $canvas.size.x,
        "height": $canvas.size.y,
        "fill": canvas.background.toSvg()
      })
    for shape in canvas.shapes:
      shape.genSvg(builder)

proc toSvg*(canvas: Canvas): string =
  var builder = XmlBuilder()
  canvas.genSvg(builder)
  result = builder.data

proc saveSvg*(canvas: Canvas, path: string) =
  writeFile(path, canvas.toSvg())
