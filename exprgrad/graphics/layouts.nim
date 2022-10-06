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

# Grid layout

import std/[sequtils, algorithm, sugar, math]
import canvas, geometrymath

type
  Figure* = ref object of RootObj
  
  GridFigure = object
    pos: Index2
    size: Index2
    figure: Figure
  
  GridLayout* = ref object of Figure
    padding: Vec2
    spacing: Vec2
    cellCounts: Index2
    figures: seq[GridFigure]

method minSize*(figure: Figure): Vec2 {.base.} = discard
method draw*(figure: Figure, rect: Box2, canvas: var Canvas) {.base.} = discard

proc pack*(layout: GridLayout, pos, size: Index2, figure: Figure) =
  layout.figures.add(GridFigure(figure: figure, pos: pos, size: size))
  layout.cellCounts = max(layout.cellCounts, pos + size)

proc pack*(layout: GridLayout, pos: Index2, figure: Figure) =
  layout.pack(pos, Index2(x: 1, y: 1), figure)

proc minCellSizes(layout: GridLayout, axis: Axis): seq[float64] =
  var order = toSeq(0..<layout.figures.len)
  order.sort((a, b) => cmp(layout.figures[a].size[axis], layout.figures[b].size[axis]))
  
  result = newSeq[float64](layout.cellCounts[axis])
  for index in order:
    let
      figure = layout.figures[index]
      size = figure.figure.minSize()[axis]
    
    let currentSize = block:
      var size = 0.0
      for offset in 0..<figure.size[axis]:
        let cell = offset + figure.pos[axis]
        size += result[cell]
      size += float64(figure.size[axis] - 1) * layout.spacing[axis]
      size
    
    let delta = size - currentSize
    if delta > 0:
      let growBy = delta / float64(figure.size[axis])
      for offset in 0..<figure.size[axis]:
        result[figure.pos[axis] + offset] += growBy

proc arrangeAxis(layout: GridLayout, axis: Axis, into: Inter): seq[Inter] =
  var cells = layout.minCellSizes(axis)
   
  let
    used = cells.sum() + layout.spacing[axis] * float64(cells.len - 1) + layout.padding[axis] * 2
    delta = into.size - used
  if delta > 0:
    for cell in cells.mitems:
      cell += delta / float64(cells.len)
  
  var
    offsets = newSeq[float64](layout.cellCounts[axis] + 1)
    offset = layout.padding[axis] + into.min
  for cellId, size in cells:
    offsets[cellId] = offset
    offset += size + layout.spacing[axis]
  offsets[^1] = offset
  
  result = newSeq[Inter](layout.figures.len)
  for it, figure in layout.figures:
    result[it] = Inter(
      min: offsets[figure.pos[axis]],
      max: offsets[figure.pos[axis] + figure.size[axis]] - layout.spacing[axis]
    )

proc arrange(layout: GridLayout, box: Box2): seq[Box2] =
  let
    xInters = layout.arrangeAxis(AxisX, box.xInter)
    yInters = layout.arrangeAxis(AxisY, box.yInter)
  result = newSeq[Box2](layout.figures.len)
  for it, box in result.mpairs:
    box.min.x = xInters[it].min
    box.min.y = yInters[it].min
    box.max.x = xInters[it].max
    box.max.y = yInters[it].max

method minSize*(layout: GridLayout): Vec2 =
  result = Vec2(
    x: layout.minCellSizes(AxisX).sum(),
    y: layout.minCellSizes(AxisY).sum()
  )
  result += toVec2(layout.cellCounts - Index2(x: 1, y: 1)) * layout.spacing
  result += 2.0 * layout.padding

method draw*(layout: GridLayout, box: Box2, canvas: var Canvas) =
  let boxes = layout.arrange(box)
  for it, figure in layout.figures:
    figure.figure.draw(boxes[it], canvas)

proc new*(_: typedesc[GridLayout],
          spacing: Vec2 = Vec2(x: 6, y: 6),
          padding: Vec2 = Vec2(x: 12, y: 12)): GridLayout =
  result = GridLayout(spacing: spacing, padding: padding)

type Spacer* = ref object of Figure
  color: Color
  size: Vec2

proc new*(_: typedesc[Spacer],
          size: Vec2 = Vec2(x: 24, y: 24),
          color: Color = Color()): Spacer =
  result = Spacer(size: size, color: color)

method minSize(spacer: Spacer): Vec2 = spacer.size
method draw(spacer: Spacer, box: Box2, canvas: var Canvas) =
  if spacer.color != Color():
    canvas.rect(box, fill = spacer.color, stroke = Color())
