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
import canvas, geometry

type
  Figure* = ref object of RootObj
  
  GridFigure = object
    pos: Index2
    size: Index2
    figure: Figure
  
  GridLayout* = ref object of Figure
    padding: Vec2
    spacing: Vec2
    cell_counts: Index2
    figures: seq[GridFigure]

method min_size*(figure: Figure): Vec2 {.base.} = discard
method draw*(figure: Figure, rect: Box2, canvas: var Canvas) {.base.} = discard

proc pack*(layout: GridLayout, pos, size: Index2, figure: Figure) =
  layout.figures.add(GridFigure(figure: figure, pos: pos, size: size))
  layout.cell_counts = max(layout.cell_counts, pos + size)

proc pack*(layout: GridLayout, pos: Index2, figure: Figure) =
  layout.pack(pos, Index2(x: 1, y: 1), figure)

proc min_cell_sizes(layout: GridLayout, axis: Axis): seq[float64] =
  var order = to_seq(0..<layout.figures.len)
  order.sort((a, b) => cmp(layout.figures[a].size[axis], layout.figures[b].size[axis]))
  
  result = new_seq[float64](layout.cell_counts[axis])
  for index in order:
    let
      figure = layout.figures[index]
      size = figure.figure.min_size()[axis]
    
    let current_size = block:
      var size = 0.0
      for offset in 0..<figure.size[axis]:
        let cell = offset + figure.pos[axis]
        size += result[cell]
      size += float64(figure.size[axis] - 1) * layout.spacing[axis]
      size
    
    let delta = size - current_size
    if delta > 0:
      let grow_by = delta / float64(figure.size[axis])
      for offset in 0..<figure.size[axis]:
        result[figure.pos[axis] + offset] += grow_by

proc arrange_axis(layout: GridLayout, axis: Axis, into: Inter): seq[Inter] =
  var cells = layout.min_cell_sizes(axis)
   
  let
    used = cells.sum() + layout.spacing[axis] * float64(cells.len - 1) + layout.padding[axis] * 2
    delta = into.size - used
  if delta > 0:
    for cell in cells.mitems:
      cell += delta / float64(cells.len)
  
  var
    offsets = new_seq[float64](layout.cell_counts[axis] + 1)
    offset = layout.padding[axis] + into.min
  for cell_id, size in cells:
    offsets[cell_id] = offset
    offset += size + layout.spacing[axis]
  offsets[^1] = offset
  
  result = new_seq[Inter](layout.figures.len)
  for it, figure in layout.figures:
    result[it] = Inter(
      min: offsets[figure.pos[axis]],
      max: offsets[figure.pos[axis] + figure.size[axis]] - layout.spacing[axis]
    )

proc arrange(layout: GridLayout, box: Box2): seq[Box2] =
  let
    x_inters = layout.arrange_axis(AxisX, box.x_inter)
    y_inters = layout.arrange_axis(AxisY, box.y_inter)
  result = new_seq[Box2](layout.figures.len)
  for it, box in result.mpairs:
    box.min.x = x_inters[it].min
    box.min.y = y_inters[it].min
    box.max.x = x_inters[it].max
    box.max.y = y_inters[it].max

method min_size*(layout: GridLayout): Vec2 =
  result = Vec2(
    x: layout.min_cell_sizes(AxisX).sum(),
    y: layout.min_cell_sizes(AxisY).sum()
  )
  result += to_vec2(layout.cell_counts - Index2(x: 1, y: 1)) * layout.spacing
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

method min_size(spacer: Spacer): Vec2 = spacer.size
method draw(spacer: Spacer, box: Box2, canvas: var Canvas) =
  if spacer.color != Color():
    canvas.rect(box, fill = spacer.color, stroke = Color())
