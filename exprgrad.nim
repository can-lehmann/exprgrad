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

# An experimental deep learning framework

import std/[macros]
import exprgrad/[tensors, parser, model, ir, dsl]
export tensors
export dsl
export Fun, `++=`, quote_expr, with_shape, copy_shape, lock, make_opt
export param, input, cache, rand, params, grad
export backwards, optimize, backprop, reshape, target, cond
export Model, compile, call, apply, fit, emit_source
export ir.`==`, ir.hash

when not defined(no_tensor_warning):
  echo ""
  echo "WARNING: The tensor dimension order in exprgrad was changed to be consistent with other tensor libraries."
  echo "WARNING: This means, that the order of the dimensions in tensor shapes and access operations must be"
  echo "WARNING: reversed when porting code from version 0.0.1 to 0.0.2 (this version)."
  echo "WARNING: Matricies are now specified in the order [height, width] and accessed with mat[y, x]."
  echo "WARNING: This warning can be disabled with -d:no_tensor_warning and will also be removed in the future."
