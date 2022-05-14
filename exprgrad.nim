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
import exprgrad/runtimes/gpu
export tensors
export dsl
export Fun, `++=`, with_shape, copy_shape, lock, make_opt
export param, input, cache, rand, params, grad
export backwards, optimize, backprop, reshape, target, CompileTarget, cond
export Model, compile, call, apply, fit, emit_ir, save_llvm
export ir.`==`, ir.hash, parser.hash
export new_gpu_context, list_devices
