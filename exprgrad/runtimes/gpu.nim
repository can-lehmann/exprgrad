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

# Common interface for interacting with the GPU

when defined(opencl):
  import cl
  export cl
else:
  type
    GpuError* = ref object of CatchableError
    GpuDevice* = object
    GpuContext* = ref object
    GpuBuffer* = object
    GpuKernelSource* = object
    GpuKernel* = ref object
  
  const MESSAGE = "Compile with -d:opencl to enable GPU support"
  
  proc list_devices*(): seq[GpuDevice] = raise GpuError(msg: MESSAGE)
  proc name*(device: GpuDevice): string = raise GpuError(msg: MESSAGE)
  proc vendor*(device: GpuDevice): string = raise GpuError(msg: MESSAGE)
  proc version*(device: GpuDevice): string = raise GpuError(msg: MESSAGE)
  proc is_gpu*(device: GpuDevice): bool = raise GpuError(msg: MESSAGE)
  proc new_gpu_context*(device: GpuDevice): GpuContext = raise GpuError(msg: MESSAGE)
  proc new_gpu_context*(): GpuContext = raise GpuError(msg: MESSAGE)
  proc alloc_buffer*(ctx: GpuContext, size: int): GpuBuffer = raise GpuError(msg: MESSAGE)
  proc write*(buffer: GpuBuffer, data: pointer, size: int) = raise GpuError(msg: MESSAGE)
  proc write*[T](buffer: GpuBuffer, data: openArray[T]) = raise GpuError(msg: MESSAGE)
  proc read*[T](buffer: GpuBuffer): seq[T] = raise GpuError(msg: MESSAGE)
  proc compile*(ctx: GpuContext, name, source: string): GpuKernel = raise GpuError(msg: MESSAGE)
  proc compile*(ctx: GpuContext, source: GpuKernelSource): GpuKernel = raise GpuError(msg: MESSAGE)
  proc arg*[T](kernel: GpuKernel, index: int, value: T): GpuKernel = raise GpuError(msg: MESSAGE)
  proc arg*(kernel: GpuKernel, index: int, buffer: GpuBuffer): GpuKernel = raise GpuError(msg: MESSAGE)
  proc run*(kernel: GpuKernel, global_size, local_size: openArray[int]) = raise GpuError(msg: MESSAGE)

type GpuTensor*[T] = ref object
  shape*: seq[int]
  buffer*: GpuBuffer

