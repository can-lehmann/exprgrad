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

import opencl except check

type
  GpuError* = ref object of CatchableError
  
  GpuDevice* = object
    id: PDeviceId
  
  GpuContext* = ref object
    device: GpuDevice
    context: PContext
    commands: PCommandQueue
  
  GpuBuffer* = object
    ctx: GpuContext
    size: int
    mem: PMem
  
  GpuKernelSource* = object
    name*: string
    source*: string
  
  GpuKernel* = ref object
    ctx: GpuContext
    kernel: PKernel

proc check(res: TClResult) =
  if res != SUCCESS:
    raise GpuError(msg: $res)

proc list_platforms(): seq[PPlatformId] =
  var platform_count: uint32
  check get_platform_ids(0, nil, platform_count.addr)
  if platform_count > 0:
    result = new_seq[PPlatformId](platform_count.int)
    check get_platform_ids(platform_count, result[0].addr, nil)

proc list_devices(platform: PPlatformId, typ: TDeviceType = DEVICE_TYPE_ALL): seq[GpuDevice] =
  var device_count: uint32
  let res = get_device_ids(platform, typ, 0, nil, device_count.addr)
  if res == DEVICE_NOT_FOUND:
    return
  check res
  if device_count > 0:
    var ids = new_seq[PDeviceId](device_count.int)
    check get_device_ids(platform, typ, device_count, ids[0].addr, nil)
    for id in ids:
      result.add(GpuDevice(id: id))

proc list_devices*(): seq[GpuDevice] =
  for platform in list_platforms():
    result.add(platform.list_devices())

proc query_string(device: GpuDevice, info: TDeviceInfo): string =
  var size: int
  check get_device_info(device.id, info, 0, nil, size.addr)
  if size > 0:
    result = new_string(size)
    check get_device_info(device.id, info, size, result[0].addr, nil)

proc name*(device: GpuDevice): string = device.query_string(DEVICE_NAME)
proc vendor*(device: GpuDevice): string = device.query_string(DEVICE_VENDOR)
proc version*(device: GpuDevice): string = device.query_string(DEVICE_VERSION)
proc is_gpu*(device: GpuDevice): bool =
  var device_type: TDeviceType
  check get_device_info(device.id, DEVICE_TYPE, sizeof(device_type), device_type.addr, nil)
  result = (device_type.int64 and DEVICE_TYPE_GPU.int64) != 0

proc new_gpu_context*(device: GpuDevice): GpuContext =
  result = GpuContext(device: device)
  
  var
    id = device.id
    status: TClResult
  result.context = create_context(nil, 1, id.addr, nil, nil, status.addr)
  check status
  
  result.commands = create_command_queue(result.context, id, 0, status.addr)
  check status

proc new_gpu_context*(): GpuContext =
  let devices = list_devices()
  if devices.len == 0:
    raise GpuError(msg: "Unable to find device")
  result = new_gpu_context(devices[0])

proc alloc_buffer*(ctx: GpuContext, size: int): GpuBuffer =
  var status: TClResult
  result.ctx = ctx
  result.size = size
  result.mem = create_buffer(ctx.context, MEM_READ_WRITE, size, nil, status.addr)
  check status

proc dealloc*(buffer: GpuBuffer) =
  check release_mem_object(buffer.mem)

proc write*(buffer: GpuBuffer, data: pointer, size: int) =
  if size != buffer.size:
    raise GpuError(msg: "Attempted to write " & $size & " bytes, but the size of the buffer is " & $buffer.size & " bytes")
  check buffer.ctx.commands.enqueue_write_buffer(
    buffer.mem, CL_TRUE, 0, size, data, 0, nil, nil
  )

proc write*[T](buffer: GpuBuffer, data: openArray[T]) =
  if data.len > 0:
    buffer.write(data[0].unsafe_addr, sizeof(T) * data.len)

proc fill*[T](buffer: GpuBuffer, value: T) =
  var val = value
  check buffer.ctx.commands.enqueue_fill_buffer(
    buffer.mem, val.addr, sizeof(T), 0, buffer.size, 0, nil, nil
  )

proc read_into*[T](buffer: GpuBuffer, data: ptr UncheckedArray[T]) =
  check buffer.ctx.commands.enqueue_read_buffer(
    buffer.mem, CL_TRUE, 0, buffer.size, data[0].addr, 0, nil, nil
  )

proc read_into*[T](buffer: GpuBuffer, data: var seq[T]) =
  if buffer.size != data.len * sizeof(T):
    raise GpuError(msg: "Buffer size is not equal to target size")
  check buffer.ctx.commands.enqueue_read_buffer(
    buffer.mem, CL_TRUE, 0, buffer.size, data[0].addr, 0, nil, nil
  )

proc read*[T](buffer: GpuBuffer): seq[T] =
  if buffer.size mod sizeof(T) != 0:
    raise GpuError(msg: "Buffer size is not divisible by item type size")
  if buffer.size > 0:
    result = new_seq[T](buffer.size div sizeof(T))
    check buffer.ctx.commands.enqueue_read_buffer(
      buffer.mem, CL_TRUE, 0, buffer.size, result[0].addr, 0, nil, nil
    )

proc compile*(ctx: GpuContext, name: string, source: string): GpuKernel =
  result = GpuKernel(ctx: ctx)
  
  let strings = alloc_cstring_array([source])
  defer: dealloc_cstring_array(strings)
  
  var
    length = source.len
    status: TClResult
  let program = ctx.context.create_program_with_source(1, strings, length.addr, status.addr)
  check status
  
  status = build_program(program, 1, ctx.device.id.addr, nil, nil, nil)
  
  if status == BUILD_PROGRAM_FAILURE:
    var log_length: int
    check get_program_build_info(program, ctx.device.id, PROGRAM_BUILD_LOG, 0, nil, log_length.addr)
    if log_length > 0:
      var log = new_string(log_length)
      check get_program_build_info(program, ctx.device.id, PROGRAM_BUILD_LOG, log_length, log[0].addr, nil)
      raise GpuError(msg: "Failed to build program: " & log)
    else:
      raise GpuError(msg: "Failed to build program")
  else:
    check status
  
  result.kernel = create_kernel(program, name.cstring, status.addr)
  check status

proc compile*(ctx: GpuContext, source: GpuKernelSource): GpuKernel =
  result = ctx.compile(source.name, source.source)

proc arg*[T](kernel: GpuKernel, index: int, value: T): GpuKernel =
  result = kernel
  var data = value
  check set_kernel_arg(kernel.kernel, uint32(index), sizeof(T), data.addr)

proc arg*(kernel: GpuKernel, index: int, buffer: GpuBuffer): GpuKernel =
  result = kernel
  check set_kernel_arg(kernel.kernel, uint32(index), sizeof(Pmem), buffer.mem.unsafe_addr)

proc run*(kernel: GpuKernel, group_size, local_size: openArray[int]) =
  if group_size.len == 0:
    raise GpuError(msg: "Group size must have at least one dimension")
  if group_size.len != local_size.len:
    raise GpuError(msg: "Dimension of group size must equal dimension of local size")
  
  var global_size = new_seq[int](group_size.len)
  for it, local in local_size:
    global_size[it] = local * group_size[it]
  check enqueue_nd_range_kernel(
    kernel.ctx.commands,
    kernel.kernel,
    uint32(global_size.len),
    nil,
    global_size[0].unsafe_addr,
    local_size[0].unsafe_addr,
    0, nil, nil
  )

when is_main_module:
  for device in list_devices():
    echo "Name: ", device.name
    echo "Vendor: ", device.vendor
    echo "OpenCL Version: ", device.version
    echo "Is GPU: ", device.is_gpu
  
  let source = """
    __kernel void add(__global float* a, __global float* b, __global float* c) {
      int index = get_global_id(0);
      c[index] = a[index] + b[index];
    }
  """
  
  let ctx = new_gpu_context()
  
  let
    a = ctx.alloc_buffer(sizeof(float32) * 32)
    b = ctx.alloc_buffer(sizeof(float32) * 32)
    c = ctx.alloc_buffer(sizeof(float32) * 32)
    data = block:
      var data = new_seq[float32](32)
      for it, item in data.mpairs:
        item = float32(it)
      data
  a.write(data)
  b.write(data)
  
  let kernel = ctx.compile("add", source)
  kernel
    .arg(0, a)
    .arg(1, b)
    .arg(2, c)
    .run([1], [32])
  
  echo read[float32](c)
  
