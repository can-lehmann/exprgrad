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
    id: Pdevice_id
  
  GpuContext* = ref object
    device: GpuDevice
    context: Pcontext
    commands: Pcommand_queue
  
  GpuBuffer* = object
    ctx: GpuContext
    size: int
    mem: Pmem
  
  GpuKernelSource* = object
    name*: string
    source*: string
  
  GpuKernel* = ref object
    ctx: GpuContext
    kernel: Pkernel

proc check(res: TClResult) =
  if res != SUCCESS:
    raise GpuError(msg: $res)

proc listPlatforms(): seq[Pplatform_id] =
  var platformCount: uint32
  check getPlatformIDs(0, nil, platformCount.addr)
  if platformCount > 0:
    result = newSeq[Pplatform_id](platformCount.int)
    check getPlatformIDs(platformCount, result[0].addr, nil)

proc listDevices(platform: Pplatform_id, typ: TDeviceType = DEVICE_TYPE_ALL): seq[GpuDevice] =
  var deviceCount: uint32
  let res = getDeviceIDs(platform, typ, 0, nil, deviceCount.addr)
  if res == DEVICE_NOT_FOUND:
    return
  check res
  if deviceCount > 0:
    var ids = newSeq[Pdevice_id](deviceCount.int)
    check getDeviceIDs(platform, typ, deviceCount, ids[0].addr, nil)
    for id in ids:
      result.add(GpuDevice(id: id))

proc listDevices*(): seq[GpuDevice] =
  for platform in listPlatforms():
    result.add(platform.listDevices())

proc queryString(device: GpuDevice, info: Tdevice_info): string =
  var size: int
  check getDeviceInfo(device.id, info, 0, nil, size.addr)
  if size > 0:
    result = newString(size)
    check getDeviceInfo(device.id, info, size, result[0].addr, nil)

proc name*(device: GpuDevice): string = device.queryString(DEVICE_NAME)
proc vendor*(device: GpuDevice): string = device.queryString(DEVICE_VENDOR)
proc version*(device: GpuDevice): string = device.queryString(DEVICE_VERSION)
proc isGpu*(device: GpuDevice): bool =
  var deviceType: TDeviceType
  check getDeviceInfo(device.id, DEVICE_TYPE, sizeof(deviceType), deviceType.addr, nil)
  result = (deviceType.int64 and DEVICE_TYPE_GPU.int64) != 0

proc newGpuContext*(device: GpuDevice): GpuContext =
  result = GpuContext(device: device)
  
  var
    id = device.id
    status: TClResult
  result.context = createContext(nil, 1, id.addr, nil, nil, status.addr)
  check status
  
  result.commands = createCommandQueue(result.context, id, 0, status.addr)
  check status

proc newGpuContext*(): GpuContext =
  let devices = listDevices()
  if devices.len == 0:
    raise GpuError(msg: "Unable to find device")
  result = newGpuContext(devices[0])

proc allocBuffer*(ctx: GpuContext, size: int): GpuBuffer =
  var status: TClResult
  result.ctx = ctx
  result.size = size
  result.mem = createBuffer(ctx.context, MEM_READ_WRITE, size, nil, status.addr)
  check status

proc dealloc*(buffer: GpuBuffer) =
  check releaseMemObject(buffer.mem)

proc write*(buffer: GpuBuffer, data: pointer, size: int) =
  if size != buffer.size:
    raise GpuError(msg: "Attempted to write " & $size & " bytes, but the size of the buffer is " & $buffer.size & " bytes")
  check buffer.ctx.commands.enqueueWriteBuffer(
    buffer.mem, CL_TRUE, 0, size, data, 0, nil, nil
  )

proc write*[T](buffer: GpuBuffer, data: openArray[T]) =
  if data.len > 0:
    buffer.write(data[0].unsafeAddr, sizeof(T) * data.len)

proc fill*[T](buffer: GpuBuffer, value: T) =
  var val = value
  check buffer.ctx.commands.enqueueFillBuffer(
    buffer.mem, val.addr, sizeof(T), 0, buffer.size, 0, nil, nil
  )

proc readInto*[T](buffer: GpuBuffer, data: ptr UncheckedArray[T]) =
  check buffer.ctx.commands.enqueueReadBuffer(
    buffer.mem, CL_TRUE, 0, buffer.size, data[0].addr, 0, nil, nil
  )

proc readInto*[T](buffer: GpuBuffer, data: var seq[T]) =
  if buffer.size != data.len * sizeof(T):
    raise GpuError(msg: "Buffer size is not equal to target size")
  check buffer.ctx.commands.enqueueReadBuffer(
    buffer.mem, CL_TRUE, 0, buffer.size, data[0].addr, 0, nil, nil
  )

proc read*[T](buffer: GpuBuffer): seq[T] =
  if buffer.size mod sizeof(T) != 0:
    raise GpuError(msg: "Buffer size is not divisible by item type size")
  if buffer.size > 0:
    result = newSeq[T](buffer.size div sizeof(T))
    check buffer.ctx.commands.enqueueReadBuffer(
      buffer.mem, CL_TRUE, 0, buffer.size, result[0].addr, 0, nil, nil
    )

proc compile*(ctx: GpuContext, name: string, source: string): GpuKernel =
  result = GpuKernel(ctx: ctx)
  
  let strings = allocCStringArray([source])
  defer: deallocCStringArray(strings)
  
  var
    length = source.len
    status: TClResult
  let program = ctx.context.createProgramWithSource(1, strings, length.addr, status.addr)
  check status
  
  status = buildProgram(program, 1, ctx.device.id.addr, nil, nil, nil)
  
  if status == BUILD_PROGRAM_FAILURE:
    var logLength: int
    check getProgramBuildInfo(program, ctx.device.id, PROGRAM_BUILD_LOG, 0, nil, logLength.addr)
    if logLength > 0:
      var log = newString(logLength)
      check getProgramBuildInfo(program, ctx.device.id, PROGRAM_BUILD_LOG, logLength, log[0].addr, nil)
      raise GpuError(msg: "Failed to build program: " & log)
    else:
      raise GpuError(msg: "Failed to build program")
  else:
    check status
  
  result.kernel = createKernel(program, name.cstring, status.addr)
  check status

proc compile*(ctx: GpuContext, source: GpuKernelSource): GpuKernel =
  result = ctx.compile(source.name, source.source)

proc arg*[T](kernel: GpuKernel, index: int, value: T): GpuKernel =
  result = kernel
  var data = value
  check setKernelArg(kernel.kernel, uint32(index), sizeof(T), data.addr)

proc arg*(kernel: GpuKernel, index: int, buffer: GpuBuffer): GpuKernel =
  result = kernel
  check setKernelArg(kernel.kernel, uint32(index), sizeof(Pmem), buffer.mem.unsafe_addr)

proc run*(kernel: GpuKernel, groupSize, localSize: openArray[int]) =
  if groupSize.len == 0:
    raise GpuError(msg: "Group size must have at least one dimension")
  if groupSize.len != localSize.len:
    raise GpuError(msg: "Dimension of group size must equal dimension of local size")
  
  var globalSize = newSeq[int](groupSize.len)
  for it, local in localSize:
    globalSize[it] = local * groupSize[it]
  check enqueueNDRangeKernel(
    kernel.ctx.commands,
    kernel.kernel,
    uint32(globalSize.len),
    nil,
    globalSize[0].unsafeAddr,
    localSize[0].unsafeAddr,
    0, nil, nil
  )

when isMainModule:
  for device in listDevices():
    echo "Name: ", device.name
    echo "Vendor: ", device.vendor
    echo "OpenCL Version: ", device.version
    echo "Is GPU: ", device.isGpu
  
  let source = """
    __kernel void add(__global float* a, __global float* b, __global float* c) {
      int index = get_global_id(0);
      c[index] = a[index] + b[index];
    }
  """
  
  let ctx = newGpuContext()
  
  let
    a = ctx.allocBuffer(sizeof(float32) * 32)
    b = ctx.allocBuffer(sizeof(float32) * 32)
    c = ctx.allocBuffer(sizeof(float32) * 32)
    data = block:
      var data = newSeq[float32](32)
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
  
