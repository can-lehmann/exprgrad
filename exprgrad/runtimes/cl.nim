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

import opencl

type
  GpuDevice* = object
    id: PDeviceId
  
  GpuContext* = ref object
    context: PContext
    commands: PCommandQueue

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
  discard

proc new_gpu_context*(): GpuContext =
  result = new_gpu_context(list_devices()[0])

when is_main_module:
  for device in list_devices():
    echo "Name: ", device.name
    echo "Vendor: ", device.vendor
    echo "OpenCL Version: ", device.version
    echo "Is GPU: ", device.is_gpu
