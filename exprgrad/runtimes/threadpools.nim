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

# Thread pool implementation

import osproc

type TaskProc* = proc(model: pointer, a, b: int, data: pointer) {.cdecl, gcsafe.}

type
  Task* = object
    fn*: TaskProc
    model*: pointer
    data*: pointer
    a*: int
    b*: int

  ThreadData = object
    tasks: ptr Channel[Task]
    done: ptr Channel[bool]
  
  ThreadPool* = object
    threads: seq[Thread[ThreadData]]
    data: seq[ThreadData]
    open: seq[int]

proc threadHandler(data: ThreadData) {.gcsafe.} =
  while true:
    let task = data.tasks[].recv()
    try:
      task.fn(task.model, task.a, task.b, task.data)
    except CatchableError as err:
      data.done[].send(false)
      continue
    data.done[].send(true)

proc allocChannel[T](): ptr Channel[T] =
  result = cast[ptr Channel[T]](allocShared0(sizeof(Channel[T])))
  result[].open()

proc initThreadPool(): ThreadPool =
  result.threads = newSeq[Thread[ThreadData]](countProcessors())
  result.open = newSeq[int](result.threads.len)
  for it, thread in result.threads.mpairs:
    let data = ThreadData(
      tasks: allocChannel[Task](),
      done: allocChannel[bool]()
    )
    result.data.add(data)
    createThread(thread, threadHandler, data)
    thread.pinToCpu(it)

proc len*(pool: ThreadPool): int = pool.threads.len

proc enqueue*(pool: var ThreadPool, thread: int, task: Task) =
  pool.data[thread].tasks[].send(task)
  pool.open[thread] += 1

proc join*(pool: var ThreadPool, thread: int) =
  while pool.open[thread] > 0:
    discard pool.data[thread].done[].recv()
    pool.open[thread] -= 1

proc join*(pool: var ThreadPool) =
  for it in 0..<pool.len:
    pool.join(it)

var threadPool* = initThreadPool()
