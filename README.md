# Exprgrad

Exprgrad is an experimental deep learning framework for Nim based on a differentiable array programming language.
Exprgrad makes creating and training neural networks easy: 

```nim
import std/random
import exprgrad, exprgrad/layers/[base, dnn]
randomize(10)

let
  net = input("x")
    .dense(2, 4).leakyRelu()  # 1st Layer
    .dense(4, 1).sigmoid()    # 2nd Layer
    .target("predict")
    .mse(input("y"))          # Loss
    .target("loss")
    .backprop(gradientDescent.makeOpt(rate=0.1)) # Train
    .target("train")
  model = compile[float32](net)

let
  trainX = Tensor.new([4, 2], @[float32 0, 0, 0, 1, 1, 0, 1, 1])
  trainY = Tensor.new([4, 1], @[float32 0, 1, 1, 0])

for epoch in 0..<5000:
  model.apply("train", {"x": trainX, "y": trainY})

echo model.call("predict", {"x": trainX})
```

Because exprgrad is based on a custom differentiable programming language, we do not need to rely on its built in layers.
Instead we can also specify the same model in terms of scalar operations on tensors.

```nim
# Layer 1
hidden*[y, x] ++= input("x")[y, it] * param([2, 4])[it, x] | (y, x, it)
hidden[y, x] ++= param([4])[x] | (y, x)
hiddenRelu*{it} ++= select(hidden{it} <= 0.0, 0.1 * hidden{it}, hidden{it}) | it
# Layer 2
output*[y, x] ++= hiddenRelu[y, it] * param([4, 1])[it, x] | (y, x, it)
output[y, x] ++= param([1])[x] | (y, x)
outputSigmoid*{it} ++= 1.0 / (1.0 + exp(-output{it})) | it
let pred = outputSigmoid.target("predict")
loss*[0] ++= sq(pred{it} - input("y"){it}) | it # Loss

proc optim(param: var Fun, grad: Fun) =
  param{it} ++= -0.1 * grad{it} | it

let net = loss.target("loss").backprop(optim).target("train") # Train

let model = compile[float32](net)
```

Since exprgrad's compiler is able to derive any program written in its domain specific language, we do not need to specify a backwards pass.
This allows you to iterate on custom layers quickly, while avoiding errors in the gradient computation.
The model is optimized and compiled using a JIT compiler, enabling fast execution times.
All layers provided by exprgrad are also implemented in the same way, allowing you to customize them easily.

## Installation

**Warning:** Exprgrad is still very early in its development.
Although all shown examples already work, bugs are expected and important features for training large models (especially Multithreading and GPU support) might still be missing.
Please report any issues you might encounter.

### Ubuntu

```bash
$ sudo apt install llvm-13-dev
$ nimble install exprgrad
```

**Note:** Your version of Ubuntu may not have the `llvm-13-dev` package in its repositories.
Follow the instructions at [apt.llvm.org](https://apt.llvm.org/) to install the required repository.

### Fedora 36

```bash
$ sudo dnf install llvm13-devel
$ nimble install exprgrad
```

### Fedora 35

```bash
$ sudo dnf install llvm-devel
$ nimble install exprgrad
```

## Documentation

### Language

Exprgrad's custom differentiable array programming language is used to specify all layers.
It is a custom language which differs greatly from Nim both in syntax and semantics.
Kernels/layers written in exprgrad's language are embedded in Nim programs and created using the `++=` macro.

The language does not have functions, procedures or structured control flow.
Instead each program is a single expression inside a series of implicitly specified nested loops.
A simple program which multiplies two matrices looks like this:

```nim
proc matmul(a, b: Fun): Fun =
  result[y, x] ++= a[y, it] * b[it, x] | (y, x, it)
```

The same program in Nim would look like this:

```nim
proc `*`*[T](a, b: Tensor[T]): Tensor[T] =
  result = Tensor[T].new([a.shape[0], b.shape[1]])
  for y in 0..<result.shape[0]:
    for it in 0..<a.shape[1]:
      for x in 0..<result.shape[1]:
        result[y, x] += a[y, it] * b[it, x]
```

As you can see, the program in exprgrad's domain-specific language is basically equivalent to the last line of the Nim program.
The shape of the output tensor and the iteration ranges of all loops are inferred automatically.

In contrast to Nim, exprgrad's type system is very simple as it includes only four types.

| Name       | Purpose                                            |
| ---------- | -------------------------------------------------- |
| `Scalar`   | Floating point value. Is differentiable.           |
| `Index`    | Integer value. Used to index into tensors.         |
| `Boolean`  | Boolean value. Only used in `select` instructions. |
| `Array[T]` | Fixed size array with items of type T.             |

Tensors may be accessed using the `[]` and `{}` operators.
While `[]` allows you index into each dimension, `{}` gives you direct access to the data of the tensor.
Because `{}` does not allow exprgrad to infer tensor shapes in all cases, `[]` should always be preferred over `{}`. 

```nim
proc identity*(a: Fun): Fun =
  result{it} ++= a{it} | it
```

Literals for each type are available.
Note that exprgrad does not have automatic type conversions.
`Scalar` literals therefore must include a point (`2.0` instead of `2`) to differentiate them from `Index` literals.

```nim
proc double*(a: Fun): Fun =
  result{it} ++= a{it} * 2.0 | it
```

Variables from Nim may be included as static values.
Only variables of type `int`, `float64` and `bool` can be included.

```nim
proc `*`*(a: Fun, factor: float64): Fun =
  result{it} ++= a{it} * factor | it
```

Conditionals can be emulated using the `select` instruction.
There is no guarantee that both branches are executed.

```nim
proc relu*(inp: Fun): Fun =
  result{it} ++= select(inp{it} >= 0.0, inp{it}, 0.0) | it
```

An expression may contain multiple statements separated using `;`.
This allows you to define variables using the `let` statement and use them later on.

```nim
proc tanh*(inp: Fun): Fun =
  result{it} ++= (
    let a = exp(inp{it});
    let b = exp(-inp{it});
    (a - b) / (a + b)
  ) | it
```

If exprgrad is not able to infer the shape of a tensor, it can be explicitly specified using `withShape` or `copyShape`.
The argument to the `withShape` macro must be of the form `[dim0, dim1, dim2, ...]` where each dimension is a valid expression in exprgrad's language.

```nim
proc upsample2*(images: Fun): Fun =
  result[image, y, x, chan] ++= images[image, y div 2, x div 2, chan] | (image, y, x, chan)
  result.withShape([
    images.shape[0],
    images.shape[1] * 2,
    images.shape[2] * 2,
    images.shape[3]
  ])
```

If the output tensor is not yet declared, the `*` operator can be added after the tensor's name to declare it.

```nim
y*{it} ++= input("x"){it} * 2.0 | it
```

Sometimes you might want to use a custom gradient implementation instead of the one automatically generated by exprgrad.
This is especially useful for ensuring numerical stability or improving performance.
Inside the `customGrad` attribute, gradient tensors are referred to using the `grad(tensor)` instruction.

```nim
identity*{x} ++= inp{x} | x do:
  customGrad:
    grad(inp){x} ++= inp{x} * grad(identity){x} | x
```

More examples can be found in the `exprgrad/layers/base.nim` and `exprgrad/layers/dnn.nim` modules.

#### Instructions

In addition to the basic operators `+`, `-`, `*`, `/`, `div`, `mod`, `==`, `<`, `>`, `<=` and `>=`, the following instructions are supported:

| Instruction          | Description                                                       | 
| -------------------- | ----------------------------------------------------------------- |
| `sq(x)`              | Computes the square of `x`                                        |
| `min(a, b)`          | Returns the minimum of `a` and `b`                                |
| `max(a, b)`          | Returns the maximum of `a` and `b`                                |
| `select(cond, a, b)` | Returns `a` if `cond` is true else returns `b`                    |
| `sin(x)`             | Returns the sine of `x`                                           |
| `cos(x)`             | Returns the cosine of `x`                                         |
| `exp(x)`             | Computes `e ^ x`                                                  |
| `pow(a, b)`          | Computes `a ^ b`                                                  |
| `sqrt(x)`            | Computes the square root of `x`                                   |
| `ln(x)`              | Computes the natural logarithm of `x`                             |
| `log2(x)`            | Computes the logarithm of base 2 of `x`                           |
| `log10(x)`           | Computes the logarithm of base 10 of `x`                          |
| `wrap(x, y)`         | Computes `(x mod y + y) mod y` (`∈ [0, y) ∩ ℤ`)                   |
| `toScalar(x)`        | Converts `x` to a `Scalar` value                                  |
| `toIndex(x)`         | Converts `x` to an `Index` value                                  |
| `tensor.shape[dim]`  | Returns the size of dimension `dim` of `tensor`                   |
| `tensor.len`         | Returns the number of items in `tensor`                           |
| `tensor.shape.len`   | Returns the rank of `tensor`                                      |
| `epoch()`            | Returns the current epoch stored in `Model.epoch`.                |
| `arr.len`            | Returns the length of the given array.                            |
| `arr[index]`         | Gets the element stored at `index` in the array.                  |

If you cannot find the instruction you are looking for, please open an issue.

### Computation Graphs

Neural networks are represented as computation graphs.
Each computation graph has a set of inputs which are provided to it at run time.
They may be images the model is supposed to classify or a text whose sentiment it is supposed to predict.
Each neural network also has a set of parameters.
These are the internal values which are learned during backpropagation.
Exprgrad refers to the output of a given computation as a target.
A target might be the actual output of the network itself, but also the loss with respect to a training dataset or the action of updating the parameters of the network using gradient descent.
In order to compute the value of a target, a series of kernels (layers) is executed.
Additionally a computation graph may include a set of caches used to save the internal state of an optimizer and randomized tensors used as inputs to dropout layers.

```nim
proc param*(shape: openArray[int],
            initRange: HSlice[float64, float64] = -0.1..0.1,
            name: string = ""): Fun
```

Creates a new parameter with the given shape.
Each parameter is randomly initialized with a uniform distribution in the range `initRange` after model compilation.

```nim
proc input*(name: string, shape: openArray[int] = []): Fun
```

Creates a new input with the given name.
The sizes of static dimensions may be specified to enable compiler optimizations.
If a shape is specified unknown dimensions should have the size `-1`.

Example: `input("x", [-1, 28, 28, 1])`

```nim
proc target*(fun: Fun, name: string): Fun
```

Creates a new target with the given name.
Targets may be called using the `Model.call`, `Model.apply` or `Model.fit` procedures.

```nim
proc backwards*(fun: Fun): Fun
```

Lazily computes the gradients for all parameters of the given computation graph (`fun`) with respect to the given loss value `fun`.
Unused gradients are not computed.

```nim
proc optimize*(gradients: Fun,
               params: HashSet[Fun],
               optim: proc (param: var Fun, grad: Fun)): Fun
proc optimize*(gradients: Fun, optim: proc (param: var Fun, grad: Fun)): Fun
```

Optimizes the given parameters using the given optimizer.
Optimizers may be created using `makeOpt`.
The `Fun.params` procedure may be used to find all parameters of a computation graph.

```nim
proc backprop*(loss: Fun, optim: proc (param: var Fun, grad: Fun)): Fun
```

Computes the gradients for all parameters of `loss` and optimizes them using the given optimizer.
Optimizers may be created using `makeOpt`.
Shortcut for `loss.backwards().optimize(optim)`.

```nim
proc reshape*(fun: Fun, shape: openArray[int]): Fun
```

Changes the shape of the given tensor.
Each reshape may include at most one unknown dimension, which should have the value `-1`.
The length of the tensor must stay constant.

Example: `x.reshape([-1, 28 * 28])`

```nim
proc cond*(branches: openArray[(string, Fun)],
           otherwise: Fun = nil): Fun
```

Selects one of the inputs depending on which target should be evaluated.
Useful for building complex architectures such as GANs.

```nim
macro makeOpt*(opt: typed, args: varargs[untyped]): untyped
```

Create an optimizer from procedure `opt` by setting all optional arguments of `opt`.
The first two arguments to `opt` are the parameter to optimize and its gradient.
They must have the types `var Fun` and `Fun`.
`opt` may not return a value.

Example: `adam.makeOpt(0.01, beta1=0.5)`

### Models

```nim
proc compile*[T](graphs: varargs[Fun]): Model[T]
```

Compiles a computation graph to a model.
The generic parameter `T` may be one of `float32` or `float64`.

```nim
proc call*[T](model: Model[T],
              target: string,
              args: openArray[(string, Tensor[T])]): Tensor[T]
```

Computes the value of `target` for the inputs `args`.

```nim
proc apply*[T](model: Model[T],
               target: string,
               args: openArray[(string, Tensor[T])])
```

Computes `target` and discards its value.
This procedure is useful for optimizing simple models.
In most cases `Model.fit` should be preferred since it can train in batches and automatically increments `model.epoch`.

```nim
proc fit*[T](model: Model[T],
             targetName: string,
             args: openArray[(string, Tensor[T])],
             batchSize: int = 32,
             logStatus: bool = true)
```

Computes the given target for all batches from the inputs `args`.
If the sample count is not divisible by the `batchSize`, the remaining samples are not used in the training process.
This will likely be fixed in the future.

```nim
proc emitIr*[T](model: Model[T]): string
```

Emits intermediate representation for all targets of `model`.
This is mainly used for debugging purposes.

### IO

Exprgrad provides an io module which can load commonly used datasets and save/load models to/from disk.

#### Saving and Loading Models

Models can be saved by calling the `save` procedure from `io/serialize`.
`loadModel` is used to load a model from a file.
Since `loadModel` loads the intermediate representation for the model from the file and compiles it using the JIT compiler, it is **not** recommended to load models from untrusted sources.

```nim
let model = loadModel[float32]("model.bin")
model.save("model.bin")
```

### Tensors

Exprgrad currently uses a simple tensor library providing basic functions aimed at preprocessing datasets for training.
Tensors can be created using `Tensor.new` and `Tensor.rand`, printed using `$` and accessed using the `[]` and `{}` operators.
Refer to `test/test_tensors.nim` for more examples of how to use the tensor library.

## References

Exprgrad borrows many successful concepts from other projects on array and differentiable programming languages.

- [Halide](https://halide-lang.org/)
- [Zygote.jl](https://github.com/FluxML/Zygote.jl)
- [LLVM](https://llvm.org/)

## Contributing

Currently exprgrad is still very early in its development.
All examples shown above already work, but there are still many possibilities for improvement:

- Improved multithreading
- GPU Support
- More automatic optimizations (tiling, loop fusion, ...)
- ...

If you would like to contribute to exprgrad, the following tasks might be of interest to you:

- Integrate with existing tensor libraries
- Image loading and saving
- Improve batching in `fit` procedure
- Document the tensors module

### Project Structure

The following diagram shows a simplified compilation pipeline which displays the functions of the different modules (files in `exprgrad/`) of exprgrad's compiler.

```
          parser       passes       llvmgen
Nim AST –––––––––> IR ––––––––> IR –––––––––> LLVM IR ––> Machine Code 
```

Exprgrad's compiler uses a custom intermediate representation (IR).
All program transformations including the automatic differentiation and optimization are performed within this representation.
It is defined in the module `ir.nim`.
The current compilation pipeline is defined in the `compile` procedure of the module `model.nim`.
All program transformations are currently defined in `passes.nim`.
Exprgrad uses the LLVM C-API through its own wrapper.
The LLVM IR generator and JIT compiler are defined in `llvmgen.nim`.

## License

Copyright 2021 - 2022 Can Joshua Lehmann

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
