Drain
=====

Deep-learning and autograd package in D.

Work in progress.

Tenets
======

- No mandatory external dependencies.
  - Utilizing this library in your project should be as simple as including it like any other D library.
  - Once your project is built, it can be deployed anywhere like any other D application, with no additional runtime dependencies.
  - Future additions (e.g. for hardware acceleration) may include *optional* dependencies.
- `nothrow`, `@nogc`.
  - Memory is pre-allocated statically, there is no dynamic memory allocation.
- Utilize the full strength of the D programming language where appropriate.
  - Instead of copying an existing API (such as TensorFlow or Keras), the library API is designed around D features (metaprogramming / templates / template constraints).
- Take advantage of powerful optimizations provided by optimizing D compilers such as LDC.
  - The library may use several layers of abstraction to facilitate expressiveness and code quality. 
  - Debug/DMD builds are much slower that LDC/release builds, however DMD performance is a non-goal.
- Minimize runtime overhead.
  - Indirections are avoided when possible. Instead, memory requirements are pre-computed for entire graphs and are made available to implementations.
