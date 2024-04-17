# wgpu-tut
A repository following [a wgpu tutorial](https://sotrh.github.io/learn-wgpu).
- bin/ builds a cross-platform runnable binary ; ~/bin : `cargo run`
- lib/ contains the code used in bin/, and compiles itself to a cdylib for Wasm/WebGL ; ~ : `wasm-pack build --target web lib`  
  -> then open lib/index.html
