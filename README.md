# Pong-rust

Implementation of the game Pong using rust as programming language.

For this project I followed the "vulkan tutorial" up to the index buffers part using `unkownue/vulkan-tutorial-rust` repository as inspiration.

# Getting started

## External dependencies

This project was conceived in an Archlinux system which already had a lot of development tools available, so some external dependencies might have been overlooked or simply ignored. Below follows a list with some of the dependencies deemed important to work with this code.

### Graphics card and drivers

This project runs using the Vulkan API, so it is expected that its host would kindly provide a graphics card with Vulkan support and have all required drivers in proper working condition.

### Shader compiler

The binary `glslc` is used to compile GLSL into SPIR-V or something of the sorts.

Arch package: [shaderc](https://archlinux.org/packages/extra/x86_64/shaderc/)

### Vulkan stuff

Arch packages:
  - [vulkan-tools](https://archlinux.org/packages/extra/x86_64/vulkan-tools/)
  - [vulkan-extra-tools](https://archlinux.org/packages/extra/x86_64/vulkan-extra-tools/)
  - [vulkan-validation-layers](https://archlinux.org/packages/extra/x86_64/vulkan-validation-layers/)

### Rust

Tested with `rustc 1.73.0`

## Compile shaders

On the root of the project, run this command:

`scripts/compile-shaders.sh shaders/src shaders/spv`

Note: there are clearly no cleanup systems in place. It would be considered good manners to cleanup things with a friendly `rm -rf ./shaders/spv` before beginning any work.

## Compile and run the game

### Debug/Dev profile

Well, `cargo build` and `cargo run`. Not using no fancy stuff in here.

### Release profile

Same as debug, but with a `--release` flag added to the listed commands.

# Wishlist

- [ ] Add text support
- [ ] Add "Welcome" and "Game Over" messages
- [ ] Show score

- [ ] Add main menu
- [ ] Create configuration menu

## Things that may be added as configuration

- Score needed for victory
- Paddle speed
- Ball speed
- Game modes
