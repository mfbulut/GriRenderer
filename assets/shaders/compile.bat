@echo off
glslc --target-env=vulkan1.2 "%~dp0shader.vert" -o "%~dp0shader.vert.spv"
glslc --target-env=vulkan1.2 "%~dp0shader.frag" -o "%~dp0shader.frag.spv"
echo Shaders compiled successfully.
