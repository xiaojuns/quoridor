cmake_minimum_required(VERSION 3.18)

project(quoridor_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QUORIDOR_BUILD_PYTHON "Build the Python extension" ON)
option(QUORIDOR_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)

add_library(quoridor_sanitizers INTERFACE)
if(QUORIDOR_ENABLE_SANITIZERS)
  target_compile_options(quoridor_sanitizers INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer
  )
  target_link_options(quoridor_sanitizers INTERFACE
    -fsanitize=address,undefined
  )
endif()

add_library(quoridor_core STATIC
  src/encoder.cpp
  src/game.cpp
  src/mcts.cpp
)
set_target_properties(quoridor_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(quoridor_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(quoridor_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(quoridor_core PUBLIC quoridor_sanitizers)

enable_testing()

add_executable(quoridor_game_tests tests/test_game.cpp)
target_link_libraries(quoridor_game_tests PRIVATE quoridor_core)
target_compile_options(quoridor_game_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME quoridor_game_tests COMMAND quoridor_game_tests)

add_executable(quoridor_encoder_tests tests/test_encoder.cpp)
target_link_libraries(quoridor_encoder_tests PRIVATE quoridor_core)
target_compile_options(quoridor_encoder_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME quoridor_encoder_tests COMMAND quoridor_encoder_tests)

add_executable(quoridor_mcts_tests tests/test_mcts.cpp)
target_link_libraries(quoridor_mcts_tests PRIVATE quoridor_core)
target_compile_options(quoridor_mcts_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME quoridor_mcts_tests COMMAND quoridor_mcts_tests)

if(QUORIDOR_BUILD_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "from torch.utils.cpp_extension import include_paths; print(include_paths()[0])"
    RESULT_VARIABLE TORCH_INCLUDE_RESULT
    OUTPUT_VARIABLE TORCH_INCLUDE_DIR
    ERROR_VARIABLE TORCH_INCLUDE_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT TORCH_INCLUDE_RESULT EQUAL 0 OR
     NOT EXISTS "${TORCH_INCLUDE_DIR}/pybind11/pybind11.h")
    message(FATAL_ERROR
      "Could not locate the pybind11 headers bundled with PyTorch. "
      "Python error: ${TORCH_INCLUDE_ERROR}")
  endif()

  Python3_add_library(quoridor_native MODULE WITH_SOABI src/bindings.cpp)
  target_link_libraries(quoridor_native PRIVATE quoridor_core)
  target_include_directories(quoridor_native PRIVATE ${TORCH_INCLUDE_DIR})
  # PyTorch's bundled pybind11 uses an empty variadic macro extension that GCC
  # diagnoses under -Wpedantic. Keep strict warnings for our core and suppress
  # that third-party-only diagnostic for the binding translation unit.
  target_compile_options(quoridor_native PRIVATE
    -Wall -Wextra -Wno-pedantic
  )
  set_target_properties(quoridor_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
  )
endif()
