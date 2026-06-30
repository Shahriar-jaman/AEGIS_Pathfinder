idf_component_register(
    SRC_DIRS 
        "edge-impulse-sdk/dsp/dct"
        "edge-impulse-sdk/dsp/kissfft"
        "edge-impulse-sdk/dsp/image"
        "edge-impulse-sdk/porting/espressif"
        "edge-impulse-sdk/tensorflow/lite/kernels"
        "edge-impulse-sdk/tensorflow/lite/kernels/internal"
        "edge-impulse-sdk/tensorflow/lite/micro"
        "edge-impulse-sdk/tensorflow/lite/micro/kernels"
        "edge-impulse-sdk/tensorflow/lite/micro/memory_planner"
        "edge-impulse-sdk/tensorflow/lite/core/api"
        "tflite-model"
    INCLUDE_DIRS 
        "."
        "edge-impulse-sdk"
        "edge-impulse-sdk/classifier"
        "edge-impulse-sdk/dsp"
        "edge-impulse-sdk/dsp/image"
        "edge-impulse-sdk/porting"
        "edge-impulse-sdk/tensorflow"
        "edge-impulse-sdk/third_party"
        "edge-impulse-sdk/third_party/flatbuffers/include"
        "edge-impulse-sdk/third_party/ruy"
        "edge-impulse-sdk/third_party/gemmlowp"
        "edge-impulse-sdk/porting/espressif/esp-dsp/modules/common/include"
        "edge-impulse-sdk/porting/espressif/esp-dsp/modules/fft/include"
        "model-parameters"
        "tflite-model"
    REQUIRES esp_timer
)

# Expand CMake's search parameters to compile every single .cpp and .cc file in the SDK
file(GLOB_RECURSE ei_sdk_sources 
    "edge-impulse-sdk/dsp/*.cpp"
    "edge-impulse-sdk/porting/*.cpp"
    "edge-impulse-sdk/tensorflow/*.cpp"
    "edge-impulse-sdk/tensorflow/*.cc"
)
target_sources(${COMPONENT_LIB} PRIVATE ${ei_sdk_sources})

# Apply hardware optimizations and completely inhibit warnings from failing the build
component_compile_options(
    -O3 
    -D__input_hw_esp32=1
    -w
)