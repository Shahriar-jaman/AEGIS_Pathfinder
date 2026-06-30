idf_component_register(SRCS "main.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES esp_camera edge_impulse nvs_flash esp_wifi esp_http_server esp_pm esp_timer)

# Suppress strict notice-warnings when compiling main.cpp
component_compile_options(
    -Wno-error=cpp
    -Wno-error=missing-field-initializers
    -Wno-missing-field-initializers
)