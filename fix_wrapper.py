import sys
with open('3rdparty/llama.cpp/ggml/src/ggml.c', 'r') as f:
    content = f.read()

# Define the wrapper before the struct
wrapper = """
static void dequantize_row_i2_s_wrapper(const void * vx, float * y, int64_t n) {
    const uint8_t * x = (const uint8_t *) vx;
    const float i2_scale = ((const float *)(x + (n / 4)))[0];
    dequantize_row_i2_s(x, y, n, i2_scale);
}

const struct ggml_type_traits type_traits[GGML_TYPE_COUNT]
"""
content = content.replace("const struct ggml_type_traits type_traits[GGML_TYPE_COUNT]", wrapper)

with open('3rdparty/llama.cpp/ggml/src/ggml.c', 'w') as f:
    f.write(content)
