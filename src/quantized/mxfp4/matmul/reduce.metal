const size_t index = thread_position_in_grid.x;
if (index >= size_t(ELEMENTS)) return;
float sum = 0.0f;
for (uint part = 0; part < PARTS; ++part) {
    sum += partials[size_t(part) * ELEMENTS + index];
}
output[index] = T(sum);
