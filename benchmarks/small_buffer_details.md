# small_buffer - Detailed Results

| Benchmark | Time | Unit | Iterations |
|-----------|------|------|------------|
| BM_newdelete<kSmallSize>/8192 | 79599.70 | ns | 8652 |
| BM_newdelete<kSmallSize>/8192/threads:16 | 100415.84 | ns | 6768 |
| BM_small_buffer_allocator<kLargeSize>/8192 | 112097.97 | ns | 6276 |
| BM_small_buffer_allocator<kMediumSize>/8192 | 113413.43 | ns | 6119 |
| BM_small_buffer_allocator<kSmallSize>/8192 | 151423.03 | ns | 4595 |
| BM_small_buffer_allocator<kSmallSize>/8192/threads:16 | 196510.87 | ns | 3648 |
| BM_small_buffer_allocator<kMediumSize>/8192/threads:16 | 215262.76 | ns | 3040 |
| BM_small_buffer_allocator<kLargeSize>/8192/threads:16 | 237795.91 | ns | 2896 |
| BM_newdelete<kSmallSize>/32768 | 333435.65 | ns | 2125 |
| BM_newdelete<kSmallSize>/32768/threads:16 | 422098.80 | ns | 1632 |
| BM_newdelete<kMediumSize>/8192 | 437184.17 | ns | 1565 |
| BM_small_buffer_allocator<kMediumSize>/32768 | 449722.73 | ns | 1567 |
| BM_small_buffer_allocator<kLargeSize>/32768 | 462629.58 | ns | 1577 |
| BM_small_buffer_allocator<kSmallSize>/32768 | 589785.80 | ns | 1194 |
| BM_newdelete<kMediumSize>/8192/threads:16 | 608506.93 | ns | 1104 |
| BM_small_buffer_allocator<kSmallSize>/32768/threads:16 | 732593.81 | ns | 880 |
| BM_newdelete<kLargeSize>/8192 | 778251.18 | ns | 908 |
| BM_small_buffer_allocator<kMediumSize>/32768/threads:16 | 866150.93 | ns | 768 |
| BM_small_buffer_allocator<kLargeSize>/32768/threads:16 | 951621.02 | ns | 704 |
| BM_newdelete<kLargeSize>/8192/threads:16 | 1198657.34 | ns | 560 |
| BM_newdelete<kMediumSize>/32768 | 2107102.61 | ns | 353 |
| BM_newdelete<kMediumSize>/32768/threads:16 | 2983152.38 | ns | 224 |
| BM_newdelete<kLargeSize>/32768 | 3427922.12 | ns | 208 |
| BM_newdelete<kLargeSize>/32768/threads:16 | 57061692.12 | ns | 16 |
