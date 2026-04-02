# once_function - Detailed Results

| Benchmark | Time | Unit | Iterations |
|-----------|------|------|------------|
| BM_move_once_function<kLargeSize> | 13.29 | ns | 54440054 |
| BM_move_once_function<kSmallSize> | 17.13 | ns | 40441071 |
| BM_move_once_function<kMediumSize> | 18.02 | ns | 38857001 |
| BM_move_once_function<kExtraLargeSize> | 33.54 | ns | 21383979 |
| BM_move_std_function<kMediumSize> | 90.97 | ns | 7756973 |
| BM_move_std_function<kSmallSize> | 91.09 | ns | 7710676 |
| BM_move_std_function<kLargeSize> | 94.42 | ns | 7299053 |
| BM_move_std_function<kExtraLargeSize> | 105.35 | ns | 6775091 |
| BM_queue_inline_function<kSmallSize> | 1041.31 | ns | 707357 |
| BM_queue_inline_function<kMediumSize> | 1862.85 | ns | 387990 |
| BM_queue_once_function<kSmallSize> | 2739.51 | ns | 260781 |
| BM_queue_once_function<kMediumSize> | 3002.41 | ns | 234757 |
| BM_queue_inline_function<kLargeSize> | 3094.38 | ns | 220748 |
| BM_queue_std_function<kSmallSize> | 3411.29 | ns | 210644 |
| BM_queue_once_function<kLargeSize> | 3546.94 | ns | 201507 |
| BM_queue_std_function<kMediumSize> | 3713.12 | ns | 200697 |
| BM_queue_std_function<kLargeSize> | 4922.03 | ns | 147424 |
| BM_queue_std_function<kExtraLargeSize> | 6862.80 | ns | 105556 |
| BM_queue_inline_function<kExtraLargeSize> | 9352.52 | ns | 72910 |
| BM_queue_once_function<kExtraLargeSize> | 58445.14 | ns | 12402 |
