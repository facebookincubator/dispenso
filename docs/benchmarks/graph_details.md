# graph - Detailed Results

| Benchmark | Time | Unit | Iterations |
|-----------|------|------|------------|
| BM_build_bi_prop_dependency_group | 430.35 | ns | 1567773 |
| BM_execute_dependency_chain<dispenso::Graph> | 7832.33 | ns | 91950 |
| BM_execute_dependency_chain<dispenso::BiPropGraph> | 7881.34 | ns | 90236 |
| BM_build_dependency_chain<dispenso::Graph> | 17572.99 | ns | 39706 |
| BM_build_dependency_chain<dispenso::BiPropGraph> | 18148.89 | ns | 39736 |
| BM_build_bi_prop_dependency_chain | 44498.37 | ns | 15908 |
| BM_forward_propagator_node<dispenso::Graph> | 1428797.45 | ns | 486 |
| BM_forward_propagator_node<dispenso::BiPropGraph> | 1463463.54 | ns | 480 |
| BM_build_big_tree<dispenso::BiPropGraph> | 1535004.83 | ns | 463 |
| BM_build_big_tree<dispenso::Graph> | 1728478.59 | ns | 411 |
| BM_taskflow_build_big_tree | 2741159.89 | ns | 247 |
