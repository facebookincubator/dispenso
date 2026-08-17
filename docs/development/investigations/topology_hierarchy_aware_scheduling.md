# Topology-hierarchy-aware scheduling (beyond L3-as-NUMA-proxy)

**Context.** `kAdaptive` prefers same-L3 steal victims and `buildThreadGroups`
groups by L2/L3, using L3 as a lightweight NUMA proxy. Holds on AMD
(CCX ≈ L3 ≈ domain) but breaks where L3 ≠ memory domain: monolithic-L3 Intel SNC,
and virtualized hosts. Would give the "NUMA and topology awareness" backlog item
below a shared substrate.

**Sourcing (tested on the EPYC-Genoa dev VM).**
- **OS topology (Linux sysfs / FreeBSD sysctl) is the portable primary** — the
  only source that works on x86 *and* ARM and carries NUMA memory domains
  (firmware ACPI SRAT/SLIT; ARM ACPI PPTT). `CpuSet` already reads NUMA domains;
  the gap is that scheduling only consumes the L2/L3 levels.
- **x86 CPUID** (`0x1F`/`0x0B`, `0x8000001D`/`0x04`, AMD `0x8000001E`) is a
  bare-metal enrichment only, and needs a sanity check. On this VM it is fully
  flattened — no leaf `0x1F`, a synthetic 256-way L3, x2APIC IDs a flat `0..165`
  with no die/socket bits, `0x8000001E` reporting one node — so it adds nothing
  over sysfs, and is meaningless under vCPU migration anyway. `/proc/cpuinfo` is
  a weaker rendering of the same data, not a distinct source.
- **Core-to-core latency probe** (atomic cacheline ping-pong across pinned
  vCPUs) is the *only* method that saw through this VM: a clean two-tier split
  (~40 ns near vs ~200 ns far) that CPUID and sysfs both reported as flat. Worth
  an **opt-in, coarse, stability-gated** discovery mode — but numbers are ~10x
  inflated on a shared VM and drift with vCPU migration, so trust it only when
  placement is stable, and use it for CPU/cache grouping, not level labeling.
- CPU **model-string → SKU-layout** lookup is a last-resort bare-metal heuristic;
  the VM reports a generic "EPYC-Genoa" with no SKU, and a real SKU still would
  not reveal the vCPU→pCPU mapping.

**Actionability caveat.** In a NUMA-flattened guest you can act on CPU/cache
grouping (vCPU affinity) but **cannot** place memory in a hidden domain (only
node 0 exists to `mbind`); realistic VM payoff is thread grouping, not memory
locality. Proper validation needs bare metal (multi-CCX AMD NPS4 / Intel SNC).

**Goal.** A real hierarchy (socket ⊃ NUMA/SNC domain ⊃ L3 ⊃ L2/SMP) in
`CpuSet`, exposed as ordered levels + nearest-common-level/distance, consumed by
`buildThreadGroups` and kAdaptive victim ranking
(same-L2 > L3 > domain > socket > remote); prefer NUMA domains over L3, degrade
to the finest level the OS differentiates.
