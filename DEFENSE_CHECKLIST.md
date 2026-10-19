# Defense Checklist - October 22, 2024

## ✅ Priority 1: MPC Timing (DONE)

**Status:** Release build benchmarked with warm starts

**Results:**
- Mean solve time: **0.83 ms** (was ~7-8 ms with debug build)
- Median solve time: **0.22 ms**
- Min solve time: **0.11 ms**
- Max solve time: **13.51 ms**

**What to update:**
- Table 4.1: Update MPC solve time from ~7.89s/step to ~0.83ms/step
- Figure 4.4: Update speed-up ratio (now much smaller but honest)
- Slides 11 and 17: Update timing numbers
- DNN speed-up: Recalculate based on new MPC timing

**Note:** The speed-up is now smaller (e.g., 0.83ms vs 0.25ms = ~3.3x instead of 30x), but this is more defensible.

---

## ⚠️ Priority 2: Fix LQR Wording (CRITICAL)

**Issue:** Slide 7 and thesis describe "fixed K" with 4-state error, but code uses gain-scheduled 3-state controller.

**What needs fixing:**

**Slide 7:**
- Change: "Fixed K, 4-state error" → "Gain-scheduled K, 3-state error"
- Update diagram to show 3 states instead of 4
- Mention 21 curvature bins for gain scheduling

**Thesis:**
- Section 4.2.1: Update LQR description to match `lqr_controller.py`
- Clarify: Q=diag([5, 0.5, 10]), R=diag([0.3, 0.3])
- Document gain scheduling over 21 curvature bins

**Additional issue in `lqr_controller.py`:**
- `e_lat` and `e_lon` names are swapped (e_lat is along-path, e_lon is cross-track)
- Cross-track error has smallest weight (0.5), which may weaken LQR under disturbance
- Consider: Either fix the names or note this in thesis

---

## ⚠️ Priority 3: Fix Stanley Wording (CRITICAL)

**Issue:** Slide 7 equations don't match implementation details.

**What needs fixing:**

**Slide 7:**
- Add: `omega = delta * v_ref / 0.1` (undocumented scale factor)
- Add: Speed reduction `v_ref * (1 - 0.3*|delta|)`
- Specify: `k_gain=10` (not default 2.5, which was tested but not used in final results)

**Thesis:**
- Section 4.2.2: Add these implementation details
- Clarify: Error computed at robot position (no front-axle projection)
- Specify: Which k_gain produced the final results

---

## ⚠️ Priority 4: Fix MPC/DNN Description (CRITICAL)

**Issue:** Slide 8 doesn't match actual implementation.

**What needs fixing:**

**Slide 8:**
- MPC bounds: Change "steering constraint" to actual bounds:
  - Speed: 0.02–4.0 m/s
  - Turn rate: ±7 rad/s
  - Path speed: 0.01–0.4 rad/s
- DNN outputs: Change "2 outputs" to "3 outputs":
  - Forward speed: 0.02–4.0 m/s
  - Angular velocity: ±7 rad/s
  - Path velocity: 0.01–0.4 rad/s

**Thesis:**
- Section 4.1: Update MPC constraint description
- Section 4.3: Update DNN architecture (3 outputs, not 2)
- Clarify which network produced Table 4.1 results

**Note:** Multiple network architectures tested:
- `src/controllers/dnn_controller.py`: [128, 128, 64, 32], dropout 0.15
- `test_dnn_on_ellipse.py`: Various architectures tested
- `final_comparison.py`: May use different architecture
- **Action:** Confirm which checkpoint produced final results

---

## ⚠️ Priority 5: Add Solver Failure Counts (IMPORTANT)

**Issue:** MPC solver failures are silent in simulations.

**What to add:**

**For all simulation scripts:**
- `simulate_mpc.py`: Count and log failures
- `final_comparison.py`: Add failure handler
- Disturbance experiments: Log failures per step

**Add to Q&A notes:**
- "How often does MPC fail?" → Provide failure count statistics
- "What happens when MPC fails?" → Explain fallback behavior

**Implementation:**
```python
# In simulation loops:
failures = 0
for step in range(steps):
    try:
        u = mpc.get_first_control(...)
    except RuntimeError:
        failures += 1
        u = [v_ref, 0, v_ref]  # fallback
```

---

## ❌ Priority 6: Unified Benchmark (POSTPONE)

**Status:** Save for post-defense revisions

**Reason:** Would change results too close to defense date

**What to do later:**
- Re-run all controllers with release build MPC
- Re-train DNN with release-build MPC data
- Update all results tables and figures

---

## Summary of Critical Updates

**Before October 22:**
1. ✅ Re-benchmark MPC with release build (DONE)
2. ⚠️ Update LQR description on Slide 7 and thesis
3. ⚠️ Update Stanley description on Slide 7 and thesis
4. ⚠️ Update MPC/DNN description on Slide 8 and thesis
5. ⚠️ Add solver failure counts to simulations and Q&A notes

**Timing estimates:**
- Slide updates: 30 minutes
- Thesis updates: 1-2 hours
- Add failure logging: 1 hour
- **Total: ~3-4 hours**

---

## Questions for Review

1. Which DNN checkpoint produced Table 4.1 results?
2. Did you test with `k_gain=2.5` or `k_gain=10` for Stanley in final results?
3. Should the LQR `e_lat`/`e_lon` naming be fixed or just documented?
4. Any other discrepancies between slides/thesis and code?
