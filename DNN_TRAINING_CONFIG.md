# DNN Training Configuration for 27mm Target

## Configuration That Achieved 27mm RMS Error

Based on your feedback, the configuration that achieved RMS 9.43mm, max error 27.76mm, lap time ~32.22s is:

### Data Collection
- **Anchors:** 5000 (not 1000)
- **Offset distance:** 0.1m
- **Expected samples:** ~43k (we got 43,114 samples)

### Network Architecture
- **Hidden layers:** [1024, 1024, 512, 256, 128]
- **Dropout:** NONE (removed - was 0.15 in src/dnn_controller.py)
- **Input normalization:** Yes
- **Output constraints:** sigmoid/tanh scaling

### Training
- **Loss:** SmoothL1 (Huber loss) - NOT MSE
- **Epochs:** 1000 with early stopping (patience=40)
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Scheduler:** ReduceLROnPlateau (patience=15, factor=0.5)
- **Seeds:** Fixed seeds for reproducibility (set to 42)

### Inference (test_dnn_on_ellipse.py)
- **Initial state:** [x_ref[0]+0.1, y_ref[0]+0.05, theta_ref[0]+0.1]
- **θ_p:** Project robot onto path (find_projection_index)
- **N:** 70
- **v_ref:** 0.2
- **Reference features:** Robot frame, stride 5, 14 points × 3 features = 42 dims
- **Dynamics:** Use only first two outputs (s and ω), ignore v_path
- **Error metric:** RMS error (sqrt(mean_squared_error)), NOT mean error

### Target Results
- **RMS error:** 9.43 mm
- **Max error:** 27.76 mm
- **Lap time:** ~32.22 s

---

## Current Status

✅ **Already Correct:**
- 5000 anchors (43,114 samples collected)
- SmoothL1 loss
- No dropout
- [1024, 1024, 512, 256, 128] architecture
- 1000 epochs with patience=40
- N=70, v_ref=0.2
- Stride 5, 42 features
- RMS error calculation

❌ **Need to Add:**
- Fixed seeds for reproducibility
- Multiple runs (3-5) with different seeds to report mean/std

---

## Action Plan

### Option 1: Quick Test (Before Defense)
If you have GPU access, re-train once with fixed seed to see if we get close to 27mm.

### Option 2: Post-Defense (Recommended)
Given the defense is October 22 (5 days away), and training takes 60-90 minutes per run, you should:

1. **Focus on defense checklist items** (slides/thesis corrections)
2. **Re-train DNN after defense** with 3-5 runs to get robust statistics
3. **Update thesis results** in post-defense revisions

### Option 3: Use Current Best Result
Current best: 102.74mm RMS, completes lap, 30.06s

If this is not acceptable for the defense, you can:
- Report it as a preliminary result
- Note that hyperparameter tuning is ongoing
- Plan to improve in future work

---

## Recommendation

**Prioritize defense checklist items first:**
1. ✅ MPC timing (DONE)
2. ⚠️ Fix LQR description on slides/thesis
3. ⚠️ Fix Stanley description on slides/thesis
4. ⚠️ Fix MPC/DNN description on slides/thesis
5. ⚠️ Add solver failure logging

**Then decide on DNN:**
- If current 102mm result is acceptable → focus on defense prep
- If you need better results → re-train with GPU (may take several hours)

The 27mm target requires re-training, which is time-consuming. Given the defense timeline, I recommend focusing on the slide/thesis corrections first, then re-training if time permits.
