# TODO

- [x] Account for systematic error via a jitter term, computed from the pre-burst flux.
- [x] Begin implementing the full injection model.
- [x] Plotting + plot formatting (LLM help allowed here).
- [ ] Replace the instant injection step with a finite ramp.
- [ ] Fix the reverse shock to physically justified values (maybe).

## Open checks
- [ ] Is the light curve time column UTC or TDB? `TRIGGER` is UTC, so TDB would mean a 69 s offset.
- [ ] Widen the `tinj` prior, which currently stops at 0.02 d. This also overlaps the forward-shock fit window.
- [ ] Check chain convergence with the autocorrelation time.
- [ ] `fit_jitter` crashes if the pre-burst reduced chi2 is below 1 (not triggered by this data).

No more alarms.
