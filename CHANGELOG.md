# Changelog

Links to `legacy@23fe192` point to the original team code. Links to `erit@1948344` point to the fix.


## 0.2.0 — 2026-09-18

### Fixed

**1. Measurements were made on the wrapped phase, with an arbitrary offset.**

A transparent cell delays light by

$$\varphi(x,y) = \frac{2\pi}{\lambda}\,\Delta n\,h(x,y), \qquad \Delta n = n_{\text{cell}} - n_{\text{medium}}$$

but the camera only gives the angle of a complex number, so what we get is

$$\varphi_w = \operatorname{wrap}(\pm\varphi + \varphi_0), \qquad \operatorname{wrap}(x) \in (-\pi, \pi]$$

Three problems follow:

- **Wrapping:** an RBC gives $\varphi \approx 3.5$ rad, more than $\pi$, so the thick parts jump from $+\pi$ to $-\pi$.
- **Offset $\varphi_0$ (piston):** it is never removed. `main.m` calls the reconstruction with `NoPistonCompensation = true` ([main.m#L71][L1]), and the final correction only uses Legendre orders 2–6 ([vortexLegendre.m#L54][L2]). The background can end up anywhere, including near $\pm\pi$.
- **Sign $\pm$:** depending on whether the +1 or −1 diffraction order is picked, the result is $U$ or $U^*$, i.e. $+\varphi$ or $-\varphi$.

`main.py` computes the unwrapped phase and then passes the *wrapped* phase to everything that follows ([main.py#L19][L3], [L27][L4], [L32][L5]):

```python
segment_rbc(phase_result["phase"], ...)      # wrapped, not phase_result["phase_unwrapped_skimage"]
```

`segment_rbc` rescales that image to 0–255 and applies Otsu with `polarity="bright"` ([functions_hematological.py#L289-L300][L6]). That assumes *cells are brighter than the background*, which is false as soon as $\varphi_0 \approx \pm\pi$ or the cells wrap. **Result:** Otsu segments the background. On a real hologram: 11 "cells" of up to 250 µm², with $\Delta\varphi \approx 5.7$ rad (that is just the $2\pi$ jump at the edge).

**Fix** — [`phase.quantitative_phase`][E1]:

$$U \leftarrow U\,e^{-i\hat\varphi_0},\quad \hat\varphi_0 = \arg\!\sum_{\text{background}} U \;\;\rightarrow\;\; \text{unwrap} \;\;\rightarrow\;\; \text{sign such that } \operatorname{skew}(\varphi) > 0$$

Cells cover less area than the background, so the histogram has a positive tail when the sign is right. It is now the default; `phase_mode="legacy_wrapped"` reproduces the old behaviour.

**2. `read_complex_field(field_key=...)` never assigned the field.** The assignment was indented under the `raise`, so it never ran and the function crashed with `None.shape` ([functions_hematological.py#L36-L41][L7]). → [io.py][E2]

**3. Cells cut by the image border were measured as whole cells.** Their area, diameter and volume come out smaller than the real ones ([functions_hematological.py#L517][L8]). On the human dataset they are 47 % of the objects. → now flagged `touches_border` ([geometry.py][E3]).

**4. `cleanDataBase.py` never found the `.mat` of a rejected image.** It looks for a `.mat` whose name *ends with* the phase name ([cleanDataBase.py#L73][L9]), but `main.m` writes `complex_field_000003_X.mat` and `phase_000003_X.png` ([main.m#L79][L10], [L92][L11]). `"complex_field_000003_X".endswith("phase_000003_X")` is `False`, so the PNG was moved and the `.mat` stayed. → [quality.py][E4]

### Added

- [`reconstruction/vortex_legendre.py`][E5]: port of the MATLAB reconstruction. The whole chain now runs without MATLAB.
- [`config.py`][E6] (acquisition parameters), [`datasets/human_rbc.py`][E7], `scripts/reconstruct.py`, `scripts/batch_human_dataset.py`.

### Known, not fixed

- **Height and volume not calibrated.** $h = \lambda\,\Delta\varphi / (2\pi\,\Delta n)$ ([L1057][L12]), $V = \sum h\,\mathrm{d}x^2$. On the human dataset, with $\lambda = 0.532$ µm, $\Delta\varphi \approx 3.6$ rad and an assumed $\Delta n = 0.06$: $h \approx 5$ µm and $V \approx 230$ fL. A normal RBC is $\sim 2.5$ µm and 80–100 fL. $\Delta n$ has to be measured, not assumed.
- **$\Delta\varphi$ uses the minimum of the profile, not the background.** The profile only goes 6 steps × 0.5 px = 3 px outside the cell ([L585-L586][L13]), so the minimum falls on the cell edge ([L693-L701][L14]).

## 0.1.0 — 2026-09-18

- Original code copied unchanged to `legacy/`.
- Reorganised as the `erit` package (one module per step). Same behaviour, verified by a test against `legacy/`.

[L1]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/VortexLegendre/main.m#L71
[L2]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/VortexLegendre/vortexLegendre.m#L54-L61
[L3]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/RBC_analize/main.py#L19
[L4]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/RBC_analize/main.py#L27
[L5]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/RBC_analize/main.py#L32
[L6]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/RBC_analize/functions_hematological.py#L289-L300
[L7]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/RBC_analize/functions_hematological.py#L36-L41
[L8]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/RBC_analize/functions_hematological.py#L517
[L9]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/cleanDataBase.py#L73
[L10]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/VortexLegendre/main.m#L79
[L11]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/VortexLegendre/main.m#L92
[L12]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/RBC_analize/functions_hematological.py#L1057
[L13]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/RBC_analize/functions_hematological.py#L585-L586
[L14]: https://github.com/SebasUr/ERIT/blob/23fe192e2ed5b98cba6ae35d0789eff37570ed41/legacy/RBC_analize/functions_hematological.py#L693-L701
[E1]: https://github.com/SebasUr/ERIT/blob/19483440f29a10e5054c7e7e5ab5c422911ae0d5/src/erit/phase.py#L81-L112
[E2]: https://github.com/SebasUr/ERIT/blob/19483440f29a10e5054c7e7e5ab5c422911ae0d5/src/erit/io.py#L30-L35
[E3]: https://github.com/SebasUr/ERIT/blob/19483440f29a10e5054c7e7e5ab5c422911ae0d5/src/erit/geometry.py
[E4]: https://github.com/SebasUr/ERIT/blob/19483440f29a10e5054c7e7e5ab5c422911ae0d5/src/erit/quality.py
[E5]: https://github.com/SebasUr/ERIT/blob/19483440f29a10e5054c7e7e5ab5c422911ae0d5/src/erit/reconstruction/vortex_legendre.py
[E6]: https://github.com/SebasUr/ERIT/blob/19483440f29a10e5054c7e7e5ab5c422911ae0d5/src/erit/config.py
[E7]: https://github.com/SebasUr/ERIT/blob/19483440f29a10e5054c7e7e5ab5c422911ae0d5/src/erit/datasets/human_rbc.py
