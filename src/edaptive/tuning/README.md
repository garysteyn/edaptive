# Tuning: elitist I/F-race

`ifrace.py` implements the elitist iterated F-race of

> M. López-Ibáñez, J. Dubois-Lacoste, L. Pérez Cáceres, M. Birattari, T. Stützle,
> "The irace package: Iterated racing for automatic algorithm configuration",
> *Operations Research Perspectives* 3 (2016) 43–58.

Section and equation numbers in the code's docstrings refer to this paper. The implementation follows the paper
except for the differences listed below.

## Intentional deviations

1. **Sobol' initialisation.** The first race samples an Owen-scrambled Sobol' sequence instead of uniform random
   points, so the initial configurations cover the parameter space more evenly.

2. **Log-scale parameters.** Numerical parameters flagged `log_scale_tuning` are handled in log space throughout:
   - initial values are spread uniformly in log space;
   - the initial standard deviation is half the log range, and sampling uses a truncated normal in log space;
   - the soft-restart distance and the sigma reset use the log range;
   - log-scale integers are sampled on `[log lb, log(ub + 1))`, centred on `log(value + 0.5)`, and floored.
     This is the log-space form of the paper's endpoint-unbiased integer scheme (§3.2, footnote 2);
   - log-scale reals are rounded to `digits` **significant figures** instead of `digits` decimal places. A fixed
     step of `10^-digits` would leave a range such as `[1e-4, 1e-3]` with only 10 distinct values (up to 33%
     relative error near `1e-4`). Significant figures keep the same relative precision in every decade.

   A log-scale parameter must have a positive lower bound.

3. **When the number of races is extended.** The paper adds a race only when the remaining budget is "enough for
   a new race". Here a race is added whenever budget remains. The next race stops at once if it can't sample a new
   configuration (`N_new <= 0`), so the two rules have the same effect.

Linear-scale reals are rounded to `digits` decimal places, as in the paper (Appendix A).

## Not supported

- **Conditional parameters.** Every parameter is always active. There are no parameters that are enabled only
  for certain values of another parameter, so the distance and sampling rules for disabled parameters (§3.2–3.3)
  don't apply.
- Forbidden configurations, capping, and recovery files.
