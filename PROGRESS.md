# Progress

Project complete. All four phases built, run end to end and documented.

## Status

| phase | what it delivers | state |
|---|---|---|
| 1 | outcome window framework and behavioural target | done |
| 2 | behavioural features, WOE binning, scorecard, validation | done |
| 3 | CLI uplift, Qini, exposure impact | done |
| 4 | decision bands, champion against challenger, write up | done |

154 tests passing. Every figure in the README comes from a real run.

## The result, in one paragraph

The behavioural scorecard works. The uplift model does not. At equal targeting depth the uplift
policy is beaten by the scorecard used on its own and by a two line utilisation rule, and it only
beats random targeting. This is consistent throughout: the uplift ranking had a rank correlation
of -0.36 against observed treated minus control effects in Phase 3, so it was never going to win
in Phase 4. The recommendation is to run the simple policy and hold the uplift model back until
there is a randomised limit increase trial to fit it on.

## What each phase found

**Phase 1.** The target design in the brief yields 15 bad accounts, because only 1,806 of 104,307
contracts ever reach 90+ DPD and delinquency here is persistent rather than spiky. Stacked
observation points fix it: 363,239 rows, 3,646 bads, 1.004%. The exclusion ledger reconciles
exactly to 16 x 104,307 account slots.

**Phase 2.** Gini 0.804 train, 0.795 grouped test, 0.797 out of time, but the segment breakdown
matters more than the headline: 82% of out of time bads sit in accounts already in arrears, and
the Gini among accounts with a clean payment record is 0.569. That is the defensible number.

**Phase 3.** The risk arm cannot run on 90+ DPD at all, with 65 treated events. Propensity
adjustment moves the estimate by only 6%, and the adjusted result still claims limit increases cut
delinquency by 70% and get drawn at 101%, neither of which is credible. The ranking does not
validate. Every unit of incremental revenue carries about five units of incremental exposure.

**Phase 4.** Decision bands show a 115 to 1 observed risk gradient from the auto increase band to
the decrease and monitor band, which is a real sanity check since the bands never saw that
outcome. Policy comparison at matched depth puts scorecard first, utilisation rule second, uplift
third, random last, identically at 10%, 20% and 30% depth.

## Bugs found and fixed along the way

These are worth remembering because each one would have produced a confidently wrong answer.

- **Zero limit accounts inflated the scorecard.** 28% of the population had no credit limit and a
  0.0099% bad rate. Utilisation is undefined without a limit, so those rows landed in the missing
  bin of every utilisation feature and drove IVs to 2.6. Fixed with a population rule.
- **Uplift covariates shared a time window with the treatment.** Raising a limit mechanically moves
  every utilisation feature. Fixed with a feature offset anchoring covariates before the treatment
  window.
- **An exclusion list stopped being correct.** `feature_columns` filtered known-bad names out of a
  merged frame, so columns added later went straight into the model. The treatment amount itself
  had an AUC of 0.972 for predicting treatment. Now it reads the feature frame and raises on any
  treatment or outcome pattern.
- **Propensity AUC was measured in sample.** A boosted model memorises who was treated. Everything
  is cross fitted now, grouped by account.
- **`pivot_table` silently dropped contracts** whose values were entirely null, misaligning the
  payment matrices against the panel by 32,000 rows.
- **The decrease and monitor band was empty by construction,** because its 30+ DPD trigger could
  never fire on a population that already excludes 30+ DPD accounts.

## If you want to take it further

- A randomised limit increase trial is the only real fix for the uplift arm. Everything else is
  working around its absence.
- The observation point drift, a factor of 29 in bad rate from month -58 to -13, is handled with an
  out of time split and intercept recalibration. A time varying model would handle it better.
- The revenue proxy rests on a stated margin and interchange rate, since this dataset has no
  profitability data. Ratios are meaningful, absolute currency amounts are illustrative.

## Things for you to do

- Nothing outstanding.
- No git repository has been initialised, per your instruction to leave these as plain files.
  Nothing has been pushed anywhere. The GitHub push is yours to make.
- Project 1's scorecard code was not in this environment, so the WOE and IV layer here is a fresh
  implementation rather than reused.
