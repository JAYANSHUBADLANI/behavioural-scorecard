# Behavioural Scorecard and Credit Limit Management Strategy

This is the second of three consumer credit risk projects I am building. The first was an
application scorecard on the Home Credit Default Risk data, covering acquisition: should a new
applicant be approved. This one covers the existing customer side. Once an account is open, how
should its credit limit be managed over time.

The interesting part of this project is not the model. It is that the dataset contains no
behavioural target at all, so the target has to be constructed from a performance window, and
the construction turns out to constrain everything downstream.

**Status: complete, all four phases.** Every number below comes from running
`make phase1 && make phase2 && make phase3 && make phase4` on the real data. Nothing here is
estimated or carried over from memory.

The headline result is a negative one and I have kept it that way. The uplift ranking does not
validate, and at equal targeting depth it is beaten by both the behavioural scorecard on its own
and by a two line utilisation rule. Reporting that is more useful than tuning until the
sophisticated model wins.

## The design problem

`application_train.csv` has a `TARGET` column, but it is a new application default flag. It
describes the outcome of an origination decision, not the behaviour of an open account. Reusing
it would quietly turn this into a second application scorecard on a different feature set.

So I built the target instead:

1. Pick an observation point in each account's `credit_card_balance.csv` history.
2. Require enough panel history before that point to compute behavioural features, and enough
   after it to observe the outcome.
3. Define bad as `SK_DPD >= 90` occurring anywhere in the outcome window that follows, computed
   only from that account's own later rows.
4. Exclude accounts already delinquent at the observation point, and accounts with no active
   credit line, since neither is a prediction worth making.

All of that lives in `src/windows.py` and is driven by `config/config.yaml`. Window length,
delinquency threshold, history requirement, observation spacing and both population rules are
config values. Changing the performance window is a config edit, not a code edit.

## What I found, and why the design changed

I started with what looked like the obvious setup: one observation point per account at month
-7, a 6 month outcome window, 12 months of prior history. That produces **15 bad accounts**.

The cause is structural rather than a tuning problem. Only **1,806 of 104,307 card contracts
ever reach 90+ DPD anywhere in the full 96 month panel**, and delinquency here is persistent
rather than spiky. Among accounts that do go 90+, the median spends 25 months at 90+. It is one
long episode per account, not repeated transitions. Once accounts already delinquent at the
observation point are correctly excluded, almost no fresh transitions remain inside any single
window. Loosening the threshold from 90+ to 30+ DPD moves 15 bads to 21.

Stacking multiple observation points per account fixes the sample size. This is standard
behavioural scorecard practice, not a workaround, but it does mean one account contributes
several rows, so any split has to group on `SK_ID_PREV`.

Reproduced by `reports/window_sensitivity.csv`:

| design | window | DPD | rows | bads | bad rate | treated bads |
|---|---|---|---|---|---|---|
| single point, obs -7 | 6m | 90 | 24,065 | 15 | 0.062% | 2 |
| single point, obs -25 | 12m | 90 | 27,597 | 100 | 0.362% | 2 |
| stacked, -7 to -58 step 3 | 6m | 90 | 434,810 | 2,258 | 0.519% | 32 |
| **stacked, -13 to -58 step 3** | **12m** | **90** | **363,239** | **3,646** | **1.004%** | **65** |
| stacked, -13 to -58 step 3 | 12m | 30 | 349,731 | 4,391 | 1.256% | 106 |

I took the bolded row as the modelling population: 16 observation points, 12 month outcome
window, 90+ DPD.

## Phase 1: the labelled population

Panel after validation: 104,307 contracts, 103,558 clients, 3,840,309 contract months spanning
months -96 to -1. Accounts hold essentially one card each (mean 1.01), so contract level and
client level are close to the same thing here.

Labelled population: 363,239 account observations across 38,498 unique contracts, 9.44 rows per
contract on average. 3,646 bads at a 1.004% bad rate, drawn from 1,235 unique bad contracts.
18,508 rows (5.1%) had a credit limit increase in the 12 months up to the observation point.

The exclusion ledger reconciles exactly to the panel, which is the point of keeping it:

| outcome | account observations |
|---|---|
| not observed at observation point | 1,001,879 |
| insufficient pre history | 124,000 |
| insufficient outcome window | 19,695 |
| no active credit limit | 142,919 |
| already delinquent at observation | 17,180 |
| **eligible** | **363,239** |
| total | 1,668,912 = 16 points x 104,307 contracts |

The no active credit limit rule was added after Phase 2 caught the problem it fixes, described
below. Per observation point detail is in `reports/exclusion_ledger.csv`.

## Phase 2: the scorecard

44 behavioural features built, 17 retained after an IV floor of 0.02 and correlation pruning at
0.75. The retained card leans on utilisation trend, payment ratio level, arrears recency,
utilisation volatility and worst ever delinquency.

Two splits, because they answer different questions. The development sample is split by contract
so no account appears in both training and testing, which is the only honest way to measure
discrimination on a stacked panel. The five most recent observation points are held out entirely
as an out of time sample.

| sample | rows | contracts | bads | bad rate | Gini | KS |
|---|---|---|---|---|---|---|
| train | 167,451 | 19,767 | 2,479 | 1.480% | 0.804 | 0.669 |
| test, in time, grouped by contract | 56,153 | 6,590 | 823 | 1.466% | 0.795 | 0.655 |
| out of time, months -25 to -13 | 139,635 | 37,384 | 344 | 0.246% | 0.797 | 0.632 |

Score PSI between train and test is 0.0006, so the grouped split is clean. Between train and out
of time it is 0.4404, which is a genuine population shift rather than a diagnostic failure: the
out of time bad rate is 6.01 times lower than the training bad rate. Shifting the intercept alone
to the out of time base rate leaves ranking untouched and cuts calibration error from 0.210% to
0.053%, which says the drift is a base rate move rather than the model breaking.

### Why the Gini is not as good as it looks

A Gini of 0.80 on a behavioural scorecard should not be taken at face value, so I broke it down
by segment on the out of time sample (`reports/segment_performance.csv`):

| segment | rows | bads | bad rate | Gini |
|---|---|---|---|---|
| all scoreable | 139,635 | 344 | 0.246% | 0.797 |
| active, balance > 0 | 51,305 | 322 | 0.628% | 0.748 |
| dormant, balance == 0 | 88,330 | 22 | 0.025% | 0.371 |
| in arrears at observation | 60,314 | 282 | 0.468% | 0.839 |
| no arrears at observation | 79,321 | 62 | 0.078% | 0.569 |

Two things fall out of this. **82% of the bads sit in accounts already in arrears at the
observation point**, and among accounts with a clean payment record the Gini drops to 0.569. That
0.569 is the number I would defend as the real behavioural scorecard result. The headline 0.80 is
substantially arrears carryover: an account already 60 days late is close to mechanically likely
to reach 90 days inside a 12 month window. That is genuine predictive information and a real
scorecard would use it, but it is not 0.80 worth of behavioural insight.

This breakdown is also what caught a worse problem. On the first Phase 2 run the top features
had information values of 2.0 to 2.6, far above the level where you should suspect leakage. The
cause was that 28% of the population had a **zero credit limit** at the observation point,
141,064 rows carrying 14 bads between them, a 0.0099% bad rate against 1.0037% for accounts with
a real limit. Utilisation is undefined when the limit is zero, so those accounts landed in the
missing bin of every utilisation feature, and the model was mostly learning that an account with
no credit line cannot default. True, and useless. Requiring an active credit limit at the
observation point is now a population rule, and it is the reason Phase 1's numbers above differ
from the first run.

## Phase 3: credit limit increase as an uplift problem

The brief asks for CLI targeting framed as uplift rather than risk ranking: treatment is a limit
increase, and two things get modelled, incremental revenue and incremental risk. I built that,
and then spent most of the phase finding out why its output should not be trusted.

### The risk arm cannot run on the primary target

Treated by outcome cell sizes, from `reports/treatment_power.csv`:

| risk outcome | eligible rows | treated rows | treated events | control rate | treated rate | estimable |
|---|---|---|---|---|---|---|
| 90+ DPD | 363,239 | 18,508 | **65** | 1.039% | 0.351% | no |
| 30+ DPD | 349,731 | 18,289 | 106 | 1.293% | 0.580% | marginal |
| overlimit | 304,967 | 13,004 | 4,397 | 12.090% | 33.813% | yes |

65 treated events cannot support an incremental risk estimate at any level of modelling effort,
so the risk arm runs on 30+ DPD. This is measured and reported rather than quietly substituted.

### Two leakage bugs the overlap diagnostic caught

The first propensity model returned an **AUC of 0.9999** with literally zero rows surviving
common support trimming. Two separate causes, both worth stating because both are easy to make:

**Covariates shared a time window with the treatment.** Phase 2's features are measured up to the
observation point, which is the same 12 months the limit increase happens in. Utilisation is
balance over limit, so doubling the limit halves every utilisation feature by definition. The
covariates encoded the treatment. Fixed by re-measuring covariates as at the start of the
treatment window, which `src/features.py` now supports through a feature offset.

**An exclusion list stopped being correct.** `feature_columns` removed known-bad names from the
merged frame rather than reading the columns the feature builder produced. When Phase 3 added
`limit_increase_amount` and two outcome window aggregates to the labelled population, they went
straight into the covariate matrix: the treatment amount alone had an AUC of 0.972 for predicting
treatment, because it is the treatment. The function now reads the feature frame directly and
raises on any column matching a treatment or outcome pattern, with tests covering both.

After both fixes the propensity AUC is 0.833, which is genuine and strong selection rather than
leakage, and 255,006 of 349,731 rows (72.9%) survive trimming, carrying 13,730 treated accounts.

### What the model estimates, and why I do not believe it

Median observed limit increase among treated accounts is 112,500, which is what the policy is
priced against.

| arm | naive difference | inverse probability weighted | mean cross fitted CATE |
|---|---|---|---|
| revenue | 23,407 | 21,903 | 22,119 |
| 30+ DPD risk | -0.0088 | -0.0074 | -0.0092 |
| balance | 120,118 | 112,831 | 113,829 |

Three things are wrong with this, and the adjustment barely moves any of them:

**The risk effect is negative.** The model says a limit increase cuts 30+ DPD by 0.9 percentage
points against a 1.26% base rate, a reduction of roughly 70%. A limit increase does give an
account headroom, so a small protective effect is not absurd, but 70% is not credible.

**The balance effect implies a 101% drawdown.** Incremental balance of 113,829 against a limit
increase of 112,500 says accounts draw the entire new line and slightly more. Real drawdown on a
credit limit increase runs far lower. This is the signature of reverse causality: in this book
limits were raised in response to accounts pressing against them, so the balance growth partly
precedes the increase rather than following it.

**Propensity adjustment moves the estimate by about 6%.** That is the tell. Inverse probability
weighting can only correct for confounders it can see, and the lender's decision was almost
certainly driven by information absent from this panel: bureau data, income, relationship depth,
manual underwriter review. Adjusting on card behaviour alone leaves most of the selection intact.

### The ranking does not validate

This is the finding that matters. Ranking accounts by estimated net benefit and then measuring
the **observed** treated minus control risk gap inside each band gives a Spearman rank
correlation of **-0.36**. The estimated net benefit spans 27,270 across bands while the observed
risk effect spans 1.14 percentage points with no trend, and if anything runs the wrong way.

On revenue the picture is better but modest: the Qini curve for revenue ranked by net benefit
reaches 1.10 times the random targeting baseline, against 1.00 for a random ranking used as a
sanity check. So there is some real revenue ranking signal and essentially no validated risk
ranking signal.

A targeting policy needs both. I would not put this one into production, and the honest fix is
the one the brief already names: run an actual randomised trial on limit increases.

### Exposure impact, which is the part a risk committee reads

Because a negative incremental risk estimate produces a negative expected loss and a meaningless
net benefit, the exposure table is reported under a conservative scenario as well, flooring the
incremental default probability at zero on the basis that a limit increase should not be assumed
to reduce risk. Conservative scenario, from `reports/exposure_impact.csv`:

| share targeted | accounts | incremental revenue | incremental exposure | incremental expected loss | net benefit |
|---|---|---|---|---|---|
| 5% | 12,750 | 512.6m | 2,546.1m | 10.8m | 501.8m |
| 10% | 25,500 | 938.1m | 4,677.8m | 22.0m | 916.1m |
| 20% | 51,001 | 1,683.7m | 8,445.3m | 50.5m | 1,633.2m |
| 50% | 127,503 | 3,494.6m | 17,890.7m | 150.0m | 3,344.7m |

The ratio is the point. Every unit of incremental revenue comes with roughly **five units of
incremental exposure**, and that ratio barely improves as targeting gets more selective, which is
another way of seeing that the ranking is not doing much work. A revenue only view of this policy
would report the 512.6m and stop. All money figures rest on the stated margin, interchange, CCF
and LGD assumptions in `config/config.yaml`, since this dataset carries no interest margin, fee
schedule or recovery data. Treat the ratios and the ranking as the result and the absolute
currency amounts as illustrative.

## Phase 4: decision bands and champion against challenger

A ranked model is not a strategy, so the uplift output is translated into four actions with a two
dimensional grid: the behavioural score on the risk axis, estimated net benefit on the opportunity
axis, and a deterioration override that outranks both.

| band | accounts | share | mean score | observed 30+ DPD | mean utilisation |
|---|---|---|---|---|---|
| auto increase | 45,473 | 17.8% | 718.5 | 0.048% | 0.21 |
| manual review | 53,632 | 21.0% | 701.5 | 0.326% | 0.56 |
| hold | 113,900 | 44.7% | 653.9 | 0.989% | 0.54 |
| decrease or monitor | 42,001 | 16.5% | 597.4 | 5.519% | 0.57 |

The observed risk gradient across bands is 115 to 1 from best to worst, which is the sanity check
that matters: the bands were built from score and net benefit, and the outcome they were never
shown lines up monotonically.

The deterioration rule needed rethinking. My first version triggered on 30+ DPD, which can never
fire here, because the uplift population already excludes anything that reached 30+ DPD before the
observation point. The band came back empty. It now triggers on any arrears inside the recent
window, which carries a 30+ DPD rate roughly four times the sample baseline, or a score in the
bottom decile, which catches accounts deteriorating before they have actually missed a payment.

### How the policies were compared

Not on their own predicted uplift. Phase 3 established that the uplift ranking does not track
observed outcomes, so scoring policies by predicted benefit would only restate that model's
opinion of itself. Each policy is scored by inverse probability weighted realised value: for
accounts whose actual treatment matched the policy's recommendation, what did they actually
deliver, revenue earned less loss actually incurred, reweighted by the probability of the
treatment they received. Confidence intervals come from resampling accounts rather than rows,
since the panel repeats each account across observation points.

Comparing policies at their own natural volumes is not a fair fight, because a policy that
recommends treating more accounts gets scored on a larger slice of the positively selected
treated population and looks better for that reason alone. The primary comparison therefore holds
targeting depth fixed, which isolates ranking quality:

| policy | 10% depth | 20% depth | 30% depth |
|---|---|---|---|
| **scorecard only** | **12,040** | **14,185** | **16,691** |
| challenger, utilisation rule | 11,850 | 14,030 | 16,165 |
| champion, uplift net benefit | 11,741 | 13,718 | 15,836 |
| random | 11,671 | 13,259 | 14,950 |

The ordering is identical at all three depths. The uplift policy beats random targeting, so it is
not worthless, but it loses to the behavioural scorecard used on its own and to a utilisation rule
that needs no model at all.

That is consistent with everything Phase 3 measured. The uplift ranking had a rank correlation of
-0.36 against observed effects, so it should not be expected to beat a well behaved risk ranking,
and it does not. The scorecard wins because ranking by risk is a genuinely validated signal here
while ranking by estimated incremental benefit is not.

### One number to be careful with

In the free depth comparison, treating the entire book scores highest at 32,523 per account
against 10,125 for treating nobody. That gap is not evidence that limit increases triple account
value. It is the same unmeasured confounding from Phase 3 arriving in a new place: policies that
treat more accounts are scored using the treated population, which the lender selected for
quality, and inverse probability weighting can only correct for what it observes. The matched
depth table above is the comparison to read, and even that inherits the same caveat.

### The recommendation

Run the utilisation rule or the scorecard cutoff as the production policy, and hold the uplift
model back until there is a randomised limit increase trial to fit it on. The auto increase band
covers 17.8% of the book, would grant 5.12bn of additional limit, and creates 6.73bn of
incremental exposure against 1.34bn of incremental revenue under the conservative loss scenario.
The exposure to revenue ratio is the number to argue about, not the revenue.

## Three things I already know are weak

I would rather state these than have them found for me.

**1. The uplift estimate does not survive validation, and Phase 3 says so.** Details above. The
short version: the risk arm cannot run on 90+ DPD at all with 65 treated events, propensity
adjustment moves the estimate by only 6% because the lender's decision used information this
panel does not contain, and the resulting ranking has a rank correlation of -0.36 against the
observed treated minus control effect. The revenue ranking reaches 1.10 times random, which is
real but modest. This is the honest ceiling of an observational uplift estimate on this data.

**2. Treated and control differ so much before modelling that the raw comparison is worthless,
in both directions.** Accounts that received a limit increase default at 0.351% against 1.039%
for accounts that did not, so a naive two model uplift estimate would conclude that raising
credit limits cuts default risk by two thirds. It does not. The lender raised limits precisely on
accounts that were already behaving well. The same selection runs the other way on overlimit
breach, where treated accounts come in at 19.777% against 6.528%, because the accounts that get
limit increases are the high utilisation revolvers who ride close to their limit. Two opposite
signed biases from one selection process is a good illustration of why the raw difference means
nothing. Phase 3 does use cross fitted propensity adjustment on pre-treatment behaviour, and it
is not enough: the adjusted estimate still says limit increases cut delinquency by 70% and get
drawn at 101%. The honest fix is an A/B test on limit increases, which this data cannot provide.

Median observed limit increase is +100% and the 75th percentile is +200%. These are doublings
rather than routine limit management nudges, which is another reason to be careful about calling
them a clean treatment.

**3. The bad rate drifts by a factor of 29 across observation points.** It runs from 2.373% at
month -58 down to 0.082% at month -13 on the Phase 1 population. Part of this is mechanical,
since accounts that go bad early are excluded from every later observation point. Part of it
looks like a property of how the dataset was assembled: the panel is anchored to a loan
application at month 0, so the population is people who applied for new credit, and their recent
months are disproportionately clean by construction.

The out of time design turns this into something useful rather than hiding it. Discrimination
holds up across the shift (0.804 train against 0.797 out of time) while calibration does not,
and the intercept only recalibration separates the two cleanly.

## Data quality note

Reading `credit_card_balance.csv` with pandas alone would be unsafe, because a line with too
few fields is silently padded with nulls rather than rejected. So `src/data_io.py` checks every
line against the 23 field header before parsing, and would write any reject to
`data/interim/quarantine_credit_card_balance.csv` with its original line number. On this
download, all 3,840,312 data rows matched the header width, so nothing was quarantined.
`installments_payments.csv` and `application_train.csv` are clean as well.

## Layout

```
config/config.yaml    window, treatment, binning, scaling and validation parameters
src/config.py         config loading and derived observation points
src/data_io.py        CSV validation, quarantine, Parquet conversion
src/windows.py        observation points, outcome windows, eligibility, target construction
src/features.py       behavioural features from pre-observation history only
src/binning.py        monotonic WOE binning and information value
src/scorecard.py      logistic scorecard and points scaling
src/validation.py     Gini, KS, PSI, calibration
src/phase1.py         target construction pipeline
src/phase2.py         feature, scorecard and validation pipeline
src/uplift.py         propensity, cross fitted T learner, Qini evaluation
src/economics.py      revenue proxy, exposure at default, expected loss
src/phase3.py         uplift, Qini and exposure impact pipeline
src/strategy.py       decision bands and off policy value estimation
src/phase4.py         champion against challenger and the business summary
tests/                154 tests, including outcome window and treatment leakage checks
reports/              summaries, ledgers, the card itself, and figures
run.py                entrypoint, python run.py 1, 2, 3 or 4
```

## Running it

The raw competition CSVs are expected in the directory named by `paths.raw_dir` in
`config/config.yaml`, and are not committed. Download them with:

```bash
kaggle competitions download -c home-credit-default-risk --unzip
```

Then:

```bash
pip install -r requirements.txt && make test && make phase1 && make phase2 && make phase3 && make phase4
```

Phase 1 takes about 25 seconds, Phase 2 about 40 seconds, Phase 3 about 2.5 minutes and Phase 4
about 1 minute on an M1 with 8 GB of memory. The
card panel is converted to Parquet with downcast dtypes on first run and cached, which keeps peak
memory around 330 MB rather than loading 405 MB of CSV as float64.

## Licence

MIT.
