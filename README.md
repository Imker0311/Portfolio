# TSFM-TEP

Forecasting the Tennessee Eastman Process (TEP) with time series foundation models (Chronos-2, TimesFM 3.0, Timestral) and an ARX baseline. The plant data is generated with a Python version of the TEP simulator, so faults, set point changes and disturbances can be controlled and the forecasts can be checked against the true response.

The general workflow is:

1. Simulate data with one of the simulators (saved as `.h5` files)
2. Optionally filter or plot the data
3. Run the forecast notebooks for each model (results saved to `Results/`)
4. Compare the models with the compare notebooks

## Files

### Simulation

| File | What it does | Main config |
|------|--------------|-------------|
| `TEP_Simulation.ipynb` | Closed-loop TEP simulation with faults (IDV 1-20) and set point changes. Saves to `tep_run.h5` | `HOURS`, `SEED`, `RECORD_EVERY`, `NOISE`, `FAULT_SCHEDULE`, `SETPOINT_SCHEDULE` |
| `TEP_Simulation.py` | One long run with a fixed fault schedule, split into train (0-300 h) and test (300-350 h) files. Writes `TEP_Run_Full_raw.h5`, `TEP_Train_Full.h5`, `TEP_Test_Full.h5` | `HOURS`, `T_SPLIT`, `FAULTS_USED`, `TRAIN_ROUNDS`, `TEST_ORDER` |
| `TEP_Setpoint_Simulation.ipynb` | Simulation without faults, where set points are stepped away from nominal and back. Saves to `tep_sp_<dataset>[_noisy].h5` | `DATASET` (`"spaced"`, `"clean"`, `"ramp_dist"`), `NOISE`, `SEED`, `DIST_STD`, `DIST_TAU` |
| `TEP_Scenario_Simulation.py` | Generates a context run (`TEP_Scenario_Context.h5`) and can restart from any saved state to simulate operator changes. Imported by the scenario forecast notebook | `CONTEXT_VARIABLES`, `CONTEXT_STEPS`, `DIST_STD`, `DIST_TAU`, `NOISE` |

### Data and plotting

| File | What it does | Main config |
|------|--------------|-------------|
| `TEP-PlotData.ipynb` | Plots a fault run: measurements, valves, set points, disturbances and fault events | `DATA_FILE`, `T_RANGE` |
| `TEP-PlotSetpointData.ipynb` | Plots a set point run and its set point changes | `DATA_FILE`, `T_RANGE` |
| `TEP-FilterData.ipynb` | Filters a noisy run (moving average or EWMA) and saves it as `*_filter.h5` | `DATA_FILE`, `FILTER`, `WINDOW`, `ALPHA` |
| `TEP_AR1_Tuning.ipynb` | Plots the AR(1) disturbance signals of the set point simulation, to tune them without running the plant | `DIST_STD`, `DIST_TAU`, `HOURS` |
| `TEP_Disturbances_Test.ipynb` | Standalone explorer of the fault disturbance signals | `HOURS`, `SEED`, `SCHEDULE` |

### Forecasting

Each forecast notebook picks a window around an event in the data, forecasts all 41 measurements (`xmeas`) and 12 valves (`xmv`), scores them against a last-value baseline and saves the result to `Results/`.

| File | Data | Model |
|------|------|-------|
| `TEP-Chronos2-Forecast.ipynb` | Fault run | Chronos-2 |
| `TEP-TimesFM3-Forecast.ipynb` | Fault run | TimesFM 3.0 |
| `TEP-ARX-Forecast.ipynb` | Fault run | ARX + Ridge |
| `TEP-Chronos2-SetpointForecast.ipynb` | Set point run | Chronos-2 |
| `TEP-TimesFM3-SetpointForecast.ipynb` | Set point run | TimesFM 3.0 |
| `TEP-Timestral-SetpointForecast.ipynb` | Set point run | Timestral |
| `TEP-ARX-SetpointForecast.ipynb` | Set point run | ARX + Ridge |
| `TEP-Chronos2-ReactorSetpointForecast.ipynb` | Set point run | Chronos-2, reactor section only 
| `TEP-Chronos2-ScenarioForecast.ipynb` | Scenario context run | Chronos-2, forecasts the response to an operator change

Main config shared by the forecast notebooks:

| Option | Meaning |
|--------|---------|
| `DATA_FILE` | Input `.h5` file |
| `EVENT_INDEX` | Which fault or set point event to forecast across |
| `START`, `BUFFER`, `HORIZON` | Where the forecast starts and how long it is |
| `USE_DIST`, `USE_SP`, `USE_IDV`, `USE_VALVE` | Which covariate groups are used |
| `DROPPED`, `PAST_ONLY`, `COVARIATES_KNOWN_FUTURE` | Leave out covariates, or only give them over the context |
| `USE_FINETUNED` | Chronos-2 only: use the LoRA fine-tuned weights instead of the base model |
| `CHANGES` | Scenario notebook only: the operator changes to simulate and forecast |

### Comparison

| File | What it does | Main config |
|------|--------------|-------------|
| `TEP-Forecast-Compare.ipynb` | Compares the fault forecasts of all models in `Results/` | `METRIC`, `PLOT_VARIABLES` |
| `TEP-SetpointForecast-Compare.ipynb` | Same for the set point forecasts | `METRIC`, `PLOT_VARIABLES` |

### Data files

The `.h5` files are generated and not tracked in git. They all share the same layout: `t` (time), `xmeas` (41 measurements), `xmv` (12 valve commands), `sp` (20 set points), `dist` (12 disturbance signals) and `y` (50 process states). Fault runs also contain `idv` and `idv_effective` (fault flags).

## Faults (IDV)

Faults are used by `TEP_Simulation.ipynb` and `TEP_Simulation.py`, and are the events forecast in the fault notebooks (`TEP-*-Forecast.ipynb`). There are 20 faults, numbered by type, so the numbers differ from the original Downs & Vogel list (D&V number in the last column).

| IDV | Type | Disturbance | Effect in the model | D&V |
|----:|------|-------------|---------------------|----:|
| 1  | Step | A/C feed ratio, B constant | Feed 4 A fraction -0.03 | 1 |
| 2  | Step | B composition, A/C ratio constant | Feed 4 B fraction +0.005, A fraction -0.0024 | 2 |
| 3  | Step | D feed temperature | D feed temp +5 C | 3 |
| 4  | Step | Reactor CW inlet temperature | Reactor CW inlet temp +5 C | 4 |
| 5  | Step | Condenser CW inlet temperature | Condenser CW inlet temp +5 C | 5 |
| 6  | Step | A feed loss | A feed (stream 1) flow set to 0 | 6 |
| 7  | Step | C header pressure loss | A and C feed (stream 4) flow -20% | 7 |
| 8  | Random walk | A, B, C feed composition | Signals 1-2: A ±0.03, B ±0.003 | 8 |
| 9  | Random walk | D feed temperature | Signal 3: ±10 C | 9 |
| 10 | Random walk | C feed temperature | Signal 4: ±10 C | 10 |
| 11 | Random walk | Reactor CW inlet temperature | Signal 5: ±10 C | 11 |
| 12 | Random walk | Condenser CW inlet temperature | Signal 6: ±10 C | 12 |
| 13 | Random walk | Reaction kinetics | Signals 7-8: reaction 1 and 2 factors ±0.25 | 13 |
| 14 | Random walk | Stripper steam heat transfer | Signal 9: steam UA factor ±0.25 | 16 |
| 15 | Pulses | Reactor heat transfer | Signal 10: reactor UA reduced by up to 35% | 17 |
| 16 | Pulses | Condenser heat transfer | Signal 11: condenser UA reduced by up to 25% | 18 |
| 17 | Pulses | Reactor-separator flow | Signal 12: flow reduced by up to 25% | 20 |
| 18 | Valve stiction | Reactor CW valve (XMV 10) | Valve only moves if the command changes by > 2% | 14 |
| 19 | Valve stiction | Condenser CW valve (XMV 11) | Valve only moves if the command changes by > 2% | 15 |
| 20 | Valve stiction | Recycle, separator underflow, stripper product and steam valves (XMV 5, 7, 8, 9) | Valves only move if the command changes by > 2% | 19 |

### Disturbance signals

The random-walk and pulse faults act through 12 continuous disturbance signals (`dist` in the saved files). With no fault active each signal stays at its nominal value. The set point and scenario simulations use the same 12 signals, but drive them with AR(1) noise instead of faults.

| # | Signal | Nominal | Driven by IDV |
|--:|--------|--------:|--------------:|
| 1  | Feed 4 A fraction | 0.485 | 8 |
| 2  | Feed 4 B fraction | 0.005 | 8 |
| 3  | D feed temp (C) | 45 | 9 |
| 4  | C feed temp (C) | 45 | 10 |
| 5  | Reactor CW inlet temp (C) | 35 | 11 |
| 6  | Condenser CW inlet temp (C) | 40 | 12 |
| 7  | Reaction 1 factor | 1.0 | 13 |
| 8  | Reaction 2 factor | 1.0 | 13 |
| 9  | Stripper steam UA factor | 0 | 14 |
| 10 | Reactor heat transfer factor | 0 | 15 |
| 11 | Condenser heat transfer factor | 0 | 16 |
| 12 | Reactor-separator flow factor | 0 | 17 |

## Scenario simulation

This section covers `TEP_Scenario_Simulation.py` and `TEP-Chronos2-ScenarioForecast.ipynb`. The idea is to give the model a context run where a few variables move, then ask it to forecast the plant response to an operator change of those same variables. The true response is simulated from the saved state at the forecast start. No IDV faults are used here; the disturbance signals and set points are moved directly.

### Background noise

All 12 disturbance signals get AR(1) noise around their mean, updated every 3 s:

| # | Signal | Std | Correlation time (h) |
|--:|--------|----:|---------------------:|
| 1  | Feed 4 A fraction | 0.0005 | 1.0 |
| 2  | Feed 4 B fraction | 0.00008 | 1.0 |
| 3  | D feed temp (C) | 0.5 | 1.0 |
| 4  | C feed temp (C) | 0.5 | 1.0 |
| 5  | Reactor CW inlet temp (C) | 0.25 | 0.5 |
| 6  | Condenser CW inlet temp (C) | 0.1 | 0.5 |
| 7  | Reaction 1 factor | 0.0025 | 2.0 |
| 8  | Reaction 2 factor | 0.0025 | 2.0 |
| 9  | Stripper steam UA factor | 0.01 | 1.0 |
| 10 | Reactor heat transfer factor | 0.005 | 0.5 |
| 11 | Condenser heat transfer factor | 0.005 | 0.5 |
| 12 | Reactor-separator flow factor | 0.005 | 0.5 |

### Context variables

The context run (`TEP_Scenario_Context.h5`) moves 4 disturbance signals and 4 set points away from nominal and back (`CONTEXT_VARIABLES`):

| Group | Index | Variable | Nominal | Largest change |
|-------|------:|----------|--------:|---------------:|
| dist | 1  | Feed 4 A fraction | 0.485 | 0.03 |
| dist | 2  | Feed 4 B fraction | 0.005 | 0.004 |
| dist | 5  | Reactor CW inlet temp (C) | 35 | 5.0 |
| dist | 6  | Condenser CW inlet temp (C) | 40 | 5.0 |
| sp   | 11 | Stripper underflow = production rate (m3/h) | 22.949 | 0.8 |
| sp   | 18 | Reactor temperature (C) | 120.4 | 1.5 |
| sp   | 19 | Purge B (mol%) | 13.823 | 2.0 |
| sp   | 20 | Product E (mol%) | 0.8357 | 0.12 |

Each variable gets four excursions (±0.5 and ±1.0 times its largest change), all in random order, with 5-7 h of nominal operation in between.

### Operator changes

In the forecast notebook the operator can only change these 8 context variables (`CHANGES`), since they are the only ones the model has seen move. The other disturbance signals are still model inputs, but only carry background noise.

The simulator itself (`sim.simulate`) can ramp any of the 12 disturbance signals and any changeable set point: 5, 7, 8, 11, 13, 14, 15, 17, 18, 19, 20. The other set points are cascaded (1, 2, 3, 4, 6, 9, 10, 16) or unused (12).

### Model inputs

The targets are the 41 measurements (`xmeas`) and 12 valves (`xmv`). With the default settings the covariates are:

| Covariates | Count | Known over the forecast horizon? |
|------------|------:|----------------------------------|
| Disturbance signals `dist_1`-`dist_12` | 12 | Only the ones the operator changes; the rest are past only |
| Selected set points `sp_11`, `sp_18`, `sp_19`, `sp_20` | 4 | Yes |
| True valve positions | 12 | Off by default (`USE_VALVE = False`) |

Covariates that are constant over the window are dropped.

## Process sections

The plant is split into three sections so that one section can be forecast on its own, with the other sections as covariates. This is used in `TEP-Chronos2-ReactorSetpointForecast.ipynb` to choose the reactor targets and covariates. Stream numbers follow Downs & Vogel. The feed header (where the fresh feeds, stripper overhead and recycle mix into the reactor feed, stream 6) is counted with the reactor.

### 1. Reactor (15 measurements)

| XMEAS | Name | Description |
|------:|------|-------------|
| 1  | A feed (kscmh) | Fresh A feed, stream 1, into the feed header |
| 2  | D feed (kg/h) | Fresh D feed, stream 2, into the feed header |
| 3  | E feed (kg/h) | Fresh E feed, stream 3, into the feed header |
| 6  | Reactor feed rate (kscmh) | Total reactor feed, stream 6: fresh feeds + stripper overhead + recycle |
| 7  | Reactor pressure (kPa) | Reactor vapour pressure |
| 8  | Reactor level (%) | Liquid level in the reactor |
| 9  | Reactor temperature (C) | Reactor liquid temperature |
| 16 | Stripper pressure (kPa) | Named after the stripper, but computed from the feed header pressure |
| 21 | Reactor CW outlet temp (C) | Reactor cooling water outlet temperature |
| 23-28 | Reactor feed A-F (mol%) | Composition of stream 6. Analyser, every 0.1 h with 0.1 h dead time |

### 2. Separator + condenser + compressor (16 measurements)

| XMEAS | Name | Description |
|------:|------|-------------|
| 5  | Recycle flow (kscmh) | Compressor discharge, stream 8, back to the feed header |
| 10 | Purge rate (kscmh) | Purge gas, stream 9 |
| 11 | Separator temperature (C) | Temperature of the condensed reactor effluent |
| 12 | Separator level (%) | Liquid level in the separator |
| 13 | Separator pressure (kPa) | Separator vapour pressure (compressor suction) |
| 14 | Separator underflow (m3/h) | Separator liquid, stream 10, to the stripper |
| 20 | Compressor work (kW) | Recycle compressor power |
| 22 | Separator CW outlet temp (C) | Condenser cooling water outlet temperature |
| 29-36 | Purge A-H (mol%) | Composition of stream 9 (= recycle composition). Analyser, every 0.1 h with 0.1 h dead time |

### 3. Stripper (10 measurements)

| XMEAS | Name | Description |
|------:|------|-------------|
| 4  | A and C feed (kscmh) | Fresh A and C feed, stream 4, enters the stripper base as stripping gas |
| 15 | Stripper level (%) | Liquid level in the stripper base |
| 17 | Stripper underflow (m3/h) | Product, stream 11 (= production rate) |
| 18 | Stripper temperature (C) | Stripper liquid temperature |
| 19 | Stripper steam flow (kg/h) | Reboiler steam |
| 37-41 | Product D-H (mol%) | Composition of stream 11. Analyser, every 0.25 h with 0.25 h dead time |

### Variables on a section boundary

A few measurements sit between two sections. Each is assigned to one section, and the neighbouring section gets it as a covariate.

| XMEAS | Assigned to | Also matters for | Why |
|------:|-------------|------------------|-----|
| 16 | Reactor | Stripper, compressor | It is the feed header pressure, which the compressor pushes the recycle against |
| 4  | Stripper | Reactor | Enters the stripper, but almost all of it ends up in the reactor feed |
| 5, 29-36 | Separator | Reactor | Flow and composition of the recycle into the feed header |
| 14 | Separator | Stripper | Separator outlet = stripper feed |

### Covariates per section

These tables list the inputs to use when forecasting a single section. Measurements from another section are only known in the past. Set points marked * can be changed by the operator and are known over the forecast horizon; the others are cascaded and moved by outer loops. Disturbances are numbered as in the [disturbance signal table](#disturbance-signals).

#### 1. Reactor

| Type | Variables | Why |
|------|-----------|-----|
| Measurements, separator | XMEAS 5, 29-36 | Recycle flow and composition into the feed header |
| Measurements, separator | XMEAS 13 | Reactor outflow depends on reactor minus separator pressure |
| Measurements, stripper | XMEAS 4, 14, 18 | Stand in for the stripper overhead (stream 5), which is not measured |
| XMV | 1, 2, 3, 10, 12 | D, E and A feed valves, reactor CW valve, agitator (12 is constant) |
| Disturbances | 3, 5, 7, 8, 10, 12 | D feed temp, reactor CW inlet temp, reaction kinetics, reactor heat transfer, reactor-separator flow |
| Disturbances (indirect) | 1, 2 | Feed 4 composition reaches the reactor through the stripper overhead |
| Set points | 13*, 14*, 15*, 17*, 18*; cascaded 1, 2, 3, 10 | Reactor feed A, D, E, reactor level, reactor temperature |

#### 2. Separator + condenser + compressor

| Type | Variables | Why |
|------|-----------|-----|
| Measurements, reactor | XMEAS 7, 9 | Reactor pressure and temperature set the effluent flow and temperature into the condenser |
| Measurements, reactor | XMEAS 16 | Header pressure = compressor discharge pressure |
| Measurements, reactor | XMEAS 23-28 | Closest proxy for the reactor effluent composition, which is not measured |
| Measurements, stripper | XMEAS 17 | Production rate is controlled with the condenser CW valve (XMV 11) |
| XMV | 5, 6, 7, 11 | Recycle valve, purge valve, separator underflow valve, condenser CW valve |
| Disturbances | 6, 11, 12 | Condenser CW inlet temp, condenser heat transfer, reactor-separator flow |
| Set points | 5*, 7*, 11*, 19*; cascaded 6 | Recycle flow, separator level, production rate, purge B |

#### 3. Stripper

| Type | Variables | Why |
|------|-----------|-----|
| Measurements, separator | XMEAS 14, 11 | Flow and temperature of the stripper feed (stream 10) |
| Measurements, reactor | XMEAS 8 | Reactor level sets the A and C feed (XMEAS 4) through the SP 17 → SP 4 cascade |
| XMV | 4, 8, 9 | A and C feed valve, stripper product valve, steam valve |
| Disturbances | 1, 2, 4, 9 | Feed 4 composition and temperature, steam heat transfer |
| Set points | 8*, 11*, 17*, 20*; cascaded 4, 9, 16 | Stripper level, production rate, reactor level, product E |

The stripper only receives material from upstream, so it is the most self-contained section. The reactor and separator are coupled in both directions through the recycle loop.

### Control loops that cross sections

Two control loops measure in one section and act in another, so their set points matter for both.

| Loop | Measured in | Acts in |
|------|-------------|---------|
| Reactor level (SP 17) → A and C feed (SP 4, XMV 4) | Reactor | Stripper |
| Production rate (SP 11, XMEAS 17) → condenser CW (XMV 11) | Stripper | Separator |
