# Generate TEP_Train_Full.h5 (0-300 h) and TEP_Test_Full.h5 (300-350 h) from one continuous run

import json
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).parent
_nb = json.loads((ROOT / "TEP_Simulation.ipynb").read_text())
for _i in (0, 1, 2, 4):
    exec("".join(_nb["cells"][_i]["source"]))

# Config

SEED = 0
RECORD_EVERY = 180        # 3 min
FLUSH_EVERY = 25 * 3600   # write to h5 every 25 simulated hours
NOISE = True
T_SPLIT = 300.0           # train 0-300 h, test 300-350 h
HOURS = 350.0

RAW_PATH = ROOT / "TEP_Run_Full_raw.h5"
TRAIN_PATH = ROOT / "TEP_Train_Full.h5"
TEST_PATH = ROOT / "TEP_Test_Full.h5"

# Faults used
FAULTS_USED = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]
FAULTS_EXCLUDED = [3, 9, 19]

# Training: 10 h normal, then two rounds of all 17 faults in 8.5 h slots (4 h fault, 4.5 h normal)
TRAIN_ROUNDS = [
    [1, 8, 15, 4, 11, 18, 2, 12, 16, 5, 13, 20, 7, 10, 17, 6, 14],
    [12, 5, 18, 2, 14, 16, 7, 11, 20, 1, 10, 15, 6, 8, 17, 4, 13],
]
TRAIN_START, TRAIN_SLOT, TRAIN_ON = 10.0, 8.5, 4.0
FAULT_ON_CAP = {6: 3.0}

# Testing: 2 h normal, then 12 faults in 4 h slots (2 h fault, 2 h normal)
TEST_ORDER = [4, 12, 1, 16, 10, 14, 2, 20, 11, 5, 15, 8]
TEST_START, TEST_SLOT, TEST_ON = T_SPLIT + 2.0, 4.0, 2.0


def build_schedule():
    schedule, t = [], TRAIN_START
    for f in [f for r in TRAIN_ROUNDS for f in r]:
        schedule.append((f, t, t + min(TRAIN_ON, FAULT_ON_CAP.get(f, TRAIN_ON))))
        t += TRAIN_SLOT
    assert t <= T_SPLIT
    t = TEST_START
    for f in TEST_ORDER:
        schedule.append((f, t, t + TEST_ON))
        t += TEST_SLOT
    assert t <= HOURS
    for r in TRAIN_ROUNDS:
        assert sorted(r) == FAULTS_USED
    assert set(TEST_ORDER) <= set(FAULTS_USED)
    return schedule


NAMES = {"xmeas": XMEAS_NAMES, "xmv": XMV_NAMES, "sp": SP_NAMES, "dist": DIST_NAMES,
         "idv": [f"IDV {i}" for i in range(1, 21)], "idv_effective": [f"IDV {i}" for i in range(1, 21)],
         "y": STATE_NAMES}


class ChunkWriter:

    def __init__(self, path):
        self.f = h5py.File(path, "w")
        self.buf = {k: [] for k in ["t", *NAMES]}

    def append(self, **row):
        for k, v in row.items():
            self.buf[k].append(v)

    def flush(self):
        if not self.buf["t"]:
            return
        for k, rows in self.buf.items():
            data = np.asarray(rows)
            if k not in self.f:
                self.f.create_dataset(k, data=data, maxshape=(None, *data.shape[1:]),
                                      chunks=True, compression="gzip")
            else:
                ds = self.f[k]
                n = ds.shape[0]
                ds.resize(n + len(data), axis=0)
                ds[n:] = data
            rows.clear()
        self.f.flush()

    def close(self, shutdown, cfg):
        self.flush()
        for key, cols in NAMES.items():
            self.f[key].attrs["names"] = cols
        self.f.attrs["shutdown"] = shutdown
        self.f.attrs["config"] = json.dumps(cfg)
        self.f.close()


# Simulation

def simulate(config, writer):

    cfg = {**CONFIG, **(config or {})}
    hours, record_every, print_every = cfg["hours"], cfg["record_every"], cfg["print_every"]
    schedule = cfg["fault_schedule"]
    noise = XNS if cfg["noise"] else np.zeros(41)
    for f, t_on, t_off in schedule:
        if not 1 <= f <= 20 or (t_off is not None and t_off <= t_on):
            raise ValueError(f"Invalid fault schedule entry {(f, t_on, t_off)}")

    plant = TEPlant(cfg["seed"])
    y = Y0.copy()
    dt = 3.0                                  # control interval (s)
    n_steps = int(hours * 3600 / dt)

    # Valve positions XMV(1..12)
    xmv = np.array([63.053, 53.980, 24.644, 61.302, 22.210, 40.064,
                    38.100, 46.534, 47.446, 41.106, 18.114, 50.0])

    # Set points SETPT(1..20) (index 0 unused)
    sp = np.zeros(21)
    for k, v in {**CONFIG["setpoints"], **cfg["setpoints"]}.items():
        sp[k] = v
    sp6_init = sp[6]

    # (gain, reset time in s or None) per loop
    tuning = {1: (1.0, None), 2: (1.0, None), 3: (1.0, None), 4: (1.0, None),
              5: (-0.083, 1.0), 6: (1.22, None), 7: (-2.06, None), 8: (-1.62, None),
              9: (0.41, None), 10: (-1.56, 1452.0), 11: (1.09, 2600.0),
              13: (18.0, 3168.0), 14: (8.3, 3168.0), 15: (2.37, 5069.0),
              16: (0.169, 236.0), 17: (1.11, 3168.0), 18: (28.3, 982.0),
              19: (-83.2 / 15, 6336.0), 20: (-3.26, 12408.0)}
    err_old = {k: 0.0 for k in tuning}
    purge_flag = 0

    # Velocity-form PI controller
    def pi(k, meas, scale, period):
        gain, taui = tuning[k]
        err = (sp[k] - meas) * 100.0 / scale
        d = gain * (err - err_old[k])
        if taui is not None:
            d += gain * err * period / taui
        err_old[k] = err
        return d

    # Analyser state (dead time + sampling)
    xmeas = np.zeros(41)
    _, meas, comp, _ = plant.model(0.0, y)
    xmeas[:22] = meas
    xmeas[22:] = comp
    comp_delayed = comp.copy()
    t_gas, t_prod = 0.1, 0.25

    shutdown = False
    t_wall = time.time()

    for step in range(n_steps + 1):
        t_s = step * dt
        t_h = t_s / 3600.0

        if print_every and t_s % print_every == 0:
            print(f"Simulated {t_h:.1f} h of {hours:.1f} h ({(time.time() - t_wall) / 60:.1f} min)", flush=True)

        # Faults
        plant.idv[:] = 0
        for f, t_on, t_off in schedule:
            if t_on <= t_h and (t_off is None or t_h < t_off):
                plant.idv[f - 1] = 1

        # Disturbances and measurements at this instant
        plant.update_walks(t_h)
        _, meas, comp, shutdown = plant.model(t_h, y)
        if shutdown:
            print(f"Plant shutdown limit reached at t = {t_h:.2f} h", flush=True)
            break
        xmeas[:22] = meas + plant.rng.normal(0, noise[:22])
        if t_h >= t_gas:   # reactor feed and purge analysers, 0.1 h
            xmeas[22:36] = comp_delayed[:14] + plant.rng.normal(0, noise[22:36])
            comp_delayed[:14] = comp[:14]
            t_gas += 0.1
        if t_h >= t_prod:  # product analyser, 0.25 h
            xmeas[36:] = comp_delayed[14:] + plant.rng.normal(0, noise[36:])
            comp_delayed[14:] = comp[14:]
            t_prod += 0.25

        # Controllers
        if step > 0:
            xmv[0] += pi(1, xmeas[1], 5811.0, dt)
            xmv[1] += pi(2, xmeas[2], 8354.0, dt)
            xmv[2] += pi(3, xmeas[0], 1.017, dt)
            xmv[3] += pi(4, xmeas[3], 15.25, dt)
            xmv[4] += pi(5, xmeas[4], 53.0, dt)

            # Purge valve with separator-pressure override
            p_sep = xmeas[12]
            if p_sep >= 2950.0:
                xmv[5], purge_flag = 100.0, 1
            elif purge_flag == 1 and p_sep >= 2633.7:
                xmv[5] = 100.0
            elif purge_flag == 1:
                xmv[5], sp[6], err_old[6], purge_flag = 40.060, sp6_init, 0.0, 0
            elif p_sep <= 2300.0:
                xmv[5], purge_flag = 0.0, 2
            elif purge_flag == 2 and p_sep <= 2633.7:
                xmv[5] = 0.0
            elif purge_flag == 2:
                xmv[5], sp[6], err_old[6], purge_flag = 40.060, sp6_init, 0.0, 0
            else:
                purge_flag = 0
                xmv[5] += pi(6, xmeas[9], 1.0, dt)

            xmv[6] += pi(7, xmeas[11], 70.0, dt)
            xmv[7] += pi(8, xmeas[14], 70.0, dt)
            xmv[8] += pi(9, xmeas[18], 460.0, dt)
            xmv[9] += pi(10, xmeas[20], 150.0, dt)
            xmv[10] += pi(11, xmeas[16], 46.0, dt)
            sp[9] += pi(16, xmeas[17], 130.0, dt) * 460.0 / 100.0
            sp[4] += pi(17, xmeas[7], 50.0, dt) * 15.25 / 100.0
            sp[10] += pi(18, xmeas[8], 150.0, dt) * 150.0 / 100.0
            if t_s % 360 == 0:   # composition loops, 6 min
                sp[3] += pi(13, xmeas[22], 100.0, 360.0) * 1.017 / 100.0
                sp[1] += pi(14, xmeas[25], 100.0, 360.0) * 5811.0 / 100.0
                sp[2] += pi(15, xmeas[26], 100.0, 360.0) * 8354.0 / 100.0
                sp[6] += pi(19, xmeas[29], 26.0, 360.0) / 100.0
            if t_s % 900 == 0:   # product E composition, 15 min
                sp[16] += pi(20, xmeas[37], 1.6, 900.0) * 130.0 / 100.0
            xmv[:11] = np.clip(xmv[:11], 0.0, 100.0)

        # Record
        if step > 0 and t_s % record_every == 0:
            dist = plant.walk(t_h)
            writer.append(t=t_s, xmeas=xmeas.copy(), xmv=xmv.copy(), sp=sp[1:].copy(), dist=dist,
                          idv=plant.idv.copy(), idv_effective=effective_idv(plant.idv, dist), y=y.copy())
        if step > 0 and t_s % FLUSH_EVERY == 0:
            writer.flush()
        if step == n_steps:
            break

        # Integrate over the control interval with valves held
        plant.update_valves(xmv)
        sol = solve_ivp(plant.rhs, (t_h, t_h + dt / 3600.0), y, method="RK45",
                        rtol=1e-6, atol=1e-8)
        y = sol.y[:, -1]

    return shutdown


# Split the raw run into train / test files

def write_split(raw, path, mask, cfg, schedule, t_range):
    with h5py.File(path, "w") as f:
        f.create_dataset("t", data=raw["t"][mask])
        for key, cols in NAMES.items():
            ds = f.create_dataset(key, data=raw[key][mask], compression="gzip")
            ds.attrs["names"] = cols
        idv = raw["idv_effective"][mask]
        fault = np.where(idv.any(axis=1), idv.argmax(axis=1) + 1, 0)
        f.create_dataset("fault", data=fault)
        f.attrs["shutdown"] = bool(raw["shutdown"])
        f.attrs["config"] = json.dumps(cfg)
        f.attrs["fault_schedule"] = json.dumps([s for s in schedule if t_range[0] <= s[1] < t_range[1]])
        f.attrs["faults_used"] = FAULTS_USED
        f.attrs["faults_excluded"] = FAULTS_EXCLUDED
        f.attrs["sample_min"] = RECORD_EVERY / 60
        f.attrs["t_range_h"] = list(t_range)


def split(cfg, schedule):
    with h5py.File(RAW_PATH, "r") as f:
        raw = {k: f[k][()] for k in ["t", *NAMES]}
        raw["shutdown"] = f.attrs["shutdown"]
    t_h = raw["t"] / 3600.0
    write_split(raw, TRAIN_PATH, t_h <= T_SPLIT, cfg, schedule, (0.0, T_SPLIT))
    write_split(raw, TEST_PATH, t_h > T_SPLIT, cfg, schedule, (T_SPLIT, HOURS))
    print(f"train: {(t_h <= T_SPLIT).sum()} samples -> {TRAIN_PATH.name}")
    print(f"test : {(t_h > T_SPLIT).sum()} samples -> {TEST_PATH.name}")


if __name__ == "__main__":
    schedule = build_schedule()
    cfg = {**CONFIG, "hours": HOURS, "seed": SEED, "record_every": RECORD_EVERY,
           "print_every": 3600 * 5, "noise": NOISE, "fault_schedule": schedule}
    for f, t_on, t_off in schedule:
        print(f"IDV {f:2d}: {t_on:6.1f} - {t_off:6.1f} h")
    writer = ChunkWriter(RAW_PATH)
    shutdown = simulate(cfg, writer)
    writer.close(shutdown, cfg)
    if shutdown:
        raise SystemExit("Shutdown during the run - partial data kept in " + RAW_PATH.name)
    split(cfg, schedule)
