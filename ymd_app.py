# =============================================================
#  Yaw Moment Diagram app  (bicycle model + Pacejka tyre model)
#  Vehicle logic copied from Excel sheet 'Yaw Moment DIagram',
#  tyre formulas copied one-for-one from Excel sheet 'Sheet1'.
#  Run with:   streamlit run ymd_app.py
# =============================================================
import re
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Numpy versions of the Excel functions, so formulas read like the sheet
SIN, COS, ATAN, EXP, SQRT, ABS, SIGN = np.sin, np.cos, np.arctan, np.exp, np.sqrt, np.abs, np.sign
PI = np.pi
MIN = np.minimum


# -------------------------------------------------------------
# 1. Read the .tir file  ->  dictionary {"PCX1": 2.307, ...}
# -------------------------------------------------------------
def read_tir(text):
    P = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z_0-9]+)\s*=\s*([-+0-9.eE]+)", line)
        if m:
            try:
                P[m.group(1).upper()] = float(m.group(2))
            except ValueError:
                pass
    return P


# -------------------------------------------------------------
# 2. The tyre model  (cell references from Sheet1 in comments)
# -------------------------------------------------------------
def tyre_model(P, Fz, alpha_deg, camber_deg, kappa, p_kpa, L, zeta=1.0):
    g = lambda k: P.get(k, 0.0)            # missing coefficient -> 0

    # ---- inputs (C2:C7, D4, D5, G4:G6) ----
    alpha = alpha_deg * PI / 180            # D4
    gam = SIN(camber_deg * PI / 180)        # D5  (sin of camber)
    Fz0 = g("FNOMIN")                       # C3
    Fz0p = 1.0 * Fz0                        # G4  Fz0' (scale G3 = 1)
    dfz = (Fz - Fz0p) / Fz0p                # G5
    dpi = (p_kpa * 1000 - g("NOMPRES")) / g("NOMPRES")   # G6
    R0 = g("UNLOADED_RADIUS")               # P3
    Vx = g("LONGVL")                        # M4

    # ---- pure longitudinal Fx0 (C9:C18) ----
    Shx = (g("PHX1") + g("PHX2") * dfz) * L["LHX"]                                   # C17
    Svx = Fz * (g("PVX1") + g("PVX2") * dfz) * (Vx / (Vx + 0.1)) * L["LVX"] * L["LMU"]  # C18
    kx = kappa + Shx                                                                  # C10
    Cx = g("PCX1") * L["LC"]                                                          # C11
    mux = (g("PDX1") + g("PDX2") * dfz) * (1 + g("PPX3") * dpi + g("PPX4") * dpi**2) * L["LMU"]  # C13
    Dx = mux * Fz                                                                     # C12
    Ex = (g("PEX1") + g("PEX2") * dfz + g("PEX3") * dfz * dfz) * (1 - g("PEX4") * SIGN(kx)) * L["LE"]  # C14
    Kxk = Fz * L["LKX"] * (g("PKX1") + g("PKX2") * dfz) * EXP(g("PKX3") * dfz) * (1 + g("PPX1") * dpi + g("PPX2") * dpi**2)  # C15
    Bx = Kxk / (Cx * Dx + 0.1)                                                        # C16
    Fx0 = Dx * SIN(Cx * ATAN(Bx * kx - Ex * (Bx * kx - ATAN(Bx * kx)))) + Svx         # C9

    # ---- pure lateral Fy0 (C21:C32) ----
    Cy = g("PCY1") * L["LC"]                                                          # C23
    muy = ((g("PDY1") + g("PDY2") * dfz) * (1 + g("PPY3") * dpi + g("PPY4") * dpi**2) / (1 + g("PDY3") * gam * gam)) * L["LMU"]  # C25
    Dy = muy * Fz                                                                     # C24
    Ey = (g("PEY1") + g("PEY2") * dfz) * (1 + g("PEY5") * gam * gam - (g("PEY3") + g("PEY4") * gam) * SIGN(alpha)) * L["LE"]  # C26
    Kya = (g("PKY1") * Fz0p * (1 + g("PPY1") * dpi)
           * SIN(g("PKY4") * ATAN(Fz / ((g("PKY2") + g("PKY5") * gam * gam) * (1 + g("PPY2") * dpi) * Fz0p)))
           / (1 + g("PKY3") * gam * gam) * L["LKY"])                                  # C27
    By = Kya / (Cy * Dy + 0.1)                                                        # C28
    Kyg0 = Fz * (g("PKY6") + g("PKY7") * dfz) * (1 + g("PPY5") * dpi) * L["LKYC"]     # C32
    Svyg = Fz * (g("PVY3") + g("PVY4") * dfz) * gam * L["LKYC"] * L["LMU"]            # C30
    Svy = Fz * (g("PVY1") + g("PVY2") * dfz) * L["LVY"] * L["LMU"] + Svyg             # C31
    Shy = (g("PHY1") + g("PHY2") * dfz) * L["LHY"] + (Kyg0 * gam - Svyg) / Kya        # C29
    ay = alpha + Shy                                                                  # C22
    Fy0 = Dy * SIN(Cy * ATAN(By * ay - Ey * (By * ay - ATAN(By * ay)))) + Svy         # C21

    # ---- pure aligning torque Mz0 (C35:C51) ----
    SHt = g("QHZ1") + g("QHZ2") * dfz + (g("QHZ3") + g("QHZ4") * dfz) * gam           # C35
    at = alpha + SHt                                                                  # C36
    Kyap = Kya                                                                        # C37
    SHf = Shy + Svy / Kyap                                                            # C38
    ar = alpha + SHf                                                                  # C39
    Bt = ((g("QBZ1") + g("QBZ2") * dfz + g("QBZ3") * dfz**2)
          * (1 + g("QBZ4") * gam + g("QBZ5") * ABS(gam) + g("QBZ6") * gam**2) * L["LKY"] / L["LMU"])  # C40
    Ct = g("QCZ1")                                                                    # C41
    Dt0 = Fz * (R0 / Fz0p) * (g("QDZ1") + g("QDZ2") * dfz) * (1 - g("PPZ1") * dpi) * L["LTR"] * SIGN(Vx)  # C42
    Dt = Dt0 * (1 + g("QDZ3") * ABS(gam) + g("QDZ4") * gam**2)                        # C43
    Et = MIN((g("QEZ1") + g("QEZ2") * dfz + g("QEZ3") * dfz**2)
             * (1 + (g("QEZ4") + g("QEZ5") * gam) * (2 / PI) * ATAN(Bt * Ct * at)), 1)  # C44
    t0 = Dt * COS(Ct * ATAN(Bt * at - Et * (Bt * at - ATAN(Bt * at)))) * COS(alpha)   # C45
    Mzp0 = -t0 * Fy0                                                                  # C46
    Br = g("QBZ9") * L["LKY"] / L["LMU"] + g("QBZ10") * By * Cy                       # C47
    Cr = zeta                                                                         # C48
    Dr = (Fz * R0 * ((g("QDZ6") + g("QDZ7") * dfz) * L["LRES"]
                     + (g("QDZ8") + g("QDZ9") * dfz) * (1 + g("PPZ2") * dpi) * gam * L["LKZC"]
                     + (g("QDZ10") + g("QDZ11") * dfz) * gam * ABS(gam))
          * COS(alpha) * L["LMU"] * SIGN(Vx))                                         # C49
    Mzr0 = Dr * COS(Cr * ATAN(Br * ar))                                               # C50
    Mz0 = Mzp0 + Mzr0                                                                 # C51

    # ---- combined longitudinal Fx (C54:C61) ----
    SHxa = g("RHX1")                                                                  # C54
    a_s = alpha + SHxa                                                                # C55
    Bxa = (g("RBX1") + g("RBX3") * gam**2) * COS(ATAN(g("RBX2") * kappa)) * L["LXAL"]  # C56
    Cxa = g("RCX1")                                                                   # C57
    Exa = MIN(g("REX1") + g("REX2") * dfz, 1)                                         # C58
    Gxa0 = COS(Cxa * ATAN(Bxa * SHxa - Exa * (Bxa * SHxa - ATAN(Bxa * SHxa))))        # C59
    Gxa = COS(Cxa * ATAN(Bxa * a_s - Exa * (Bxa * a_s - ATAN(Bxa * a_s)))) / Gxa0     # C60
    Fx = Gxa * Fx0                                                                    # C61

    # ---- combined lateral Fy (C64:C73) ----
    SHyk = g("RHY1") + g("RHY2") * dfz                                                # C64
    ks = kappa + SHyk                                                                 # C65
    Byk = (g("RBY1") + g("RBY4") * gam**2) * COS(ATAN(g("RBY2") * (alpha - g("RBY3")))) * L["LYKA"]  # C66
    Cyk = g("RCY1")                                                                   # C67
    Eyk = MIN(g("REY1") + g("REY2") * dfz, 1)                                         # C68
    DVyk = muy * Fz * (g("RVY1") + g("RVY2") * dfz + g("RVY3") * gam) * COS(ATAN(g("RVY4") * alpha))  # C69
    SVyk = DVyk * SIN(g("RVY5") * ATAN(g("RVY6") * kappa)) * L["LVYKA"]               # C70
    Gyk0 = COS(Cyk * ATAN(Byk * SHyk - Eyk * (Byk * SHyk - ATAN(Byk * SHyk))))        # C71
    Gyk = COS(Cyk * ATAN(Byk * ks - Eyk * (Byk * ks - ATAN(Byk * ks)))) / Gyk0        # C72
    Fy = Gyk * Fy0 + SVyk                                                             # C73

    # ---- combined aligning torque Mz (C76:C84) ----
    ratio = Kxk / Kyap                                                                # C76
    at_eq = SQRT(at**2 + (ratio * kappa)**2) * SIGN(at)                               # C77
    ar_eq = SQRT(ar**2 + (ratio * kappa)**2) * SIGN(ar)                               # C78
    t = Dt * COS(Ct * ATAN(Bt * at_eq - Et * (Bt * at_eq - ATAN(Bt * at_eq)))) * COS(alpha)  # C79
    Fyp = Fy - SVyk                                                                   # C80
    Mzp = -t * Fyp                                                                    # C81
    Mzr = Dr * COS(Cr * ATAN(Br * ar_eq))                                             # C82
    s = R0 * (g("SSZ1") + g("SSZ2") * (Fy / Fz0p) + (g("SSZ3") + g("SSZ4") * dfz) * gam) * L["LS"]  # C83
    Mz = Mzp + Mzr + s * Fx                                                           # C84

    # ---- overturning moment Mx (C87:C88) ----
    Mx0 = Fz * R0 * (g("QSX1") - g("QSX2") * gam * (1 + g("PPMX1") * dpi) + g("QSX3") * Fy0 / Fz0p) * L["LMX"]  # C87
    Mx = Fz * R0 * (g("QSX1") - g("QSX2") * gam * (1 + g("PPMX1") * dpi) + g("QSX3") * Fy / Fz0p) * L["LMX"]    # C88

    return dict(Fx0=Fx0, Fy0=Fy0, Mz0=Mz0, Fx=Fx, Fy=Fy, Mz=Mz, Mx0=Mx0, Mx=Mx,
                kx=kx, Cx=Cx, Dx=Dx, mux=mux, Ex=Ex, Kxk=Kxk, Bx=Bx, Shx=Shx, Svx=Svx,
                ay=ay, Cy=Cy, Dy=Dy, muy=muy, Ey=Ey, Kya=Kya, By=By, Shy=Shy, Svyg=Svyg, Svy=Svy, Kyg0=Kyg0,
                SHt=SHt, Bt=Bt, Ct=Ct, Dt=Dt, Et=Et, t0=t0, Br=Br, Dr=Dr, Mzr0=Mzr0,
                Bxa=Bxa, Exa=Exa, Gxa=Gxa, Byk=Byk, Eyk=Eyk, SVyk=SVyk, Gyk=Gyk,
                at_eq=at_eq, ar_eq=ar_eq, t=t, Mzr=Mzr, s=s)


# -------------------------------------------------------------
# 3. Vehicle model  (cells from sheet 'Yaw Moment DIagram')
# -------------------------------------------------------------
def solve_ymd(P, L, veh, delta_deg, beta_deg, steps=60, ay_range=10.0):
    """Steady-state bicycle model. Works on single numbers or whole grids.
    For each (delta, beta) it keeps guessing ay until the ay produced by the
    tyre forces equals the guess (bisection: halve the search window each step)."""
    g = 9.81
    W = veh["m"] * g                                  # E4  weight (N)
    a, b, V = veh["a"], veh["b"], veh["V"]            # E6, E7, E16
    Fzf = W * b / (a + b) / 2                         # E8/2  load per front tyre
    Fzr = W * a / (a + b) / 2                         # E9/2  load per rear tyre
    d = np.radians(delta_deg)                         # E14
    bt = np.radians(beta_deg)                         # E15

    def forces(ay):                                   # ay = the guess (I4)
        r = ay * g / V                                # I7  yaw velocity
        af = bt + a * r / V - d                       # I9  front slip angle
        ar = bt - b * r / V                           # I10 rear slip angle
        Ff = 2 * tyre_model(P, Fzf, np.degrees(af), veh["camf"], 0.0, veh["p"], L)["Fy0"] * np.cos(d)  # I12
        Fr = 2 * tyre_model(P, Fzr, np.degrees(ar), veh["camr"], 0.0, veh["p"], L)["Fy0"]              # I13
        return Ff, Fr, af, ar, r

    shape = np.broadcast(d, bt).shape
    lo = np.full(shape, -ay_range)                    # lowest possible guess (g)
    hi = np.full(shape, ay_range)                     # highest possible guess (g)
    for _ in range(steps):
        guess = (lo + hi) / 2
        Ff, Fr, af, ar, r = forces(guess)
        too_low = (Ff + Fr) / W > guess               # tyres give more ay than guessed -> guess higher
        lo = np.where(too_low, guess, lo)
        hi = np.where(too_low, hi, guess)
    ay = (lo + hi) / 2                                # matched guess
    Ff, Fr, af, ar, r = forces(ay)
    resid = np.abs((Ff + Fr) / W - ay)                # I15 vs I4 mismatch (should be ~0)
    Mz = Ff * a - Fr * b                              # I16 yaw moment
    return dict(ay=ay, Mz=Mz, Ff=Ff, Fr=Fr, af=np.degrees(af), ar=np.degrees(ar), r=r, resid=resid)


def colour(v, vmax, pos=(30, 60, 230), neg=(220, 40, 40)):
    if vmax == 0 or v == 0:
        return "rgb(0,0,0)"
    k = 0.3 + 0.7 * min(abs(v) / vmax, 1)
    base = pos if v > 0 else neg
    return "rgb(%d,%d,%d)" % tuple(int(255 - (255 - c) * k) for c in base)


def find_max_ay(P, L, veh, d_lim, b_lim, n=41, zooms=5):
    """Highest ay anywhere inside the delta/beta range (any yaw moment)."""
    d_lo, d_hi, b_lo, b_hi = -d_lim, d_lim, -b_lim, b_lim
    for _ in range(zooms):
        D, B = np.meshgrid(np.linspace(d_lo, d_hi, n), np.linspace(b_lo, b_hi, n))
        s = solve_ymd(P, L, veh, D, B)
        i = np.unravel_index(np.argmax(s["ay"]), s["ay"].shape)
        dd, bb = D[i], B[i]
        wd, wb = (d_hi - d_lo) * 4 / (n - 1), (b_hi - b_lo) * 4 / (n - 1)
        d_lo, d_hi = max(dd - wd, -d_lim), min(dd + wd, d_lim)
        b_lo, b_hi = max(bb - wb, -b_lim), min(bb + wb, b_lim)
    return dict(ay=float(s["ay"][i]), d=float(dd), b=float(bb), Mz=float(s["Mz"][i]))


def find_max_trim(P, L, veh, d_lim, b_lim, n=41, zooms=6):
    """Highest ay where the car is in balance (yaw moment Mz = 0)."""
    d_lo, d_hi = -d_lim, d_lim
    best = None
    for _ in range(zooms):
        dl = np.linspace(d_lo, d_hi, n)
        bl = np.linspace(-b_lim, b_lim, 4 * n)
        s = solve_ymd(P, L, veh, dl[:, None], bl[None, :])
        mz, ay = s["Mz"], s["ay"]
        cross = np.sign(mz[:, :-1]) * np.sign(mz[:, 1:]) <= 0          # Mz changes sign between two betas
        t = np.where(cross, mz[:, :-1] / np.where(mz[:, :-1] == mz[:, 1:], 1, mz[:, :-1] - mz[:, 1:]), 0)
        ay_c = np.where(cross, ay[:, :-1] + t * (ay[:, 1:] - ay[:, :-1]), -np.inf)
        b_c = bl[:-1] + t * (bl[1] - bl[0])
        if not np.isfinite(ay_c).any():
            return best
        i = np.unravel_index(np.argmax(ay_c), ay_c.shape)
        best = dict(ay=float(ay_c[i]), d=float(dl[i[0]]), b=float(b_c[i]))
        w = (d_hi - d_lo) * 4 / (n - 1)
        d_lo, d_hi = max(best["d"] - w, -d_lim), min(best["d"] + w, d_lim)
    return best


def at_edge(x, lim):
    return abs(abs(x) - lim) < 1e-6 * max(lim, 1)


# -------------------------------------------------------------
# 4. The app screen
# -------------------------------------------------------------
st.set_page_config(page_title="Yaw Moment Diagram", layout="wide")
st.title("Yaw Moment Diagram")

tir_file = st.sidebar.file_uploader("Load tyre file (.tir or .txt)", type=["tir", "txt"])
if tir_file is None:
    st.info("Load your .tir file in the left panel to start.")
    st.stop()
P = read_tir(tir_file.read().decode("utf-8", errors="ignore"))
st.sidebar.success(f"Loaded {len(P)} parameters  |  FNOMIN = {P.get('FNOMIN', 0):.0f} N")

st.sidebar.header("Vehicle")
m = st.sidebar.number_input("Mass incl. driver (kg)", 50.0, 2000.0, 316.0, 1.0)
wb = st.sidebar.number_input("Wheelbase (mm)", 500.0, 5000.0, 1580.0, 1.0)
a_mm = st.sidebar.number_input("CG to front axle, a (mm)", 1.0, 5000.0, 774.2, 0.1)
v_kph = st.sidebar.number_input("Speed (km/h)", 5.0, 300.0, 80.0, 1.0)
camf = st.sidebar.number_input("Front camber (deg)", -6.0, 6.0, 0.0, 0.1)
camr = st.sidebar.number_input("Rear camber (deg)", -6.0, 6.0, 0.0, 0.1)

st.sidebar.header("Tyre settings")
fz0 = st.sidebar.number_input("Nominal load Fz0 (N)", 50.0, 10000.0, float(P.get("FNOMIN", 1000)), 1.0)
p_kpa = st.sidebar.number_input("Pressure (kPa)", 30.0, 200.0, P.get("INFLPRES", P.get("NOMPRES", 83000)) / 1000, 1.0)
lmu = st.sidebar.number_input("λμ  peak friction scaling", 0.1, 2.0, 0.7, 0.05)
lky = st.sidebar.number_input("λKyα  cornering stiffness scaling", 0.1, 3.0, 1.0, 0.05)

st.sidebar.header("Diagram range")
d_max = st.sidebar.number_input("Steer δ: ± max (deg)", 0.5, 30.0, 7.0, 0.5)
d_step = st.sidebar.number_input("Steer δ: step between lines (deg)", 0.1, 10.0, 1.0, 0.1)
b_max = st.sidebar.number_input("Body slip β: ± max (deg)", 0.5, 30.0, 6.0, 0.5)
b_step = st.sidebar.number_input("Body slip β: step between lines (deg)", 0.1, 10.0, 1.0, 0.1)
fine = st.sidebar.slider("Curve smoothness (points per line)", 20, 200, 80, 10)

# ---- set up tyre parameters and vehicle ----
P = dict(P)
P["FNOMIN"] = fz0
L = {k: 1.0 for k in ["LMU", "LKX", "LKY", "LC", "LE", "LHX", "LHY", "LVX", "LVY", "LKYC",
                      "LKZC", "LTR", "LRES", "LXAL", "LYKA", "LVYKA", "LS", "LMX"]}
L["LMU"] = lmu
L["LKY"] = lky
veh = dict(m=m, a=a_mm / 1000, b=(wb - a_mm) / 1000, V=v_kph / 3.6, camf=camf, camr=camr, p=p_kpa)

deltas = np.round(np.arange(-d_max, d_max + d_step / 2, d_step), 6)
betas = np.round(np.arange(-b_max, b_max + b_step / 2, b_step), 6)
b_fine = np.linspace(-b_max, b_max, fine)
d_fine = np.linspace(-d_max, d_max, fine)

# ---- compute ----
iso_d = solve_ymd(P, L, veh, deltas[:, None], b_fine[None, :])   # one row per δ line
iso_b = solve_ymd(P, L, veh, d_fine[None, :], betas[:, None])    # one row per β line
worst = max(iso_d["resid"].max(), iso_b["resid"].max())

# ---- search for maximum ay ----
@st.cache_data(show_spinner=False)
def search_max(P, L, veh, d_max, b_max):        # remembered, so it only re-runs when inputs change
    return find_max_ay(P, L, veh, d_max, b_max), find_max_trim(P, L, veh, d_max, b_max)


with st.spinner("Searching for maximum lateral acceleration..."):
    mx, tr = search_max(P, L, veh, d_max, b_max)

# ---- diagram ----
fig = go.Figure()
for i, bv in enumerate(betas):
    fig.add_trace(go.Scatter(x=iso_b["ay"][i], y=iso_b["Mz"][i], mode="lines", name=f"β = {bv:g}°",
                             line=dict(color="rgba(120,120,120,0.7)", width=1, dash="dash"), legendgroup="beta",
                             customdata=np.c_[d_fine, np.full(fine, bv)],
                             hovertemplate="δ = %{customdata[0]:.2f}°<br>β = %{customdata[1]:.2f}°<br>ay = %{x:.3f} g<br>Mz = %{y:.0f} Nm<extra></extra>"))
for i, dv in enumerate(deltas):
    fig.add_trace(go.Scatter(x=iso_d["ay"][i], y=iso_d["Mz"][i], mode="lines", name=f"δ = {dv:g}°",
                             line=dict(color=colour(dv, d_max), width=2.2 if dv == 0 else 1.8), legendgroup="delta",
                             customdata=np.c_[np.full(fine, dv), b_fine],
                             hovertemplate="δ = %{customdata[0]:.2f}°<br>β = %{customdata[1]:.2f}°<br>ay = %{x:.3f} g<br>Mz = %{y:.0f} Nm<extra></extra>"))
fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"))
fig.add_vline(x=0, line=dict(color="black", width=1, dash="dot"))
fig.update_layout(height=700, xaxis_title="Lateral acceleration ay (g)", yaxis_title="Yaw moment Mz (Nm)",
                  title=f"Yaw Moment Diagram at {v_kph:g} km/h  —  coloured: constant steer δ,  grey dashed: constant body slip β",
                  margin=dict(l=10, r=10, t=60, b=10))
fig.add_trace(go.Scatter(x=[mx["ay"]], y=[mx["Mz"]], mode="markers", name="Max ay",
                         marker=dict(symbol="star", size=16, color="gold", line=dict(color="black", width=1))))
if tr:
    fig.add_trace(go.Scatter(x=[tr["ay"]], y=[0], mode="markers", name="Max trimmed ay (Mz = 0)",
                             marker=dict(symbol="diamond", size=12, color="limegreen", line=dict(color="black", width=1))))
st.plotly_chart(fig, use_container_width=True)

# ---- key numbers ----
h = 0.1
stab = (solve_ymd(P, L, veh, 0.0, h)["Mz"] - solve_ymd(P, L, veh, 0.0, -h)["Mz"]) / (2 * h)
ctrl = (solve_ymd(P, L, veh, h, 0.0)["Mz"] - solve_ymd(P, L, veh, -h, 0.0)["Mz"]) / (2 * h)

st.subheader("Maximum lateral acceleration")
c = st.columns(4)
c[0].metric("Max ay (g)", f"{mx['ay']:.3f}")
c[1].metric("Steer δ for max ay (deg)", f"{mx['d']:.2f}")
c[2].metric("Body slip β for max ay (deg)", f"{mx['b']:.2f}")
c[3].metric("Yaw moment at max ay (Nm)", f"{mx['Mz']:.0f}")
c = st.columns(4)
if tr:
    c[0].metric("Max trimmed ay, Mz = 0 (g)", f"{tr['ay']:.3f}")
    c[1].metric("Steer δ for trimmed max (deg)", f"{tr['d']:.2f}")
    c[2].metric("Body slip β for trimmed max (deg)", f"{tr['b']:.2f}")
else:
    c[0].metric("Max trimmed ay, Mz = 0 (g)", "none in range")
c[3].metric("Corner radius at max ay (m)", f"{veh['V'] ** 2 / (mx['ay'] * 9.81):.1f}" if mx["ay"] > 0 else "—")
c = st.columns(4)
c[0].metric("Stability dMz/dβ at origin (Nm/deg)", f"{float(stab):.0f}")
c[1].metric("Control dMz/dδ at origin (Nm/deg)", f"{float(ctrl):.0f}")
if at_edge(mx["d"], d_max) or at_edge(mx["b"], b_max) or (tr and at_edge(tr["d"], d_max)):
    st.warning("The maximum was found at the edge of your δ or β range, so the real maximum may be higher. "
               "Increase 'Steer δ: ± max' or 'Body slip β: ± max' in the left panel (needed especially at low speed).")
st.caption(f"Solver check: largest mismatch between guessed and resulting ay = {worst:.1e} g (should be tiny). "
           "Max ay is searched over the whole δ/β range in the left panel, not just the plotted lines.")

# ---- single point check (compare with Excel 'Yaw Moment DIagram') ----
st.subheader("Single point check")
cc = st.columns(2)
d1 = cc[0].number_input("Steer δ (deg)", -30.0, 30.0, 4.0, 0.5)
b1 = cc[1].number_input("Body slip β (deg)", -30.0, 30.0, -4.0, 0.5)
s = solve_ymd(P, L, veh, d1, b1)
c = st.columns(4)
c[0].metric("Lateral acceleration (g)", f"{float(s['ay']):.4f}")
c[1].metric("Yaw moment (Nm)", f"{float(s['Mz']):.2f}")
c[2].metric("Front axle force (N)", f"{float(s['Ff']):.1f}")
c[3].metric("Rear axle force (N)", f"{float(s['Fr']):.1f}")
c = st.columns(4)
c[0].metric("Front slip angle (deg)", f"{float(s['af']):.3f}")
c[1].metric("Rear slip angle (deg)", f"{float(s['ar']):.3f}")
c[2].metric("Yaw velocity (rad/s)", f"{float(s['r']):.4f}")
c[3].metric("Corner radius (m)", f"{veh['V'] ** 2 / (float(s['ay']) * 9.81):.1f}" if float(s["ay"]) != 0 else "—")

# ---- download grid ----
grid = solve_ymd(P, L, veh, deltas[:, None], betas[None, :])
rows = ["delta_deg,beta_deg,ay_g,Mz_Nm,front_slip_deg,rear_slip_deg"]
for i, dv in enumerate(deltas):
    for j, bv in enumerate(betas):
        rows.append(f"{dv:g},{bv:g},{grid['ay'][i, j]:.5f},{grid['Mz'][i, j]:.3f},{grid['af'][i, j]:.4f},{grid['ar'][i, j]:.4f}")
st.download_button("Download grid points as CSV (opens in Excel)", "\n".join(rows), file_name="ymd_grid.csv")