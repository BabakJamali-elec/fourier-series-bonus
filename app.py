import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------- Page config ----------
st.set_page_config(
    page_title="Fourier Series Calculator",
    page_icon="📈",
    layout="centered"
)

st.title("Fourier Series Calculator")
st.markdown("""
Compute the Fourier series of a set of arbitrary points up to order **N**
using the **least squares** method.
""")

st.divider()

# ---------- Points input ----------
st.subheader("1) Enter your points")
st.markdown("""
Write each point on a separate line as `x,y`. Example:
0,1
1,2
2,1
3,0
4,-1
5,0
6,1
""")

default_points = """0,1
1,2
2,1
3,0
4,-1
5,0
6,1"""

points_text = st.text_area(
    "Points (x,y) — one point per line",
    value=default_points,
    height=200,
    key="points"
)

# ---------- N input ----------
st.subheader("2) Number of Fourier terms (N)")

N = st.slider(
    "Choose N",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

st.divider()

# ---------- Compute ----------
if st.button(" Compute & Plot", use_container_width=True):

    # --- Parse points ---
    try:
        lines = [l.strip() for l in points_text.strip().split("\n") if l.strip()]
        pts = []
        for line in lines:
            parts = line.replace(";", ",").split(",")
            if len(parts) != 2:
                st.error(f"Invalid line: '{line}' — must be in x,y format.")
                st.stop()
            pts.append([float(parts[0]), float(parts[1])])
        pts = np.array(pts)
    except Exception as e:
        st.error(f"Error while parsing points: {e}")
        st.stop()

    if len(pts) < 3:
        st.error("At least 3 points are required.")
        st.stop()

    pts = pts[np.argsort(pts[:, 0])]  # sort by x
    x = pts[:, 0]
    y = pts[:, 1]

    T = x[-1] - x[0]
    if T <= 0:
        st.error("x values must be distinct.")
        st.stop()

    w0 = 2 * np.pi / T

    # --- Build basis matrix ---
    # f(x) ≈ a0/2 + Σ [a_n cos(n w0 x) + b_n sin(n w0 x)]
    cols = [np.ones_like(x)]
    for n in range(1, N + 1):
        cols.append(np.cos(n * w0 * x))
        cols.append(np.sin(n * w0 * x))

    A = np.column_stack(cols)

    # --- Least squares solve ---
    try:
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception as e:
        st.error(f"Error while solving: {e}")
        st.stop()

    # --- Reconstruct on fine grid ---
    x_fine = np.linspace(x[0], x[-1], 800)
    cols_fine = [np.ones_like(x_fine)]
    for n in range(1, N + 1):
        cols_fine.append(np.cos(n * w0 * x_fine))
        cols_fine.append(np.sin(n * w0 * x_fine))
    A_fine = np.column_stack(cols_fine)
    y_fine = A_fine @ coeffs

    # --- Plot ---
    st.subheader("📊 Fourier Approximation")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, color="red", s=60, zorder=5,
               label="Input points", edgecolors="black")
    ax.plot(x_fine, y_fine, color="royalblue", linewidth=2,
            label=f"Fourier series (N={N})")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Fourier series approximation (least squares)")
    st.pyplot(fig)

    # --- Coefficients ---
    st.subheader("🧮 Fourier Coefficients")
    st.markdown(f"**Period:** T = {T:.4f}  |  **Fundamental frequency:** ω₀ = {w0:.4f}")

    coeff_data = {"Coefficient": ["a₀/2"], "Value": [f"{coeffs[0]:.6f}"]}
    for n in range(1, N + 1):
        coeff_data["Coefficient"].append(f"a_{n}")
        coeff_data["Value"].append(f"{coeffs[2*n-1]:.6f}")
        coeff_data["Coefficient"].append(f"b_{n}")
        coeff_data["Value"].append(f"{coeffs[2*n]:.6f}")

    st.dataframe(coeff_data, use_container_width=True, hide_index=True)

    # --- Error metric ---
    y_approx = A @ coeffs
    rmse = np.sqrt(np.mean((y - y_approx) ** 2))
    st.metric("RMSE (Root Mean Square Error)", f"{rmse:.6f}")

    # --- Final formula ---
    st.subheader("📝 Approximation Formula")
    formula = f"f(x) ≈ {coeffs[0]:.4f}"
    for n in range(1, N + 1):
        a = coeffs[2*n-1]
        b = coeffs[2*n]
        sign_a = "+" if a >= 0 else "-"
        sign_b = "+" if b >= 0 else "-"
        formula += f" {sign_a} {abs(a):.4f}*cos({n}*w0*x) {sign_b} {abs(b):.4f}*sin({n}*w0*x)"
    st.code(formula, language="text")

else:
    st.info("👆 Set your points and N, then click 'Compute & Plot'.")
