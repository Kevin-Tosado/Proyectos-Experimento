import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import io
import zipfile  # Needed to compress and download all plots at once

# =========================================================================
# ACCURATE CORE LOGIC (MATLAB REPLICATION)
# =========================================================================

def calculate_fisher_scores(X, y):
    """Exact replication of the Fisher Score logic."""
    idx0, idx1 = (y == 0), (y == 1)
    if not (np.any(idx0) and np.any(idx1)):
        return np.zeros(X.shape[1])
    m0, m1 = np.mean(X[idx0], axis=0), np.mean(X[idx1], axis=0)
    mt = np.mean(X, axis=0)
    v0, v1 = np.var(X[idx0], axis=0), np.var(X[idx1], axis=0)
    return ((m0 - mt)**2 + (m1 - mt)**2) / (v0 + v1 + 1e-6)

def run_loocv_sweep(X_full, y, k_range, kernel, strategy, C_f=0.1, g_f=0.001):
    """Performs true LOOCV with Nested Tuning for parameter selection."""
    n_samples = len(y)
    acc_results = []
    C_grid = [0.01, 0.1, 1, 10, 100]
    g_grid = [0.001, 0.01, 0.1, 1, 10]

    for k in k_range:
        preds = np.zeros(n_samples)
        for fold in range(n_samples):
            X_tr, y_tr = np.delete(X_full, fold, axis=0), np.delete(y, fold)
            X_te = X_full[fold, :].reshape(1, -1)
            
            fs = calculate_fisher_scores(X_tr, y_tr)
            sel = np.argsort(fs)[::-1][:k]
            
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr[:, sel])
            X_te_s = sc.transform(X_te[:, sel])
            
            best_C, best_g = C_f, g_f
            
            if strategy == 'N':
                best_inner_acc = -1
                n_inner = len(y_tr)
                
                for cc in C_grid:
                    for gg in g_grid:
                        if kernel == 'linear' and gg != g_grid[0]: continue
                        
                        inner_preds = np.zeros(n_inner)
                        for ifold in range(n_inner):
                            Xi_tr, yi_tr = np.delete(X_tr_s, ifold, axis=0), np.delete(y_tr, ifold)
                            Xi_te = X_tr_s[ifold, :].reshape(1, -1)
                            
                            m_in = SVC(kernel=kernel, C=cc, gamma=gg).fit(Xi_tr, yi_tr)
                            inner_preds[ifold] = m_in.predict(Xi_te).item()
                        
                        inner_acc = np.mean(inner_preds == y_tr)
                        if inner_acc > best_inner_acc:
                            best_inner_acc, best_C, best_g = inner_acc, cc, gg
            
            final_mdl = SVC(kernel=kernel, C=best_C, gamma=best_g).fit(X_tr_s, y_tr)
            preds[fold] = final_mdl.predict(X_te_s).item()
            
        acc_results.append(np.mean(preds == y))
    return np.array(acc_results)

# =========================================================================
# PROFESSIONAL UI & PLOTTING
# =========================================================================

st.set_page_config(page_title="SVM Research Suite", layout="wide")
st.title("📊 SVM Experiment Battery")

# --- INITIAL EXPORT INITIALIZATION IN SIDEBAR ---
st.sidebar.subheader("⚙️ Export Configuration")
export_option = st.sidebar.radio(
    "How would you like to handle the generated plots?",
    ["Download all together in a single .ZIP file", "View and save individually at the bottom"]
)

uploaded_file = st.sidebar.file_uploader("Upload Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    y_L1 = df.iloc[:, 0].values
    X = df.iloc[:, 2:].values
    y_L2 = np.array([0,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1])
    k_range = np.arange(2, 32, 2)

    if st.sidebar.button("🚀 Run Full"):
        st.warning("Running true Nested LOOCV. This will take a few minutes but the results will be CORRECT.")
        
        results = {}
        exps = [
            ('L1_Lin_C', y_L1, 'linear', 'C'), ('L2_Lin_C', y_L2, 'linear', 'C'),
            ('L1_Lin_N', y_L1, 'linear', 'N'), ('L2_Lin_N', y_L2, 'linear', 'N'),
            ('L1_RBF_C', y_L1, 'rbf', 'C'),    ('L2_RBF_C', y_L2, 'rbf', 'C'),
            ('L1_RBF_N', y_L1, 'rbf', 'N'),    ('L2_RBF_N', y_L2, 'rbf', 'N')
        ]
        
        pbar = st.progress(0)
        for i, (name, y_lab, kern, strat) in enumerate(exps):
            st.write(f"Processing {name}...")
            results[name] = run_loocv_sweep(X, y_lab, k_range, kern, strat)
            pbar.progress((i + 1) / len(exps))

        # Store calculations in Streamlit Session State to keep them available
        st.session_state['results'] = results
        st.session_state['ready'] = True

    # Check if the calculations are ready to render
    if 'ready' in st.session_state and st.session_state['ready']:
        results = st.session_state['results']

        # --- SUBPLOT RENDERING FUNCTION ---
        def draw_subplot(d1, d2, l1, l2, title):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(k_range, d1, '-o', label=l1, color='#0072BD', linewidth=2, markersize=6)
            ax.plot(k_range, d2, '-s', label=l2, color='#D95319', linewidth=2, markersize=6)
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.set_xlabel("k (Features)")
            ax.set_ylabel("Accuracy")
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            ax.legend()
            return fig

        # Pre-generate all figures and keep them in memory
        figures = {
            '1. L1 vs L2 (Lin Fixed)': draw_subplot(results['L1_Lin_C'], results['L2_Lin_C'], 'L1', 'L2', '1. L1 vs L2 (Lin Fixed)'),
            '2. L1 vs L2 (Lin Nested)': draw_subplot(results['L1_Lin_N'], results['L2_Lin_N'], 'L1', 'L2', '2. L1 vs L2 (Lin Nested)'),
            '3. Lin vs RBF (L1 Fixed)': draw_subplot(results['L1_Lin_C'], results['L1_RBF_C'], 'Lin', 'RBF', '3. Lin vs RBF (L1 Fixed)'),
            '4. Lin vs RBF (L1 Nested)': draw_subplot(results['L1_Lin_N'], results['L1_RBF_N'], 'Lin', 'RBF', '4. Lin vs RBF (L1 Nested)'),
            '5. Lin vs RBF (L2 Fixed)': draw_subplot(results['L2_Lin_C'], results['L2_RBF_C'], 'Lin', 'RBF', '5. Lin vs RBF (L2 Fixed)'),
            '6. Lin vs RBF (L2 Nested)': draw_subplot(results['L2_Lin_N'], results['L2_RBF_N'], 'Lin', 'RBF', '6. Lin vs RBF (L2 Nested)'),
            '7. Fixed vs Nested (L1 Lin)': draw_subplot(results['L1_Lin_C'], results['L1_Lin_N'], 'Fixed', 'Nested', '7. Fixed vs Nested (L1 Lin)'),
            '8. Fixed vs Nested (L2 Lin)': draw_subplot(results['L2_Lin_C'], results['L2_Lin_N'], 'Fixed', 'Nested', '8. Fixed vs Nested (L2 Lin)')
        }

        # --- GLOBAL ZIP DOWNLOAD HANDLE ---
        if export_option == "Download all together in a single .ZIP file":
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for name, fig in figures.items():
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format="png", bbox_inches='tight')
                    filename = name.lower().replace(" ", "_").replace(".", "").replace("(", "").replace(")", "") + ".png"
                    zip_file.writestr(filename, img_buf.getvalue())
            
            st.success("✅ Calculations finished! You can batch download your plots below.")
            st.download_button(
                label="📦 Download All Plots (.ZIP)",
                data=zip_buffer.getvalue(),
                file_name="svm_experiment_plots.zip",
                mime="application/zip"
            )
            st.markdown("---")

        # --- GRID VIEW PANEL ---
        st.subheader("📊 Experiment Summary Panel")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(figures['1. L1 vs L2 (Lin Fixed)'])
            st.pyplot(figures['3. Lin vs RBF (L1 Fixed)'])
            st.pyplot(figures['5. Lin vs RBF (L2 Fixed)'])
            st.pyplot(figures['7. Fixed vs Nested (L1 Lin)'])
        with c2:
            st.pyplot(figures['2. L1 vs L2 (Lin Nested)'])
            st.pyplot(figures['4. Lin vs RBF (L1 Nested)'])
            st.pyplot(figures['6. Lin vs RBF (L2 Nested)'])
            st.pyplot(figures['8. Fixed vs Nested (L2 Lin)'])

        # --- MATLAB-STYLE INDIVIDUAL PLOT INSPECTOR ---
        st.markdown("---")
        st.subheader("🔍 Individual Plot Inspector (MATLAB Style)")
        st.write("Select a specific experiment plot from the dropdown menu to inspect it in high resolution and export it separately:")
        
        selected_plot = st.selectbox("Choose a plot to inspect:", list(figures.keys()))
        
        if selected_plot:
            fig_individual = figures[selected_plot]
            st.pyplot(fig_individual)
            
            # Save button for the active individual plot
            buf_ind = io.BytesIO()
            fig_individual.savefig(buf_ind, format="png", bbox_inches='tight')
            st.download_button(
                label=f"💾 Save '{selected_plot}' as PNG",
                data=buf_ind.getvalue(),
                file_name=f"{selected_plot.lower().replace(' ', '_').replace('.', '')}.png",
                mime="image/png"
            )
