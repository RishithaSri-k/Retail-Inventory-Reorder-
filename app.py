"""
Retail Inventory Reorder Assistant - Streamlit Dashboard
Hackathon MVP
Tech Stack: Streamlit, Plotly, Pandas, NumPy, Scikit-learn, TensorFlow/Keras
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

import importlib
import pipeline
importlib.reload(pipeline)

# Page configuration
st.set_page_config(
    page_title="Retail Inventory Reorder Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, hackathon-ready aesthetics
st.markdown("""
<style>
    /* Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Action Badges */
    .badge-reorder {
        background-color: #ef4444;
        color: white;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        letter-spacing: 0.05em;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
    }
    .badge-hold {
        background-color: #10b981;
        color: white;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        letter-spacing: 0.05em;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
    }
    .badge-special {
        background-color: #f59e0b;
        color: #1e1e2f;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
        margin-left: 8px;
    }
    .callout-box {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# CACHED DATA LOADING & MODEL TRAINING
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_dataset():
    """Loads filtered data and active holidays."""
    all_families = ['GROCERY I', 'BEVERAGES', 'CLEANING', 'DAIRY', 'HARDWARE']
    df, holidays = pipeline.load_dataset(
        train_path='train.csv',
        holidays_path='holidays_events.csv',
        store_nbr=1,
        families=all_families,
        start_date='2016-01-01'
    )
    return df, holidays


@st.cache_data(show_spinner=False)
def process_and_train_family(family_name, _df, holidays):
    """
    Processes time-series, trains MA baseline and ANN model, evaluates metrics.
    Cached so model only trains once per product family.
    """
    sub_df = _df[_df['family'] == family_name].copy()
    demand_stats = pipeline.compute_demand_analysis(sub_df)
    df_feat = pipeline.prepare_time_series(sub_df, holidays)

    train_df, test_df = pipeline.chronological_split(df_feat, test_days=28)

    # Moving Average Baseline
    ma_pred = pipeline.compute_moving_average_baseline(train_df, test_df, window=7)
    ma_metrics = pipeline.evaluate_forecast(test_df['sales'], ma_pred)

    # ANN Forecasting Model
    feat_cols = [c for c in df_feat.columns if c not in ['date', 'sales', 'store_nbr', 'family']]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feat_cols])
    y_train = train_df['sales'].values
    X_test = scaler.transform(test_df[feat_cols])
    y_test = test_df['sales'].values

    model, history = pipeline.build_and_train_ann(
        X_train, y_train, X_test, y_test, epochs=25, batch_size=32
    )

    ann_pred = model.predict(X_test, verbose=0).flatten()
    ann_metrics = pipeline.evaluate_forecast(y_test, ann_pred)

    # Generate 7-day recursive forecast
    recent_data = df_feat.tail(30)
    forecast_7d = pipeline.forecast_next_7_days(
        model, scaler, feat_cols, recent_data, holiday_dates=holidays
    )

    # Evaluation period dates & predictions
    eval_df = pd.DataFrame({
        'date': test_df['date'],
        'actual': test_df['sales'],
        'ma_pred': ma_pred,
        'ann_pred': ann_pred
    })

    return {
        'demand_stats': demand_stats,
        'df_feat': df_feat,
        'train_df': train_df,
        'test_df': test_df,
        'ma_metrics': ma_metrics,
        'ann_metrics': ann_metrics,
        'forecast_7d': forecast_7d,
        'eval_df': eval_df,
        'historical_std': float(train_df['sales'].std())
    }


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Control Panel")

# 1. Store selector
store_selection = st.sidebar.selectbox(
    "Store Location",
    options=["Store 1 (Quito - Central Hub)"],
    index=0,
    help="Store 1 from Kaggle Favorita dataset"
)

# 2. Product selector (Standard, Slow-Moving, and 3 Demo Cold-Start SKUs)
product_options = [
    "GROCERY I (High-Volume Staple)",
    "BEVERAGES (High Turnover & Promo-Sensitive)",
    "CLEANING (Household Staple)",
    "DAIRY (Perishable Staple)",
    "HARDWARE (Slow-Moving / Intermittent)",
    "[NEW] Artisanal Cold-Brew (Demo Cold-Start SKU)",
    "[NEW] Organic Protein Bar (Demo Cold-Start SKU)",
    "[NEW] Premium Energy Drink (Demo Cold-Start SKU)"
]

product_selection = st.sidebar.selectbox(
    "Product / Family Selector",
    options=product_options,
    index=0
)

# Identify selected family type
is_cold_start = "[NEW]" in product_selection
is_slow_moving = "HARDWARE" in product_selection

if is_cold_start:
    cold_sku_info = pipeline.COLD_START_PRODUCTS[product_selection]
    family_key = "COLD_START"
    current_default_stock = cold_sku_info["default_stock"]
else:
    cold_sku_info = None
    family_key = product_selection.split(" (")[0]
    default_stocks = {
        'GROCERY I': 2500,
        'BEVERAGES': 1800,
        'CLEANING': 800,
        'DAIRY': 650,
        'HARDWARE': 6
    }
    current_default_stock = default_stocks.get(family_key, 1000)

# 3. Forecast Horizon
forecast_horizon = st.sidebar.number_input(
    "Forecast Horizon (Days)",
    min_value=7,
    max_value=7,
    value=7,
    disabled=True,
    help="Fixed 7-day forward horizon per requirements"
)

# 4. Inventory Parameters with Dynamic Defaults
current_stock_input = st.sidebar.number_input(
    "Current On-Hand Stock (Units)",
    min_value=0,
    max_value=100000,
    value=current_default_stock,
    step=10 if not (is_slow_moving or is_cold_start) else 1,
    help="Current stock inventory available in warehouse"
)

lead_time_days = st.sidebar.slider(
    "Supplier Lead Time (Days)",
    min_value=1,
    max_value=7,
    value=2,
    help="Days required for replenishment delivery"
)

service_level_option = st.sidebar.selectbox(
    "Target Service Level (Z)",
    options=["95% Service Level (Z = 1.65)", "99% Service Level (Z = 2.33)", "90% Service Level (Z = 1.28)"],
    index=0
)
service_level_z = 1.65 if "1.65" in service_level_option else (2.33 if "2.33" in service_level_option else 1.28)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Economic Cost Parameters")

stockout_cost_unit = st.sidebar.number_input(
    "Stock-out Cost per Unit ($)",
    min_value=1.0,
    max_value=200.0,
    value=15.0,
    step=1.0,
    help="Penalized cost of unmet customer demand and lost margin"
)

holding_cost_unit = st.sidebar.number_input(
    "Holding Cost per Unit ($)",
    min_value=0.1,
    max_value=50.0,
    value=2.0,
    step=0.5,
    help="Carrying cost of excess inventory, storage, and shrinkage"
)


# ---------------------------------------------------------
# DATA & PIPELINE EXECUTION
# ---------------------------------------------------------
with st.spinner("Loading dataset and initializing forecasting engine..."):
    df_raw, holidays = get_dataset()

if not is_cold_start:
    with st.spinner(f"Computing demand features and training ANN for {family_key}..."):
        res = process_and_train_family(family_key, df_raw, holidays)

    forecast_7d = res['forecast_7d']
    demand_stats = res['demand_stats']
    historical_std = res['historical_std']
    df_feat = res['df_feat']
    eval_df = res['eval_df']
    ma_metrics = res['ma_metrics']
    ann_metrics = res['ann_metrics']

    # Compute Reorder Recommendation with Cost-Aware Policy
    reorder_res = pipeline.calculate_reorder_recommendation(
        current_stock=current_stock_input,
        forecast_7d_df=forecast_7d,
        lead_time_days=lead_time_days,
        service_level_z=service_level_z,
        stockout_cost_per_unit=stockout_cost_unit,
        holding_cost_per_unit=holding_cost_unit,
        is_slow_moving=is_slow_moving,
        is_new_or_insufficient=False,
        historical_std=historical_std
    )
else:
    # Dynamic Cold-Start Demo SKU Handling
    sample_obs = cold_sku_info['observed_sales']
    forecast_7d, est_mean, est_std = pipeline.handle_new_or_insufficient_product(
        new_product_sales_sample=sample_obs,
        category_mean_daily=cold_sku_info['category_mean'],
        category_std_daily=cold_sku_info['category_std']
    )
    obs_mean = float(np.mean(sample_obs))
    obs_std = float(np.std(sample_obs)) if len(sample_obs) > 1 else (obs_mean * 0.25)
    demand_stats = {
        'total_records': len(sample_obs),
        'total_sales': sum(sample_obs),
        'mean_daily_sales': round(obs_mean, 2),
        'std_daily_sales': round(obs_std, 2),
        'median_sales': round(float(np.median(sample_obs)), 2),
        'min_sales': min(sample_obs),
        'max_sales': max(sample_obs),
        'zero_sales_ratio': round(float((np.array(sample_obs) == 0).mean() * 100), 2),
        'coefficient_of_variation': round(float(obs_std / obs_mean), 2) if obs_mean > 0 else 0.0,
        'demand_type': f"Synthetic Cold-Start ({cold_sku_info['history_count']}d history)"
    }
    reorder_res = pipeline.calculate_reorder_recommendation(
        current_stock=current_stock_input,
        forecast_7d_df=forecast_7d,
        lead_time_days=lead_time_days,
        service_level_z=service_level_z,
        stockout_cost_per_unit=stockout_cost_unit,
        holding_cost_per_unit=holding_cost_unit,
        is_slow_moving=False,
        is_new_or_insufficient=True,
        historical_std=est_std
    )
    df_feat = None
    eval_df = None
    ma_metrics = None
    ann_metrics = None
    # Simulate policy over available observed pilot history
    cold_policy_sim = pipeline.simulate_reorder_policy_inventory(
        y_true=sample_obs,
        y_pred=[est_mean] * len(sample_obs),
        lead_time_days=lead_time_days,
        stockout_cost_per_unit=stockout_cost_unit,
        holding_cost_per_unit=holding_cost_unit,
        historical_std=est_std,
        service_level_z=service_level_z,
        initial_stock=current_stock_input,
        is_new_or_insufficient=True
    )


# ---------------------------------------------------------
# DASHBOARD HEADER & STATUS BADGES
# ---------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("📦 Retail Inventory Reorder Assistant")
    st.caption("AI-Powered Time-Series Demand Forecasting & Automated Inventory Replenishment")

with header_col2:
    st.write("")
    if is_cold_start:
        st.markdown("<div style='text-align: right;'><span class='badge-special'>⚡ COLD-START PRODUCT</span></div>", unsafe_allow_html=True)
    elif is_slow_moving:
        st.markdown("<div style='text-align: right;'><span class='badge-special'>🐢 SLOW-MOVING SKU</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align: right;'><span style='background:#3b82f6;color:white;padding:4px 10px;border-radius:6px;font-weight:700;font-size:0.8rem;'>ACTIVE TIME-SERIES</span></div>", unsafe_allow_html=True)

st.markdown("---")


# ---------------------------------------------------------
# 2. TOP KPI CARDS
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Current Stock</div>
        <div class="metric-value">{reorder_res['current_stock']:,.0f}</div>
        <div class="metric-sub">{reorder_res['days_of_supply']:.1f} Days of Supply</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">7-Day Forecast</div>
        <div class="metric-value">{reorder_res['total_forecast_7d']:,.0f}</div>
        <div class="metric-sub">Avg {reorder_res['daily_avg_demand']:.1f} u/day</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Reorder Point (ROP)</div>
        <div class="metric-value">{reorder_res['reorder_point']:,.0f}</div>
        <div class="metric-sub">Operational Trigger</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Target Stock</div>
        <div class="metric-value" style="color: #60a5fa;">{reorder_res['cost_aware_target_stock']:,.0f}</div>
        <div class="metric-sub">Cost-Aware Economic</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">Recommended Order</div>
        <div class="metric-value" style="color: {'#f87171' if reorder_res['reorder_needed'] else '#34d399'};">
            {reorder_res['recommended_order_qty']:,.0f}
        </div>
        <div class="metric-sub">Target - Stock</div>
    </div>
    """, unsafe_allow_html=True)

with kpi6:
    badge_html = "<span class='badge-reorder'>🚨 REORDER NOW</span>" if reorder_res['reorder_needed'] else "<span class='badge-hold'>✅ HOLD STOCK</span>"
    st.markdown(f"""
    <div class="metric-container" style="text-align: center;">
        <div class="metric-label">Status & Critical Ratio</div>
        <div style="margin-top: 6px;">{badge_html}</div>
        <div class="metric-sub" style="margin-top: 6px;">CR: {reorder_res['critical_ratio']:.3f} (Z: {reorder_res['z_cost']:.2f})</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# 10. WHY THIS RECOMMENDATION? (TRANSPARENT REASONING)
# ---------------------------------------------------------
model_name = "ANN Neural Network" if not is_cold_start else "Bayesian Shrinkage Fallback Heuristic"
action_sentence = (
    f"Because current stock ({reorder_res['current_stock']:,.0f}) is <b>BELOW</b> the reorder point ({reorder_res['reorder_point']:,.0f}), "
    f"the system recommends ordering <b>{reorder_res['recommended_order_qty']:,.0f} units</b> (Target {reorder_res['cost_aware_target_stock']:,.0f} - Current {reorder_res['current_stock']:,.0f}) to prevent stockouts while balancing holding costs."
    if reorder_res['reorder_needed']
    else
    f"Because current stock ({reorder_res['current_stock']:,.0f}) is <b>ABOVE</b> the reorder point ({reorder_res['reorder_point']:,.0f}), "
    f"existing inventory is sufficient for the next {reorder_res['days_of_supply']:.1f} days. Recommended order is 0 units."
)

notes_str = ("<br>" + "<br>".join([f"• <i>{note}</i>" for note in reorder_res['notes']])) if reorder_res['notes'] else ""

st.markdown(f"""
<div class="callout-box">
    <div style="font-size: 1.05rem; font-weight: 700; color: #60a5fa; margin-bottom: 6px;">
        💡 Transparent Cost-Aware Recommendation Rationale
    </div>
    <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
        • <b>Demand Forecast</b>: {model_name} predicts <b>{reorder_res['total_forecast_7d']:,.1f} units</b> over the next 7 days (daily average: <b>{reorder_res['daily_avg_demand']:,.1f} units/day</b>).<br>
        • <b>Operational Reorder Trigger (ROP)</b>: With a supplier lead time of <b>{lead_time_days} days</b>, anticipated lead-time demand is <b>{reorder_res['lead_time_demand']:,.1f} units</b> and operational safety stock is <b>{reorder_res['safety_stock']:,.1f} units</b>, setting the operational trigger ROP at <b>{reorder_res['reorder_point']:,.1f} units</b>.<br>
        • <b>Cost Model & Critical Ratio</b>: Configured Stock-out Cost = <b>${reorder_res['stockout_cost_per_unit']:.2f}</b> and Holding Cost = <b>${reorder_res['holding_cost_per_unit']:.2f}</b> yield Critical Ratio <i>CR = Cu / (Cu + Co)</i> = <b>{reorder_res['critical_ratio']:.4f}</b> (Optimal Economic Quantile <i>Z_cost</i> = <b>{reorder_res['z_cost']:.2f}</b>).<br>
        • <b>Cost-Aware Target Stock</b>: LTD ({reorder_res['lead_time_demand']:,.1f}) + 7-Day Forecast ({reorder_res['total_forecast_7d']:,.1f}) + Cost-Aware Safety Stock ({reorder_res['cost_aware_safety_stock']:,.1f}) = <b>{reorder_res['cost_aware_target_stock']:,.1f} units</b>.<br>
        • <b>Decision & Action</b>: {action_sentence}{notes_str}
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 8 & 9. COLD-START OR SLOW-MOVING SPECIAL CALLOUTS
# ---------------------------------------------------------
if is_cold_start:
    st.info(f"""
    **⚡ COLD-START PRODUCT FALLBACK ACTIVATION (Synthetic/Demo Cold-Start SKU)**
    - **Product**: {product_selection}
    - **History Available**: {cold_sku_info['history_count']} day(s) of pilot sales observed ({cold_sku_info['observed_sales']} {cold_sku_info['unit']}).
    - **Status**: **COLD-START / INSUFFICIENT HISTORY** (< 30 observations). The deep ANN is NOT trained on this SKU to avoid severe overfitting.
    - **Fallback Method**: **Category Bayesian Shrinkage** (Reference Category: `{cold_sku_info['category_reference']}`, Prior Mean: {cold_sku_info['category_mean']:.1f}, Prior Std: {cold_sku_info['category_std']:.1f}).
    - **7-Day Forecast**: **{reorder_res['total_forecast_7d']:.1f} {cold_sku_info['unit']}** (Daily demand rate: {reorder_res['daily_avg_demand']:.1f} {cold_sku_info['unit']}/day).
    - **Cost-Aware Target Stock**: **{reorder_res['cost_aware_target_stock']:.1f} {cold_sku_info['unit']}** (Includes +35% cold-start uncertainty buffer).
    - **Recommended Order Quantity**: **{reorder_res['recommended_order_qty']:,} {cold_sku_info['unit']}**.
    """)

if is_slow_moving:
    st.warning(f"""
    **🐢 SLOW-MOVING / INTERMITTENT PRODUCT LOGIC ACTIVATION**
    - **Zero-Sales Frequency**: {demand_stats['zero_sales_ratio']:.1f}% of days have zero transactions (Mean sales: {demand_stats['mean_daily_sales']:.2f} units/day).
    - **Enforced Safety Stock Floor**: Standard continuous distribution would round safety stock down toward zero, risking stockouts during sporadic spikes. The pipeline enforces a **minimum 5-unit safety stock floor**.
    """)


# ---------------------------------------------------------
# 3. DEMAND FORECAST CHART (PLOTLY)
# ---------------------------------------------------------
st.markdown("### 📈 Demand Forecast & Time-Series History")

fig = go.Figure()

if not is_cold_start and df_feat is not None:
    # Show recent 60 days of historical actuals
    recent_history = df_feat.tail(60)
    fig.add_trace(go.Scatter(
        x=recent_history['date'],
        y=recent_history['sales'],
        mode='lines',
        name='Historical Actual Sales',
        line=dict(color='#94a3b8', width=2)
    ))

    # Evaluation period overlay (ANN vs MA on test set)
    fig.add_trace(go.Scatter(
        x=eval_df['date'],
        y=eval_df['ma_pred'],
        mode='lines',
        name='Moving Average Baseline (Test Period)',
        line=dict(color='#f59e0b', width=1.5, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=eval_df['date'],
        y=eval_df['ann_pred'],
        mode='lines',
        name='ANN Prediction (Test Period)',
        line=dict(color='#38bdf8', width=2)
    ))

    # Next 7-Day Forecast
    fig.add_trace(go.Scatter(
        x=forecast_7d['date'],
        y=forecast_7d['forecast_sales'],
        mode='lines+markers',
        name='7-Day ANN Forward Forecast',
        line=dict(color='#a855f7', width=3),
        marker=dict(size=8, symbol='circle')
    ))

    # Mark the start of the 7-day forecast horizon
    split_date = forecast_7d['date'].iloc[0]
    fig.add_vline(
        x=split_date,
        line_width=2,
        line_dash="dot",
        line_color="#ec4899",
        annotation_text="7-Day Horizon Starts",
        annotation_position="top left"
    )

else:
    # Cold-Start Chart
    fig.add_trace(go.Scatter(
        x=forecast_7d['date'],
        y=forecast_7d['forecast_sales'],
        mode='lines+markers',
        name='Cold-Start Heuristic 7-Day Forecast',
        line=dict(color='#38bdf8', width=3),
        marker=dict(size=8)
    ))

fig.update_layout(
    template='plotly_dark',
    height=420,
    margin=dict(l=40, r=40, t=30, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title="Date",
    yaxis_title="Units Sold / Demanded",
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# 4. MODEL PERFORMANCE BENCHMARK
# ---------------------------------------------------------
# ---------------------------------------------------------
# 4 & 7. EVALUATION: FORECASTING ACCURACY & INVENTORY ECONOMICS
# ---------------------------------------------------------
st.markdown("---")
st.markdown("## 📊 Evaluation-Period Benchmark & Inventory Cost Comparison")

if not is_cold_start and eval_df is not None:
    # Stateful Reorder Policy Inventory Simulation over 28-day chronological test set
    ma_cost = pipeline.simulate_reorder_policy_inventory(
        y_true=eval_df['actual'],
        y_pred=eval_df['ma_pred'],
        lead_time_days=lead_time_days,
        stockout_cost_per_unit=stockout_cost_unit,
        holding_cost_per_unit=holding_cost_unit,
        historical_std=historical_std,
        service_level_z=service_level_z
    )
    ann_cost = pipeline.simulate_reorder_policy_inventory(
        y_true=eval_df['actual'],
        y_pred=eval_df['ann_pred'],
        lead_time_days=lead_time_days,
        stockout_cost_per_unit=stockout_cost_unit,
        holding_cost_per_unit=holding_cost_unit,
        historical_std=historical_std,
        service_level_z=service_level_z
    )

    eval_len = len(eval_df)
    eval_start_str = eval_df['date'].min().strftime('%Y-%m-%d')
    eval_end_str = eval_df['date'].max().strftime('%Y-%m-%d')
    cost_diff = ma_cost['total_cost'] - ann_cost['total_cost']

    # Overview KPI row for Evaluation-Period Comparison
    ev_kpi1, ev_kpi2, ev_kpi3, ev_kpi4 = st.columns(4)
    with ev_kpi1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Evaluation Period Length</div>
            <div class="metric-value">{eval_len} Days</div>
            <div class="metric-sub">{eval_start_str} to {eval_end_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with ev_kpi2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">MA Policy Stock-out Units</div>
            <div class="metric-value">{ma_cost['total_stockout_units']:,.1f}</div>
            <div class="metric-sub">Total Ordered: {ma_cost['total_units_ordered']:,.0f} u</div>
        </div>
        """, unsafe_allow_html=True)

    with ev_kpi3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">ANN Policy Stock-out Units</div>
            <div class="metric-value">{ann_cost['total_stockout_units']:,.1f}</div>
            <div class="metric-sub">Total Ordered: {ann_cost['total_units_ordered']:,.0f} u</div>
        </div>
        """, unsafe_allow_html=True)

    with ev_kpi4:
        diff_color = "#34d399" if cost_diff > 0 else "#60a5fa"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Policy Cost Difference</div>
            <div class="metric-value" style="color: {diff_color};">${abs(cost_diff):,.2f}</div>
            <div class="metric-sub">{'ANN Policy Saves $' + f'{cost_diff:,.2f}' if cost_diff > 0 else 'MA Policy Lower by $' + f'{abs(cost_diff):,.2f}'}</div>
        </div>
        """, unsafe_allow_html=True)

    # 1. Exact unified comparison table (Task 10 E / Req 6): Forecasting Performance & Inventory Economics
    st.markdown("#### 🏆 Benchmark: Forecasting Performance & Inventory Economics")
    unified_table_data = [
        {
            "Metric": "MAE (Mean Absolute Error)",
            "Moving Average Policy": f"{ma_metrics['mae']:,.2f}",
            "ANN Policy": f"{ann_metrics['mae']:,.2f}"
        },
        {
            "Metric": "RMSE (Root Mean Squared Error)",
            "Moving Average Policy": f"{ma_metrics['rmse']:,.2f}",
            "ANN Policy": f"{ann_metrics['rmse']:,.2f}"
        },
        {
            "Metric": "Stock-out Cost",
            "Moving Average Policy": f"${ma_cost['total_stockout_cost']:,.2f}",
            "ANN Policy": f"${ann_cost['total_stockout_cost']:,.2f}"
        },
        {
            "Metric": "Holding Cost",
            "Moving Average Policy": f"${ma_cost['total_holding_cost']:,.2f}",
            "ANN Policy": f"${ann_cost['total_holding_cost']:,.2f}"
        },
        {
            "Metric": "Total Inventory Cost",
            "Moving Average Policy": f"${ma_cost['total_cost']:,.2f}",
            "ANN Policy": f"${ann_cost['total_cost']:,.2f}"
        }
    ]
    st.table(pd.DataFrame(unified_table_data).set_index("Metric"))

    # 2. Detailed Reorder Policy Simulation Operational Breakdown (Req 2 & 3)
    st.markdown("#### 📦 Stateful Reorder Policy Simulation Tracking")
    op_table_data = [
        {"Operational Metric": "Initial Inventory (Day 0)", "Moving Average Policy": f"{ma_cost['initial_inventory']:,.1f} units", "ANN Policy": f"{ann_cost['initial_inventory']:,.1f} units"},
        {"Operational Metric": "Total Units Ordered", "Moving Average Policy": f"{ma_cost['total_units_ordered']:,.0f} units", "ANN Policy": f"{ann_cost['total_units_ordered']:,.0f} units"},
        {"Operational Metric": "Total Actual Demand", "Moving Average Policy": f"{ma_cost['total_actual_demand']:,.1f} units", "ANN Policy": f"{ann_cost['total_actual_demand']:,.1f} units"},
        {"Operational Metric": "Total Stock-out (Unmet) Units", "Moving Average Policy": f"{ma_cost['total_stockout_units']:,.1f} units", "ANN Policy": f"{ann_cost['total_stockout_units']:,.1f} units"},
        {"Operational Metric": "Total Excess Inventory (Unit-Days)", "Moving Average Policy": f"{ma_cost['total_excess_inventory']:,.1f} units", "ANN Policy": f"{ann_cost['total_excess_inventory']:,.1f} units"},
        {"Operational Metric": "Average Daily On-Hand Inventory", "Moving Average Policy": f"{ma_cost['average_daily_inventory']:,.1f} units/day", "ANN Policy": f"{ann_cost['average_daily_inventory']:,.1f} units/day"},
        {"Operational Metric": "Stockout Days Count", "Moving Average Policy": f"{ma_cost['stockout_days']} / {eval_len} days", "ANN Policy": f"{ann_cost['stockout_days']} / {eval_len} days"}
    ]
    st.table(pd.DataFrame(op_table_data).set_index("Operational Metric"))

    if cost_diff > 0:
        st.success(
            f"💡 **Cost Optimization Result**: Over the {eval_len}-day test period, the **ANN Policy achieved ${cost_diff:,.2f} lower total inventory cost** "
            f"compared to the Moving Average Baseline (${ann_cost['total_cost']:,.2f} vs. ${ma_cost['total_cost']:,.2f}) "
            f"at ${stockout_cost_unit:.2f}/stock-out unit and ${holding_cost_unit:.2f}/holding unit."
        )
    else:
        st.info(
            f"💡 **Cost Optimization Result**: Over the {eval_len}-day test period, the Moving Average policy cost was ${ma_cost['total_cost']:,.2f} "
            f"and the ANN policy cost was ${ann_cost['total_cost']:,.2f} (Delta: ${abs(cost_diff):,.2f}). "
            f"Adjust the stock-out cost (${stockout_cost_unit:.2f}) or holding cost (${holding_cost_unit:.2f}) in the sidebar to simulate different economic penalty regimes."
        )
else:
    # Cold-Start Product Inventory Simulation over observed pilot days (Req 5)
    st.markdown("#### ⚡ Cold-Start Fallback Inventory Simulation (Observed Pilot Period)")
    st.caption(f"Simulating daily inventory policy over the {len(cold_sku_info['observed_sales'])} observed pilot days for {product_selection}.")
    cold_eval_table = [
        {"Cold-Start Metric": "Observed History Available", "Value": f"{cold_sku_info['history_count']} days ({cold_sku_info['observed_sales']} {cold_sku_info['unit']})"},
        {"Cold-Start Metric": "Initial Inventory (Day 0)", "Value": f"{cold_policy_sim['initial_inventory']:,.1f} {cold_sku_info['unit']}"},
        {"Cold-Start Metric": "Total Actual Demand Observed", "Value": f"{cold_policy_sim['total_actual_demand']:,.1f} {cold_sku_info['unit']}"},
        {"Cold-Start Metric": "Total Units Ordered", "Value": f"{cold_policy_sim['total_units_ordered']:,.0f} {cold_sku_info['unit']}"},
        {"Cold-Start Metric": "Total Stock-out Units", "Value": f"{cold_policy_sim['total_stockout_units']:,.1f} {cold_sku_info['unit']}"},
        {"Cold-Start Metric": "Total Excess Inventory (Unit-Days)", "Value": f"{cold_policy_sim['total_excess_inventory']:,.1f} {cold_sku_info['unit']}"},
        {"Cold-Start Metric": "Stock-out Cost", "Value": f"${cold_policy_sim['total_stockout_cost']:,.2f}"},
        {"Cold-Start Metric": "Holding Cost", "Value": f"${cold_policy_sim['total_holding_cost']:,.2f}"},
        {"Cold-Start Metric": "Total Simulated Inventory Cost", "Value": f"${cold_policy_sim['total_cost']:,.2f}"}
    ]
    st.table(pd.DataFrame(cold_eval_table).set_index("Cold-Start Metric"))


# ---------------------------------------------------------
# REQUIREMENT 7: CONCISE METHODOLOGY EXPLANATION SECTION
# ---------------------------------------------------------
st.markdown("---")
st.markdown("## 🧠 End-to-End System Methodology & Reorder Pipeline")
st.markdown("""
<div class="callout-box" style="border-left-color: #10b981;">
    <div style="font-weight: 700; color: #34d399; font-size: 1.05rem; margin-bottom: 8px;">
        Pipeline Workflow (8-Step Optimization Architecture)
    </div>
    <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.7;">
        <b>1. Demand Forecasting:</b> TensorFlow ANN forecasts multi-step forward demand using lag-1, lag-7, lag-14, rolling means, promotional status, and calendar features (or Bayesian shrinkage category fallback for cold-start SKUs).<br>
        <b>2. Uncertainty Estimation:</b> Forecast uncertainty (historical residual & demand standard deviation &sigma;) is projected across supplier lead time: &sigma;<sub>LTD</sub> = &radic;L &times; &sigma;.<br>
        <b>3. Economic Trade-off (Critical Ratio):</b> Stock-out cost (<i>C<sub>u</sub></i>) and holding cost (<i>C<sub>o</sub></i>) define the optimal service level: <i>CR = C<sub>u</sub> / (C<sub>u</sub> + C<sub>o</sub>)</i>.<br>
        <b>4. Cost-Aware Safety Buffer:</b> Standard normal inverse CDF determines <i>Z<sub>cost</sub> = &Phi;<sup>-1</sup>(CR)</i>, scaling the economic safety buffer.<br>
        <b>5. Cost-Aware Target Stock:</b> Sized as <i>Target Stock = Lead-Time Demand + 7-Day Forecast + Cost-Aware Safety Stock</i>.<br>
        <b>6. Operational Trigger Evaluation:</b> Current on-hand inventory is checked against operational Reorder Point (<i>ROP = LTD + Operational SS</i>).<br>
        <b>7. Recommended Order Generation:</b> When stock &le; ROP, the system orders <i>Recommended Order = max(0, Target Stock - Current Stock)</i>.<br>
        <b>8. Policy Simulation Scoring:</b> Recommendations are evaluated day-by-day via stateful simulation tracking: <i>Beginning Inventory + Orders Received - Actual Demand = Ending Inventory</i>.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 5 & 6. DETAILED REORDER BREAKDOWN & COST MODEL FORMULA
# ---------------------------------------------------------
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📋 Inventory Recommendation Breakdown", "📐 Cost Model & Economic Tradeoff", "📊 Demand Profile Analysis"])

with tab1:
    rec_col1, rec_col2 = st.columns([1, 1])
    with rec_col1:
        st.markdown("#### Cost-Aware Replenishment Matrix")
        summary_table = [
            {"Parameter": "Current Inventory on Hand", "Value": f"{reorder_res['current_stock']:,.1f} units"},
            {"Parameter": "Predicted 7-Day Demand", "Value": f"{reorder_res['total_forecast_7d']:,.1f} units"},
            {"Parameter": "Average Daily Demand Rate", "Value": f"{reorder_res['daily_avg_demand']:,.1f} units/day"},
            {"Parameter": "Supplier Lead Time", "Value": f"{lead_time_days} days"},
            {"Parameter": "Lead-Time Demand (LTD)", "Value": f"{reorder_res['lead_time_demand']:,.1f} units"},
            {"Parameter": "Operational Safety Stock (SS)", "Value": f"{reorder_res['safety_stock']:,.1f} units"},
            {"Parameter": "Reorder Point (ROP = LTD + SS)", "Value": f"{reorder_res['reorder_point']:,.1f} units [Trigger]"},
            {"Parameter": "Stock-out Cost per Unit", "Value": f"${reorder_res['stockout_cost_per_unit']:.2f}"},
            {"Parameter": "Holding Cost per Unit", "Value": f"${reorder_res['holding_cost_per_unit']:.2f}"},
            {"Parameter": "Critical Ratio (Cu / (Cu + Co))", "Value": f"{reorder_res['critical_ratio']:.4f} ({reorder_res['critical_ratio']*100:.1f}%)"},
            {"Parameter": "Optimal Economic Quantile (Z_cost)", "Value": f"{reorder_res['z_cost']:.3f}"},
            {"Parameter": "Cost-Aware Safety Stock", "Value": f"{reorder_res['cost_aware_safety_stock']:,.1f} units"},
            {"Parameter": "Cost-Aware Target Stock", "Value": f"{reorder_res['cost_aware_target_stock']:,.1f} units [Target]"},
            {"Parameter": "Days of Inventory Remaining", "Value": f"{reorder_res['days_of_supply']:.1f} days"},
            {"Parameter": "Recommended Order Quantity", "Value": f"{reorder_res['recommended_order_qty']:,.0f} units"}
        ]
        st.dataframe(pd.DataFrame(summary_table), use_container_width=True, hide_index=True)

    with rec_col2:
        st.markdown("#### 7-Day Forward Horizon Schedule")
        display_fc = forecast_7d[['date', 'forecast_sales', 'onpromotion', 'is_weekend', 'is_holiday']].copy()
        display_fc['date'] = display_fc['date'].dt.strftime('%Y-%m-%d (%a)')
        display_fc.columns = ['Date', 'Forecast Demand', 'Promo Items', 'Weekend', 'Holiday']
        st.dataframe(display_fc, use_container_width=True, hide_index=True)


with tab2:
    st.markdown("#### ⚖️ Cost Model & Newsvendor Economic Tradeoff")
    st.markdown(f"""
    In retail inventory management, every forecast must be converted into a reorder quantity using an explicit cost model balancing **Stock-out Penalty** against **Holding / Carrying Cost**.

    **1. Economic Critical Ratio ($CR$):**
    $$CR = \\frac{{C_{{\\text{{stockout}}}}}}{{C_{{\\text{{stockout}}}} + C_{{\\text{{holding}}}}}} = \\frac{{{stockout_cost_unit:.2f}}}{{{stockout_cost_unit:.2f} + {holding_cost_unit:.2f}}} = {reorder_res['critical_ratio']:.4f}$$
    
    **2. Optimal Economic Quantile ($Z_{{\\text{{cost}}}}$):**
    $$Z_{{\\text{{cost}}}} = \\Phi^{{-1}}(CR) = \\Phi^{{-1}}({reorder_res['critical_ratio']:.4f}) = {reorder_res['z_cost']:.3f}$$

    **3. Cost-Aware Safety Stock & Target Level:**
    $$\\text{{Cost-Aware Safety Stock}} = Z_{{\\text{{cost}}}} \\times \\sqrt{{L}} \\times \\sigma_{{\\text{{demand}}}} = {reorder_res['cost_aware_safety_stock']:,.1f} \\text{{ units}}$$
    $$\\text{{Cost-Aware Target Stock}} = \\text{{LTD}} + D_{{7d}} + \\text{{Cost-Aware Safety Stock}} = {reorder_res['cost_aware_target_stock']:,.1f} \\text{{ units}}$$

    **4. Operational Trigger vs. Economic Target:**
    - **Reorder Point (ROP = LTD + SS)**: {reorder_res['reorder_point']:,.1f} units (Determines **WHEN** to reorder).
    - **Cost-Aware Target Stock**: {reorder_res['cost_aware_target_stock']:,.1f} units (Determines **HOW MUCH** to reorder).
    - **Recommended Order**: $\\max(0, \\lceil \\text{{Cost-Aware Target Stock}} - \\text{{Current Stock}} \\rceil) = {reorder_res['recommended_order_qty']:,} \\text{{ units}}$.

    **5. Evaluation Cost Formulation:**
    $$\\text{{Stock-out Cost}} = \\sum \\max(0, \\text{{Demand}} - \\text{{Stock}}) \\times C_{{\\text{{stockout}}}} = \\text{{Stockout Units}} \\times \\${stockout_cost_unit:.2f}$$
    $$\\text{{Holding Cost}} = \\sum \\max(0, \\text{{Stock}} - \\text{{Demand}}) \\times C_{{\\text{{holding}}}} = \\text{{Excess Units}} \\times \\${holding_cost_unit:.2f}$$
    $$\\text{{Total Inventory Cost}} = \\text{{Stock-out Cost}} + \\text{{Holding Cost}}$$
    """)


with tab3:
    st.markdown("#### 🔍 Historical Demand Statistical Profile")
    dp_col1, dp_col2, dp_col3, dp_col4 = st.columns(4)
    with dp_col1:
        st.metric("Total Observed Records", f"{demand_stats['total_records']:,}")
        st.metric("Mean Daily Sales", f"{demand_stats['mean_daily_sales']:,.2f}")
    with dp_col2:
        st.metric("Std Deviation", f"{demand_stats['std_daily_sales']:,.2f}")
        st.metric("Median Daily Sales", f"{demand_stats['median_sales']:,.2f}")
    with dp_col3:
        st.metric("Zero-Sales Days Ratio", f"{demand_stats['zero_sales_ratio']:.2f}%")
        st.metric("Coefficient of Variation", f"{demand_stats['coefficient_of_variation']:.2f}")
    with dp_col4:
        st.metric("Historical Min Sales", f"{demand_stats['min_sales']:,.1f}")
        st.metric("Historical Max Sales", f"{demand_stats['max_sales']:,.1f}")

# Footer
st.markdown("---")
st.caption("Retail Inventory Reorder Assistant | Hackathon MVP | Store Sales Favorita Dataset | Running via Streamlit")
