"""
Retail Inventory Reorder Assistant - Core Pipeline
Handles:
1. Demand analysis
2. Data preprocessing & date reindexing
3. Lag & Calendar & Promotion feature engineering
4. Chronological train/test split
5. Moving Average baseline
6. ANN forecasting model (TensorFlow/Keras)
7. Model evaluation (MAE, RMSE, Stock-out Oriented Underprediction Loss)
8. Recursive 7-day forecasting horizon
9. Inventory calculation & Reorder recommendation (ROP, Safety Stock, Order Qty)
10. Insufficient-history / New product handling
11. Slow-moving / Intermittent product handling
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# Default constants
DEFAULT_STORE = 1
DEFAULT_FAMILIES = ['GROCERY I', 'BEVERAGES', 'CLEANING', 'DAIRY', 'HARDWARE']
FORECAST_HORIZON = 7


def load_dataset(
    train_path='train.csv',
    holidays_path='holidays_events.csv',
    store_nbr=DEFAULT_STORE,
    families=None,
    start_date='2015-01-01'
):
    """
    Loads train.csv with selected store and product families, merges holidays.
    Fast and memory efficient.
    """
    if families is None:
        families = DEFAULT_FAMILIES

    # Read needed columns
    df = pd.read_csv(
        train_path,
        usecols=['date', 'store_nbr', 'family', 'sales', 'onpromotion']
    )
    df['date'] = pd.to_datetime(df['date'])

    # Filter store and families
    df_filtered = df[(df['store_nbr'] == store_nbr) & (df['family'].isin(families))].copy()
    if start_date:
        df_filtered = df_filtered[df_filtered['date'] >= pd.to_datetime(start_date)].copy()

    # Load holidays
    holiday_dates = set()
    if os.path.exists(holidays_path):
        h = pd.read_csv(holidays_path)
        # Exclude transferred holidays
        if 'transferred' in h.columns:
            h_active = h[h['transferred'] == False]
        else:
            h_active = h
        holiday_dates = set(pd.to_datetime(h_active['date']))

    return df_filtered, holiday_dates


def compute_demand_analysis(df_family):
    """
    Computes statistical demand profile for a product family:
    - Mean, std, median, min, max, total sales
    - Zero-sales ratio (intermittent indicator)
    - Classification: High-volume, Medium, Slow-moving
    """
    sales = df_family['sales']
    zero_ratio = float((sales == 0).mean())
    mean_val = float(sales.mean())
    std_val = float(sales.std()) if len(sales) > 1 else 0.0
    cv = float(std_val / mean_val) if mean_val > 0 else 0.0

    # Classification
    if zero_ratio > 0.25 or mean_val < 5.0:
        demand_type = "Slow-Moving / Intermittent"
    elif mean_val > 1000:
        demand_type = "High-Volume Staple"
    else:
        demand_type = "Regular Demand"

    return {
        'total_records': int(len(df_family)),
        'total_sales': float(sales.sum()),
        'mean_daily_sales': round(mean_val, 2),
        'std_daily_sales': round(std_val, 2),
        'median_sales': round(float(sales.median()), 2),
        'min_sales': round(float(sales.min()), 2),
        'max_sales': round(float(sales.max()), 2),
        'zero_sales_ratio': round(zero_ratio * 100, 2),
        'coefficient_of_variation': round(cv, 2),
        'demand_type': demand_type
    }


def prepare_time_series(df_family, holiday_dates=None):
    """
    Ensures complete daily date index, fills store closures (e.g. Christmas) with 0,
    adds calendar, promotion, and lag features.
    """
    df_sorted = df_family.sort_values('date').copy()
    df_sorted.set_index('date', inplace=True)

    # Reindex to full daily frequency
    full_idx = pd.date_range(df_sorted.index.min(), df_sorted.index.max(), freq='D')
    df_reindexed = df_sorted.reindex(full_idx)
    df_reindexed['sales'] = df_reindexed['sales'].fillna(0.0)
    df_reindexed['onpromotion'] = df_reindexed['onpromotion'].fillna(0)
    df_reindexed.index.name = 'date'
    df_feat = df_reindexed.reset_index()

    # 1. Calendar Features
    df_feat['dayofweek'] = df_feat['date'].dt.dayofweek
    df_feat['day'] = df_feat['date'].dt.day
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)

    if holiday_dates:
        df_feat['is_holiday'] = df_feat['date'].isin(holiday_dates).astype(int)
    else:
        df_feat['is_holiday'] = 0

    # 2. Promotion features
    df_feat['promo_lag_1'] = df_feat['onpromotion'].shift(1).fillna(0)

    # 3. Lag features (sales)
    df_feat['sales_lag_1'] = df_feat['sales'].shift(1)
    df_feat['sales_lag_7'] = df_feat['sales'].shift(7)
    df_feat['sales_lag_14'] = df_feat['sales'].shift(14)

    # Rolling statistics (using shift(1) to prevent leakage of today's sales)
    df_feat['rolling_mean_7'] = df_feat['sales'].shift(1).rolling(window=7, min_periods=1).mean()
    df_feat['rolling_std_7'] = df_feat['sales'].shift(1).rolling(window=7, min_periods=1).std().fillna(0)

    # Drop early rows with NaN from 14-day lags
    df_feat = df_feat.dropna().reset_index(drop=True)

    return df_feat


def chronological_split(df_feat, test_days=28):
    """
    Chronological train/test split (no data leakage).
    """
    if len(df_feat) <= test_days:
        split_idx = int(len(df_feat) * 0.8)
    else:
        split_idx = len(df_feat) - test_days

    train_df = df_feat.iloc[:split_idx].copy()
    test_df = df_feat.iloc[split_idx:].copy()
    return train_df, test_df


def compute_moving_average_baseline(train_df, test_df, window=7):
    """
    Moving Average baseline forecast.
    Predicts each test day using the 7-day rolling average of recent sales.
    """
    combined_sales = pd.concat([train_df['sales'], test_df['sales']])
    ma_series = combined_sales.shift(1).rolling(window=window, min_periods=1).mean()
    test_ma = ma_series.iloc[len(train_df):].values
    return test_ma


def evaluate_forecast(y_true, y_pred, underpredict_penalty=3.0, overpredict_penalty=1.0):
    """
    Calculates:
    - MAE
    - RMSE
    - Stock-out oriented Asymmetric Loss:
        Loss = (w_under * max(0, y_true - y_pred) + w_over * max(0, y_pred - y_true)) / N
    - Stock-out Frequency (% of days actual demand exceeded forecast)
    - Total Missed Units (Stock-out units)
    """
    y_true = np.array(y_true)
    y_pred = np.maximum(0, np.array(y_pred))  # Non-negative predictions

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    underprediction = np.maximum(0, y_true - y_pred)
    overprediction = np.maximum(0, y_pred - y_true)

    # Stock-out oriented asymmetric loss
    stockout_asymmetric_loss = float(np.mean(
        underpredict_penalty * underprediction + overpredict_penalty * overprediction
    ))

    # Stockout risk: percentage of days with underprediction
    stockout_days = int(np.sum(underprediction > 0.01))
    stockout_frequency = float((stockout_days / len(y_true)) * 100) if len(y_true) > 0 else 0.0
    total_missed_units = float(np.sum(underprediction))

    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'stockout_asymmetric_loss': round(stockout_asymmetric_loss, 2),
        'stockout_frequency_pct': round(stockout_frequency, 1),
        'stockout_days': stockout_days,
        'total_missed_units': round(total_missed_units, 1)
    }


def build_and_train_ann(
    X_train, y_train,
    X_val, y_val,
    epochs=25,
    batch_size=32,
    random_state=42
):
    """
    Builds and trains a Dense Artificial Neural Network for tabular time series forecasting.
    """
    tf.keras.utils.set_random_seed(random_state)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.15),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='relu')  # ReLU ensures non-negative sales predictions
    ])

    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    return model, history


def forecast_next_7_days(
    model,
    scaler,
    feature_cols,
    recent_df,
    future_promotions=None,
    holiday_dates=None
):
    """
    Recursively forecasts the next 7 days using the trained ANN model.
    """
    predictions = []
    temp_df = recent_df.copy().reset_index(drop=True)
    last_date = temp_df['date'].max()

    for step in range(1, FORECAST_HORIZON + 1):
        next_date = last_date + pd.Timedelta(days=step)
        next_dow = next_date.dayofweek
        next_day = next_date.day
        next_month = next_date.month
        next_is_weekend = 1 if next_dow in [5, 6] else 0
        next_is_holiday = 1 if holiday_dates and next_date in holiday_dates else 0

        promo_val = future_promotions[step - 1] if (future_promotions and len(future_promotions) >= step) else 0

        # Lags from temp_df
        sales_lag_1 = temp_df['sales'].iloc[-1]
        sales_lag_7 = temp_df['sales'].iloc[-7] if len(temp_df) >= 7 else sales_lag_1
        sales_lag_14 = temp_df['sales'].iloc[-14] if len(temp_df) >= 14 else sales_lag_7

        rolling_mean_7 = temp_df['sales'].iloc[-7:].mean()
        rolling_std_7 = temp_df['sales'].iloc[-7:].std()
        if pd.isna(rolling_std_7):
            rolling_std_7 = 0.0

        promo_lag_1 = temp_df['onpromotion'].iloc[-1]

        # Assemble feature row matching feature_cols
        row_dict = {
            'onpromotion': promo_val,
            'dayofweek': next_dow,
            'day': next_day,
            'month': next_month,
            'is_weekend': next_is_weekend,
            'is_holiday': next_is_holiday,
            'promo_lag_1': promo_lag_1,
            'sales_lag_1': sales_lag_1,
            'sales_lag_7': sales_lag_7,
            'sales_lag_14': sales_lag_14,
            'rolling_mean_7': rolling_mean_7,
            'rolling_std_7': rolling_std_7
        }

        feat_df = pd.DataFrame([row_dict])[feature_cols]
        feat_scaled = scaler.transform(feat_df)
        pred_val = float(model.predict(feat_scaled, verbose=0)[0][0])
        pred_val = max(0.0, pred_val)

        predictions.append({
            'date': next_date,
            'forecast_sales': round(pred_val, 2),
            'onpromotion': int(promo_val),
            'is_holiday': int(next_is_holiday),
            'is_weekend': int(next_is_weekend)
        })

        # Append to temp_df for next recursive step
        new_row = pd.DataFrame([{
            'date': next_date,
            'sales': pred_val,
            'onpromotion': promo_val
        }])
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    return pd.DataFrame(predictions)


def calculate_reorder_recommendation(
    current_stock,
    forecast_7d_df,
    lead_time_days=2,
    service_level_z=1.65,  # Operational service level for ROP trigger
    stockout_cost_per_unit=15.0,  # Economic stock-out penalty ($/unit)
    holding_cost_per_unit=2.0,    # Economic holding cost ($/unit)
    is_slow_moving=False,
    is_new_or_insufficient=False,
    historical_std=None,
    **kwargs
):
    """
    Calculates Inventory Reorder Parameters with explicit cost-aware optimization:
    1. Operational Parameters:
       - Lead Time Demand (LTD) = daily_avg_demand * lead_time_days
       - Operational Safety Stock (SS) = service_level_z * sqrt(L) * std_demand
       - Reorder Point (ROP) = LTD + SS (Operational trigger)
    2. Economic Cost-Aware Parameters:
       - Critical Ratio (CR) = stockout_cost / (stockout_cost + holding_cost)
       - Optimal Quantile (Z_cost) = norm.ppf(CR)
       - Cost-Aware Safety Stock = Z_cost * sqrt(L) * std_demand
       - Cost-Aware Target Stock = LTD + 7d_forecast + Cost-Aware Safety Stock
       - Recommended Reorder Qty = max(0, ceil(Cost-Aware Target Stock - Current Stock)) if reorder_needed else 0
    """
    if 'stockout_cost' in kwargs:
        stockout_cost_per_unit = kwargs['stockout_cost']
    if 'holding_cost' in kwargs:
        holding_cost_per_unit = kwargs['holding_cost']
    if 'stock_out_cost_per_unit' in kwargs:
        stockout_cost_per_unit = kwargs['stock_out_cost_per_unit']
    if 'excess_cost_per_unit' in kwargs:
        holding_cost_per_unit = kwargs['excess_cost_per_unit']

    total_forecast_7d = float(forecast_7d_df['forecast_sales'].sum())
    daily_avg_demand = total_forecast_7d / len(forecast_7d_df)

    if historical_std is None or historical_std == 0:
        std_demand = forecast_7d_df['forecast_sales'].std()
        if pd.isna(std_demand) or std_demand == 0:
            std_demand = daily_avg_demand * 0.25
    else:
        std_demand = historical_std

    # 1. Lead Time Demand: demand during delivery period
    lead_time_demand = daily_avg_demand * lead_time_days

    # 2. Operational Safety Stock & ROP (Operational Trigger)
    safety_stock = service_level_z * np.sqrt(lead_time_days) * std_demand

    notes = []
    if is_slow_moving:
        min_safety_floor = 5.0  # Guarantee at least 5 units buffer so intermittent demand won't stock out
        if safety_stock < min_safety_floor:
            safety_stock = min_safety_floor
            notes.append("Slow-moving item: Enforced minimum safety stock floor (5 units).")

    if is_new_or_insufficient:
        # Buffer safety stock by 35% to protect against cold-start uncertainty
        safety_stock = safety_stock * 1.35
        notes.append("New/Low-History item: Applied 35% cold-start uncertainty buffer.")

    reorder_point = lead_time_demand + safety_stock

    # 3. Cost-Aware Economic Policy via Critical Ratio
    # Critical Ratio = Cu / (Cu + Co)
    critical_ratio = float(stockout_cost_per_unit / (stockout_cost_per_unit + holding_cost_per_unit))
    # Clamping for extreme values
    cr_clamped = max(0.001, min(0.999, critical_ratio))
    z_cost = float(norm.ppf(cr_clamped))

    # Cost-aware safety buffer based on economic quantile
    cost_aware_safety_stock = z_cost * np.sqrt(lead_time_days) * std_demand

    if is_slow_moving:
        min_safety_floor = 5.0
        if cost_aware_safety_stock < min_safety_floor:
            cost_aware_safety_stock = min_safety_floor

    if is_new_or_insufficient:
        cost_aware_safety_stock = cost_aware_safety_stock * 1.35

    # Cost-aware target stock covers LTD + 7-day replenishment horizon + economic safety stock
    cost_aware_target_stock = lead_time_demand + total_forecast_7d + cost_aware_safety_stock
    cost_aware_target_stock = max(0.0, cost_aware_target_stock)

    # 4. Days of Supply (DOS)
    if daily_avg_demand > 0:
        days_of_supply = current_stock / daily_avg_demand
    else:
        days_of_supply = 999.0

    # 5. Reorder Recommendation
    if current_stock <= reorder_point:
        reorder_needed = True
        recommended_order = max(0.0, np.ceil(cost_aware_target_stock - current_stock))
        if current_stock <= lead_time_demand:
            urgency = "CRITICAL (Stockout imminent within lead time)"
        else:
            urgency = "HIGH (Reorder point breached)"
    else:
        reorder_needed = False
        recommended_order = 0.0
        urgency = "NORMAL (Stock level sufficient)"

    return {
        'current_stock': round(float(current_stock), 1),
        'lead_time_days': int(lead_time_days),
        'daily_avg_demand': round(daily_avg_demand, 2),
        'total_forecast_7d': round(total_forecast_7d, 2),
        'lead_time_demand': round(lead_time_demand, 2),
        'safety_stock': round(float(safety_stock), 2),
        'reorder_point': round(float(reorder_point), 2),
        'stockout_cost_per_unit': round(float(stockout_cost_per_unit), 2),
        'holding_cost_per_unit': round(float(holding_cost_per_unit), 2),
        'critical_ratio': round(float(critical_ratio), 4),
        'z_cost': round(float(z_cost), 3),
        'cost_aware_safety_stock': round(float(cost_aware_safety_stock), 2),
        'cost_aware_target_stock': round(float(cost_aware_target_stock), 2),
        'days_of_supply': round(float(days_of_supply), 1),
        'reorder_needed': bool(reorder_needed),
        'recommended_order_qty': int(recommended_order),
        'urgency': urgency,
        'notes': notes
    }


def handle_new_or_insufficient_product(
    new_product_sales_sample,
    category_mean_daily=150.0,
    category_std_daily=30.0,
    days_observed=5
):
    """
    Handling for new products or products with insufficient history (< 30 days).
    Uses a Bayesian/shrinkage heuristic combining observed sample (if any)
    with category reference demand.
    """
    if len(new_product_sales_sample) == 0:
        estimated_mean = category_mean_daily
        estimated_std = category_std_daily
    else:
        obs_mean = float(np.mean(new_product_sales_sample))
        # Weight observed vs category based on sample size (shrinkage)
        weight = min(1.0, len(new_product_sales_sample) / 30.0)
        estimated_mean = (weight * obs_mean) + ((1.0 - weight) * category_mean_daily)
        estimated_std = category_std_daily * 1.2  # Add uncertainty margin

    # Generate 7-day heuristic forecast
    heuristic_dates = pd.date_range(pd.Timestamp.today(), periods=FORECAST_HORIZON, freq='D')
    heuristic_forecast = pd.DataFrame({
        'date': heuristic_dates,
        'forecast_sales': [round(estimated_mean, 2)] * FORECAST_HORIZON,
        'onpromotion': [0] * FORECAST_HORIZON,
        'is_holiday': [0] * FORECAST_HORIZON,
        'is_weekend': [1 if d.dayofweek in [5, 6] else 0 for d in heuristic_dates]
    })

    return heuristic_forecast, estimated_mean, estimated_std


# ---------------------------------------------------------
# COLD-START DEMO / SYNTHETIC PRODUCT CATALOG
# Clearly documented cold-start SKUs with little/no history
# ---------------------------------------------------------
COLD_START_PRODUCTS = {
    "[NEW] Artisanal Cold-Brew (Demo Cold-Start SKU)": {
        "sku_name": "Artisanal Cold-Brew",
        "category_reference": "BEVERAGES",
        "category_mean": 18.0,
        "category_std": 4.5,
        "observed_sales": [12.0, 15.0, 10.0],
        "history_count": 3,
        "default_stock": 12,
        "unit": "bottles",
        "description": "Specialty bottled cold-brew coffee with only 3 days of pilot sales."
    },
    "[NEW] Organic Protein Bar (Demo Cold-Start SKU)": {
        "sku_name": "Organic Protein Bar",
        "category_reference": "GROCERY I",
        "category_mean": 25.0,
        "category_std": 6.0,
        "observed_sales": [8.0, 14.0, 11.0, 9.0, 13.0],
        "history_count": 5,
        "default_stock": 18,
        "unit": "bars",
        "description": "Plant-based high-protein snack bar with only 5 days of early launch data."
    },
    "[NEW] Premium Energy Drink (Demo Cold-Start SKU)": {
        "sku_name": "Premium Energy Drink",
        "category_reference": "BEVERAGES",
        "category_mean": 32.0,
        "category_std": 8.0,
        "observed_sales": [20.0],
        "history_count": 1,
        "default_stock": 25,
        "unit": "cans",
        "description": "Zero-sugar natural energy drink launched yesterday (1 single observation)."
    }
}


def simulate_reorder_policy_inventory(
    y_true,
    y_pred,
    lead_time_days=2,
    stockout_cost_per_unit=15.0,
    holding_cost_per_unit=2.0,
    historical_std=None,
    service_level_z=1.65,
    initial_stock=None,
    is_slow_moving=False,
    is_new_or_insufficient=False
):
    """
    Simulates the actual inventory reorder policy over the chronological evaluation period.
    Tracks state day-by-day:
        Beginning Inventory
        + Recommended Reorder Quantity (orders received after lead-time)
        - Actual Demand
        = Ending Inventory

    At each step:
        Unmet Units = max(0, Actual Demand - Available Inventory)
        Excess Inventory = max(0, Ending Inventory)
        Stock-out Cost = Unmet Units * Stock-out Cost Per Unit
        Holding Cost = Excess Inventory * Holding Cost Per Unit
        Total Cost = Stock-out Cost + Holding Cost
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.array(y_pred, dtype=float))
    T = len(y_true)
    L = int(lead_time_days)

    mean_demand = float(np.mean(y_true)) if T > 0 else 1.0
    if historical_std is None or historical_std == 0:
        std_demand = float(np.std(y_true)) if T > 1 else (mean_demand * 0.25)
    else:
        std_demand = float(historical_std)

    # Critical ratio and economic quantile
    cr = float(stockout_cost_per_unit / (stockout_cost_per_unit + holding_cost_per_unit))
    cr_clamped = max(0.001, min(0.999, cr))
    z_cost = float(norm.ppf(cr_clamped))

    # Base ROP and target stock sizing for initial state
    ltd_init = mean_demand * L
    ss_init = service_level_z * np.sqrt(L) * std_demand
    if is_slow_moving:
        ss_init = max(5.0, ss_init)
    if is_new_or_insufficient:
        ss_init = ss_init * 1.35
    rop_init = ltd_init + ss_init

    # If initial stock not given, initialize at ROP (normal operational level)
    if initial_stock is None:
        inventory = float(rop_init)
    else:
        inventory = float(initial_stock)

    initial_inventory = inventory
    orders_placed = np.zeros(T)
    orders_received = np.zeros(T)
    unmet_units_arr = np.zeros(T)
    excess_units_arr = np.zeros(T)
    beginning_inv_arr = np.zeros(T)

    for t in range(T):
        beginning_inv_arr[t] = inventory

        # 1. Receive orders placed L days ago
        if t >= L:
            orders_received[t] = orders_placed[t - L]
        else:
            orders_received[t] = 0.0

        available = inventory + orders_received[t]
        actual_demand = y_true[t]

        # 2. Demand fulfillment
        unmet = max(0.0, actual_demand - available)
        ending = max(0.0, available - actual_demand)
        inventory = ending

        unmet_units_arr[t] = unmet
        excess_units_arr[t] = ending

        # 3. Policy Reorder Decision using model forecast
        d_hat = y_pred[t] if t < len(y_pred) else mean_demand
        ltd = d_hat * L
        ss_oper = service_level_z * np.sqrt(L) * std_demand
        ss_econ = z_cost * np.sqrt(L) * std_demand

        if is_slow_moving:
            ss_oper = max(5.0, ss_oper)
            ss_econ = max(5.0, ss_econ)
        if is_new_or_insufficient:
            ss_oper = ss_oper * 1.35
            ss_econ = ss_econ * 1.35

        rop = ltd + ss_oper
        target_stock = max(0.0, ltd + (d_hat * FORECAST_HORIZON) + ss_econ)

        # Outstanding pipeline orders currently in transit
        pipeline_in_transit = float(np.sum(orders_placed[max(0, t - L + 1):t])) if (t > 0 and L > 1) else 0.0
        effective_inventory = inventory + pipeline_in_transit

        if effective_inventory <= rop:
            order_qty = max(0.0, np.ceil(target_stock - effective_inventory))
        else:
            order_qty = 0.0

        orders_placed[t] = order_qty

    total_stockout_units = float(np.sum(unmet_units_arr))
    total_excess_inventory = float(np.sum(excess_units_arr))
    total_units_ordered = float(np.sum(orders_placed))
    total_actual_demand = float(np.sum(y_true))
    stockout_days = int(np.sum(unmet_units_arr > 0.01))

    total_stockout_cost = total_stockout_units * stockout_cost_per_unit
    total_holding_cost = total_excess_inventory * holding_cost_per_unit
    total_cost = total_stockout_cost + total_holding_cost

    return {
        'initial_inventory': round(initial_inventory, 1),
        'total_actual_demand': round(total_actual_demand, 1),
        'total_units_ordered': round(total_units_ordered, 1),
        'total_stockout_units': round(total_stockout_units, 1),
        'total_excess_units': round(total_excess_inventory, 1),
        'total_excess_inventory': round(total_excess_inventory, 1),
        'stockout_days': stockout_days,
        'average_daily_inventory': round(float(np.mean(excess_units_arr)) if T > 0 else 0.0, 1),
        'total_stockout_cost': round(total_stockout_cost, 2),
        'total_holding_cost': round(total_holding_cost, 2),
        'total_cost': round(total_cost, 2),
        'orders_placed': orders_placed.tolist(),
        'orders_received': orders_received.tolist(),
        'ending_inventory': excess_units_arr.tolist(),
        'unmet_units': unmet_units_arr.tolist()
    }


def simulate_inventory_cost(
    y_true,
    y_pred,
    stockout_cost_per_unit=15.0,
    holding_cost_per_unit=2.0,
    lead_time_days=2,
    historical_std=None,
    service_level_z=1.65,
    initial_stock=None,
    is_slow_moving=False,
    is_new_or_insufficient=False
):
    """
    Simulates inventory costs by evaluating the actual reorder policy over the evaluation period.
    """
    return simulate_reorder_policy_inventory(
        y_true=y_true,
        y_pred=y_pred,
        lead_time_days=lead_time_days,
        stockout_cost_per_unit=stockout_cost_per_unit,
        holding_cost_per_unit=holding_cost_per_unit,
        historical_std=historical_std,
        service_level_z=service_level_z,
        initial_stock=initial_stock,
        is_slow_moving=is_slow_moving,
        is_new_or_insufficient=is_new_or_insufficient
    )
