"""Streamlit dashboard for Task 7.

This app provides four views:
Sales overview, forecast explorer, anomaly report, and product demand segments.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

st.set_page_config(
	page_title="Sales Forecasting Dashboard",
	page_icon="📈",
	layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
	df = pd.read_csv("train.csv")
	df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
	df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
	df = df.dropna(subset=["Order Date", "Sales"]).copy()
	df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0.0)
	df["Year"] = df["Order Date"].dt.year
	df["Month"] = df["Order Date"].dt.month
	df["Year-Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
	return df


def season_for_month(month: int) -> str:
	if month in [12, 1, 2]:
		return "Winter"
	if month in [3, 4, 5]:
		return "Spring"
	if month in [6, 7, 8]:
		return "Summer"
	return "Fall"


def build_supervised_frame(monthly_series: pd.Series) -> pd.DataFrame:
	monthly_series = monthly_series.asfreq("M", fill_value=0)
	supervised_df = pd.DataFrame({"Sales": monthly_series})
	supervised_df["Lag_1"] = supervised_df["Sales"].shift(1)
	supervised_df["Lag_2"] = supervised_df["Sales"].shift(2)
	supervised_df["Lag_3"] = supervised_df["Sales"].shift(3)
	supervised_df["Rolling_Mean_3"] = supervised_df["Sales"].shift(1).rolling(window=3).mean()
	supervised_df["Month"] = supervised_df.index.month
	supervised_df["Quarter"] = supervised_df.index.quarter
	supervised_df["Season"] = supervised_df["Month"].apply(season_for_month)
	supervised_df = pd.get_dummies(supervised_df, columns=["Season"], prefix="Season")
	return supervised_df.dropna().copy()


def make_future_features(feature_columns: list[str], history: list[float], next_month: pd.Timestamp) -> pd.DataFrame:
	next_row = {
		"Lag_1": history[-1],
		"Lag_2": history[-2] if len(history) > 1 else history[-1],
		"Lag_3": history[-3] if len(history) > 2 else history[-1],
		"Rolling_Mean_3": float(np.mean(history[-3:])),
		"Month": next_month.month,
		"Quarter": next_month.quarter,
		"Season_Winter": 1 if next_month.month in [12, 1, 2] else 0,
		"Season_Spring": 1 if next_month.month in [3, 4, 5] else 0,
		"Season_Summer": 1 if next_month.month in [6, 7, 8] else 0,
		"Season_Fall": 1 if next_month.month in [9, 10, 11] else 0,
	}
	next_features = pd.DataFrame([next_row])
	for column in feature_columns:
		if column not in next_features.columns:
			next_features[column] = 0
	return next_features[feature_columns].fillna(0)


@st.cache_data(show_spinner=False)
def get_monthly_sales(df: pd.DataFrame, column: str, value: str) -> pd.Series:
	filtered = df[df[column] == value].copy()
	monthly_series = filtered.set_index("Order Date")["Sales"].resample("M").sum().dropna()
	return monthly_series.asfreq("M", fill_value=0)


@st.cache_resource(show_spinner=False)
def train_forecast_model(segment_key: str, monthly_series: pd.Series):
	supervised_df = build_supervised_frame(monthly_series)
	if len(supervised_df) < 6:
		raise ValueError("Not enough monthly observations to train a forecast model.")

	X = supervised_df.drop(columns=["Sales"])
	y = supervised_df["Sales"]
	validation_size = min(3, max(1, len(supervised_df) // 4))
	split_index = len(supervised_df) - validation_size
	X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
	y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

	model = XGBRegressor(
		n_estimators=200,
		learning_rate=0.05,
		max_depth=3,
		subsample=0.9,
		colsample_bytree=0.9,
		random_state=42,
		objective="reg:squarederror",
	)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	mae = mean_absolute_error(y_test, y_pred)
	rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

	return {
		"model": model,
		"feature_columns": list(X.columns),
		"mae": mae,
		"rmse": rmse,
		"y_test": y_test,
		"y_pred": pd.Series(y_pred, index=y_test.index),
	}


def forecast_future(monthly_series: pd.Series, model, feature_columns: list[str], horizon: int) -> pd.Series:
	history = list(monthly_series.iloc[-3:].values)
	current_month = monthly_series.index[-1]
	future_dates = []
	future_values = []

	for _ in range(horizon):
		next_month = current_month + pd.offsets.MonthEnd(1)
		future_features = make_future_features(feature_columns, history, next_month)
		prediction = float(model.predict(future_features)[0])
		future_dates.append(next_month)
		future_values.append(prediction)
		history.append(prediction)
		current_month = next_month

	return pd.Series(future_values, index=future_dates)


@st.cache_data(show_spinner=False)
def build_anomaly_frame(df: pd.DataFrame) -> pd.DataFrame:
	weekly_sales = df.sort_values("Order Date").set_index("Order Date")["Sales"].resample("W").sum().dropna()
	weekly_frame = pd.DataFrame({"Sales": weekly_sales})
	weekly_frame["Rolling_Mean_4"] = weekly_frame["Sales"].rolling(window=4, min_periods=1).mean()
	weekly_frame["Rolling_Std_4"] = weekly_frame["Sales"].rolling(window=4, min_periods=1).std().fillna(0)
	weekly_frame["Lag_1"] = weekly_frame["Sales"].shift(1)
	weekly_frame["Lag_2"] = weekly_frame["Sales"].shift(2)
	weekly_frame["Pct_Change"] = weekly_frame["Sales"].pct_change()
	weekly_frame = weekly_frame.dropna().copy()

	anomaly_features = weekly_frame[["Sales", "Rolling_Mean_4", "Rolling_Std_4", "Lag_1", "Lag_2", "Pct_Change"]].fillna(0)
	model = IsolationForest(contamination=0.08, random_state=42)
	weekly_frame["Isolation_Flag"] = model.fit_predict(anomaly_features)
	weekly_frame["Isolation_Anomaly"] = weekly_frame["Isolation_Flag"].eq(-1)
	weekly_frame["Z_Score"] = (weekly_frame["Sales"] - weekly_frame["Rolling_Mean_4"]) / weekly_frame["Rolling_Std_4"].replace(0, np.nan)
	weekly_frame["Z_Score"] = weekly_frame["Z_Score"].fillna(0)
	weekly_frame["Z_Anomaly"] = weekly_frame["Z_Score"].abs() > 2
	weekly_frame["Method"] = np.where(
		weekly_frame["Isolation_Anomaly"] & weekly_frame["Z_Anomaly"],
		"Isolation Forest + Z-Score",
		np.where(weekly_frame["Isolation_Anomaly"], "Isolation Forest", np.where(weekly_frame["Z_Anomaly"], "Z-Score", "Normal")),
	)
	return weekly_frame


@st.cache_data(show_spinner=False)
def build_cluster_frame(df: pd.DataFrame) -> pd.DataFrame:
	work_df = df.copy()
	work_df["Year"] = work_df["Order Date"].dt.year
	work_df["Month Period"] = work_df["Order Date"].dt.to_period("M")

	monthly_subcat_sales = work_df.groupby(["Sub-Category", "Month Period"])["Sales"].sum().reset_index()
	monthly_subcat_sales["Month Period"] = monthly_subcat_sales["Month Period"].dt.to_timestamp()

	subcat_features = work_df.groupby("Sub-Category").agg(
		total_sales_volume=("Sales", "sum"),
		average_order_value=("Sales", "mean"),
		order_count=("Sales", "size"),
	).reset_index()

	yearly_subcat_sales = work_df.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()
	yearly_subcat_sales = yearly_subcat_sales.sort_values(["Sub-Category", "Year"]).copy()
	yearly_subcat_sales["YoY_Growth"] = yearly_subcat_sales.groupby("Sub-Category")["Sales"].pct_change()
	growth_feature = yearly_subcat_sales.groupby("Sub-Category")["YoY_Growth"].mean().reset_index(name="sales_growth_rate_yoy")
	volatility_feature = monthly_subcat_sales.groupby("Sub-Category")["Sales"].std().reset_index(name="sales_volatility")

	feature_frame = subcat_features.merge(growth_feature, on="Sub-Category", how="left").merge(volatility_feature, on="Sub-Category", how="left")
	feature_frame[["sales_growth_rate_yoy", "sales_volatility", "average_order_value", "total_sales_volume"]] = feature_frame[
		["sales_growth_rate_yoy", "sales_volatility", "average_order_value", "total_sales_volume"]
	].replace([np.inf, -np.inf], np.nan).fillna(0)

	feature_cols = ["total_sales_volume", "sales_growth_rate_yoy", "sales_volatility", "average_order_value"]
	X_cluster = feature_frame[feature_cols]
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X_cluster)

	kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
	feature_frame["cluster"] = kmeans.fit_predict(X_scaled)

	cluster_profile = feature_frame.groupby("cluster")[feature_cols].mean().sort_values("total_sales_volume", ascending=False)
	cluster_labels = {}
	for cluster_id, row in cluster_profile.iterrows():
		if row["total_sales_volume"] >= cluster_profile["total_sales_volume"].median() and row["sales_volatility"] <= cluster_profile["sales_volatility"].median():
			label = "High Volume, Stable Demand"
		elif row["sales_growth_rate_yoy"] > 0:
			label = "Growing Demand"
		elif row["sales_growth_rate_yoy"] < 0:
			label = "Declining Demand"
		else:
			label = "Low Volume, High Volatility"
		cluster_labels[cluster_id] = label

	feature_frame["cluster_label"] = feature_frame["cluster"].map(cluster_labels)
	pca = PCA(n_components=2, random_state=42)
	components = pca.fit_transform(X_scaled)
	feature_frame["pca_1"] = components[:, 0]
	feature_frame["pca_2"] = components[:, 1]
	return feature_frame


def sales_overview_page(df: pd.DataFrame) -> None:
	st.title("Sales Overview Dashboard")
	st.caption("Interactive summary of yearly sales, monthly trends, and region/category filters.")

	col1, col2, col3 = st.columns(3)
	col1.metric("Total Sales", f"${df['Sales'].sum():,.2f}")
	col2.metric("Orders", f"{len(df):,}")
	col3.metric("Date Range", f"{df['Order Date'].min().date()} to {df['Order Date'].max().date()}")

	with st.sidebar:
		st.header("Filters")
		regions = sorted(df["Region"].dropna().unique().tolist())
		categories = sorted(df["Category"].dropna().unique().tolist())
		selected_regions = st.multiselect("Region", regions, default=regions)
		selected_categories = st.multiselect("Category", categories, default=categories)

	filtered = df[df["Region"].isin(selected_regions) & df["Category"].isin(selected_categories)].copy()
	if filtered.empty:
		st.warning("No records match the selected region and category filters.")
		return

	left, right = st.columns([1.1, 1])
	yearly_sales = filtered.groupby("Year", as_index=False)["Sales"].sum()
	monthly_sales = filtered.groupby("Year-Month", as_index=False)["Sales"].sum()

	with left:
		st.subheader("Total Sales by Year")
		fig_year = px.bar(yearly_sales, x="Year", y="Sales", text_auto="$.2s", color="Sales", color_continuous_scale="tealrose")
		fig_year.update_layout(coloraxis_showscale=False, xaxis_title="Year", yaxis_title="Sales")
		st.plotly_chart(fig_year, width="stretch")

	with right:
		st.subheader("Monthly Sales Trend")
		fig_month = px.line(monthly_sales, x="Year-Month", y="Sales", markers=True)
		fig_month.update_layout(xaxis_title="Month", yaxis_title="Sales")
		st.plotly_chart(fig_month, width="stretch")

	st.subheader("Sales by Region and Category")
	c1, c2 = st.columns(2)
	region_sales = filtered.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
	category_sales = filtered.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)

	with c1:
		fig_region = px.bar(region_sales, x="Region", y="Sales", color="Region", text_auto="$.2s")
		fig_region.update_layout(showlegend=False, xaxis_title="Region", yaxis_title="Sales")
		st.plotly_chart(fig_region, width="stretch")

	with c2:
		fig_category = px.bar(category_sales, x="Category", y="Sales", color="Category", text_auto="$.2s")
		fig_category.update_layout(showlegend=False, xaxis_title="Category", yaxis_title="Sales")
		st.plotly_chart(fig_category, width="stretch")

	heatmap = filtered.pivot_table(index="Region", columns="Category", values="Sales", aggfunc="sum", fill_value=0)
	fig_heatmap = px.imshow(heatmap, text_auto=True, aspect="auto", color_continuous_scale="Blues", labels={"color": "Sales"})
	fig_heatmap.update_layout(title="Region x Category Sales Heatmap")
	st.plotly_chart(fig_heatmap, width="stretch")


def forecast_explorer_page(df: pd.DataFrame) -> None:
	st.title("Forecast Explorer")
	st.caption("Choose a category or region and forecast 1 to 3 months ahead using the best-performing XGBoost model.")

	choice_type = st.radio("Forecast by", ["Category", "Region"], horizontal=True)
	options = sorted(df[choice_type].dropna().unique().tolist())
	selected_value = st.selectbox(f"Select {choice_type}", options)
	horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3, step=1)

	try:
		monthly_series = get_monthly_sales(df, choice_type, selected_value)
		if len(monthly_series) < 6:
			st.warning("Not enough monthly history for a reliable forecast on this selection.")
			return

		result = train_forecast_model(f"{choice_type}:{selected_value}", monthly_series)
		future_forecast = forecast_future(monthly_series, result["model"], result["feature_columns"], horizon)

		metric_col1, metric_col2, metric_col3 = st.columns(3)
		metric_col1.metric("MAE", f"{result['mae']:,.2f}")
		metric_col2.metric("RMSE", f"{result['rmse']:,.2f}")
		metric_col3.metric("Training Months", f"{len(monthly_series):,}")

		forecast_chart_df = pd.DataFrame({"Date": future_forecast.index, "Sales": future_forecast.values, "Type": "Forecast"})
		history_chart_df = pd.DataFrame({"Date": monthly_series.index, "Sales": monthly_series.values, "Type": "History"})
		chart_df = pd.concat([history_chart_df, forecast_chart_df], ignore_index=True)

		fig = px.line(chart_df, x="Date", y="Sales", color="Type", markers=True, title=f"{choice_type}: {selected_value} Forecast")
		fig.update_layout(xaxis_title="Month", yaxis_title="Sales")
		st.plotly_chart(fig, width="stretch")

		st.subheader("Forecast Output")
		forecast_table = pd.DataFrame({"Forecast Month": future_forecast.index.strftime("%Y-%m"), "Predicted Sales": future_forecast.values})
		st.dataframe(forecast_table, width="stretch", hide_index=True)

	except ValueError as exc:
		st.warning(str(exc))


def anomaly_report_page(df: pd.DataFrame) -> None:
	st.title("Anomaly Report")
	st.caption("Weekly sales anomalies detected with Isolation Forest and a rolling Z-Score check.")

	weekly_frame = build_anomaly_frame(df)
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=weekly_frame.index, y=weekly_frame["Sales"], mode="lines", name="Weekly Sales", line=dict(color="#2b6cb0", width=2)))
	iso = weekly_frame[weekly_frame["Isolation_Anomaly"]]
	zed = weekly_frame[weekly_frame["Z_Anomaly"] & ~weekly_frame["Isolation_Anomaly"]]
	if not iso.empty:
		fig.add_trace(go.Scatter(x=iso.index, y=iso["Sales"], mode="markers", name="Isolation Forest", marker=dict(color="#e53e3e", size=10, symbol="circle")))
	if not zed.empty:
		fig.add_trace(go.Scatter(x=zed.index, y=zed["Sales"], mode="markers", name="Z-Score", marker=dict(color="#dd6b20", size=10, symbol="x")))
	fig.update_layout(title="Task 5: Weekly Sales Anomaly Detection", xaxis_title="Week", yaxis_title="Sales", height=520)
	st.plotly_chart(fig, width="stretch")

	anomaly_table = weekly_frame[weekly_frame["Isolation_Anomaly"] | weekly_frame["Z_Anomaly"]].copy()
	anomaly_table = anomaly_table.reset_index()
	anomaly_table = anomaly_table.rename(columns={anomaly_table.columns[0]: "Week"})
	anomaly_table["Week"] = anomaly_table["Week"].dt.strftime("%Y-%m-%d")
	display_table = anomaly_table[["Week", "Sales", "Rolling_Mean_4", "Z_Score", "Method"]].sort_values("Week", ascending=False)
	st.subheader("Detected Anomaly Dates")
	st.dataframe(display_table, width="stretch", hide_index=True)


def cluster_page(df: pd.DataFrame) -> None:
	st.title("Product Demand Segments")
	st.caption("Sub-categories grouped into demand clusters using K-Means and PCA visualization.")

	cluster_frame = build_cluster_frame(df)
	fig = px.scatter(
		cluster_frame,
		x="pca_1",
		y="pca_2",
		color="cluster_label",
		text="Sub-Category",
		hover_data={"cluster": True, "total_sales_volume": ":,.2f", "sales_growth_rate_yoy": ":.2f", "sales_volatility": ":,.2f"},
		title="Task 6: Product Demand Segmentation by Sub-Category",
	)
	fig.update_traces(textposition="top center")
	fig.update_layout(xaxis_title="PCA Component 1", yaxis_title="PCA Component 2", height=600)
	st.plotly_chart(fig, width="stretch")

	cluster_table = cluster_frame[["Sub-Category", "cluster", "cluster_label", "total_sales_volume", "sales_growth_rate_yoy", "sales_volatility", "average_order_value"]].sort_values(
		["cluster", "total_sales_volume"], ascending=[True, False]
	)
	st.subheader("Sub-Category to Demand Cluster Mapping")
	st.dataframe(cluster_table, width="stretch", hide_index=True)


def main() -> None:
	df = load_data()

	st.sidebar.title("Navigation")
	page = st.sidebar.radio(
		"Choose a page",
		["Sales Overview Dashboard", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
	)

	st.sidebar.markdown("---")
	st.sidebar.write("Dataset")
	st.sidebar.write(f"Rows: {len(df):,}")
	st.sidebar.write(f"Categories: {df['Category'].nunique():,}")
	st.sidebar.write(f"Regions: {df['Region'].nunique():,}")

	if page == "Sales Overview Dashboard":
		sales_overview_page(df)
	elif page == "Forecast Explorer":
		forecast_explorer_page(df)
	elif page == "Anomaly Report":
		anomaly_report_page(df)
	else:
		cluster_page(df)

	st.sidebar.markdown("---")
	st.sidebar.info("Deploy this app on Streamlit Community Cloud after pushing the repo with requirements.txt.")


if __name__ == "__main__":
	main()
