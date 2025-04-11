import pandas as pd
from prophet import Prophet

# Convert DataFrame to string summary
def df_to_summary_string(df: pd.DataFrame) -> str:
    lines = [
        f"Date: {row['Date']}, Service: {row['Service']}, Cost: ${row['Cost']}, Tag: {row['Tag']}"
        for _, row in df.iterrows()
    ]
    return "\n".join(lines)

def extract_total_cost(df: pd.DataFrame) -> float:
    return df['Cost'].sum()

def extract_average_cost(df: pd.DataFrame) -> float:
    return df.groupby(df["Date"].dt.date)["Cost"].sum().mean()

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    daily_costs = df.groupby(df["Date"].dt.date)["Cost"].sum().reset_index()
    daily_costs.columns = ["Date", "Cost"]

    mean = daily_costs["Cost"].mean()
    std = daily_costs["Cost"].std()
    threshold_upper = mean + 2 * std
    threshold_lower = mean - 2 * std

    anomalies = daily_costs[(daily_costs["Cost"] > threshold_upper) | (daily_costs["Cost"] < threshold_lower)]
    return anomalies.rename(columns={"Cost": "Anomalous Cost"})

def suggest_optimizations(df: pd.DataFrame) -> list:
    """
    Suggest AWS S3 cost optimizations based on tags, service usage, and cost patterns.
    """
    tips = []

    # 1. High-cost tags
    if 'Tag' in df.columns and df['Tag'].notnull().any():
        tag_costs = df.groupby('Tag')['Cost'].sum().sort_values(ascending=False)
        top_tag = tag_costs.index[0]
        top_cost = tag_costs.iloc[0]
        if top_cost > 50:  # Arbitrary threshold, can be made dynamic
            tips.append(f"Tag '{top_tag}' has the highest total cost (${top_cost:.2f}). Consider reviewing or optimizing these resources.")

        if tag_costs.get("backup", 0) > 5:
            tips.append("Frequent 'backup' tag usage detected. Consider archiving old backups to Glacier or similar storage.")

    # 2. Frequent service usage
    if 'Service' in df.columns and df['Service'].notnull().any():
        top_service = df['Service'].value_counts().idxmax()
        count = df['Service'].value_counts().max()
        tips.append(f"Service '{top_service}' appears most frequently ({count} times). Evaluate if usage is justified or can be optimized.")

    # 3. Zero-cost detection (potential idle resources)
    if df['Cost'].min() == 0:
        tips.append("Some entries show $0 cost — check for idle or unused resources and consider stopping or removing them.")

    # 4. Cost spike warning
    if df['Cost'].max() > 100:
        tips.append(f"High cost spike detected: ${df['Cost'].max():.2f}. Review recent changes or resource overuse.")

    # 5. Suggest enabling S3 lifecycle policies
    if 'Tag' in df.columns and 'archive' not in df['Tag'].unique():
        tips.append("No 'archive' tag detected. Consider applying S3 lifecycle policies to auto-transition old data to lower-cost tiers.")

    return tips[:3] if tips else ["Usage appears optimized based on current data."]

def cost_forecast(df: pd.DataFrame) -> dict:
    from prophet import Prophet
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage
    from langchain.prompts import PromptTemplate

    # Step 1: Clean & validate data
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Cost'])
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
    df = df.dropna(subset=['Cost'])

    if df.empty:
        return {
            "forecast": "No valid data after cleaning.",
            "raw_prediction": None
        }

    # Step 2: Aggregate for Prophet
    daily_costs = df.groupby('Date')['Cost'].sum().reset_index()
    prophet_input = daily_costs.rename(columns={"Date": "ds", "Cost": "y"})

    if len(prophet_input) < 2:
        return {
            "forecast": "Not enough historical data to forecast.",
            "raw_prediction": None
        }

    # Step 3: Fit Prophet & forecast
    model = Prophet()
    model.fit(prophet_input)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Step 4: Extract next 30-day forecast
    today = datetime.today().date()
    forecast_start = today + timedelta(days=1)
    forecast_end = today + timedelta(days=30)
    forecast_period = forecast[
        (forecast['ds'].dt.date >= forecast_start) &
        (forecast['ds'].dt.date <= forecast_end)
    ]

    if forecast_period.empty:
        return {
            "forecast": "Forecast period is empty.",
            "raw_prediction": None
        }

    predicted_cost = round(forecast_period['yhat'].sum(), 2)

    # Step 5: Compare with last 3 months average
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_totals = df.groupby('Month')['Cost'].sum()
    if len(monthly_totals) < 3:
        last_3_months_avg = monthly_totals.mean()
    else:
        last_3_months_avg = monthly_totals[-3:].mean()

    delta_percent = ((predicted_cost - last_3_months_avg) / last_3_months_avg) * 100 if last_3_months_avg else 0
    trend = "increase" if delta_percent > 0 else "decrease"

    # Step 6: LLM Summary Prompt
    summarize_forecast_prompt = PromptTemplate.from_template("""
    The predicted S3 cost for next month is ${forecast_cost}.
    This is a {trend_direction} of {delta_percent}% compared to the recent average.

    Summarize this in a FinOps advisor tone, highlighting potential causes like backup uploads, traffic spikes, or seasonal changes.
    Provide an actionable recommendation if necessary.
    """)
    import os
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")

    forecast_input = summarize_forecast_prompt.format(
        forecast_cost=predicted_cost,
        trend_direction=trend,
        delta_percent=abs(round(delta_percent, 2))
    )

    result = llm.invoke(forecast_input)
    forecast_summary = result.content if isinstance(result, AIMessage) else str(result)

    return {
        "forecast": forecast_summary,
        "raw_prediction": forecast_period
    }