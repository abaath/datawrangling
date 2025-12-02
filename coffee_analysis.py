import pandas as pd
import requests
import matplotlib.pyplot as plt

df = pd.read_csv('coffee-shop-sales-revenue.csv', sep='|')
df = df[df['store_location'] == 'Lower Manhattan']
df['date'] = pd.to_datetime(df['transaction_date'])
df['hour'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S').dt.hour
df['revenue'] = df['transaction_qty'] * df['unit_price']

response = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        'latitude': 40.7128,
        'longitude': -74.0060,
        'start_date': '2023-01-01',
        'end_date': '2023-06-30',
        'daily': 'temperature_2m_mean,precipitation_sum',
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York'
    }
)

weather_data = response.json()['daily']
weather_df = pd.DataFrame({
    'date': pd.to_datetime(weather_data['time']),
    'temperature': weather_data['temperature_2m_mean'],
    'rain': weather_data['precipitation_sum']
})

df = df.merge(weather_df, on='date', how='left')

# ANALYSIS OUTPUT
print("\n" + "="*60)
print("COFFEE SHOP ANALYSIS - LOWER MANHATTAN")
print("="*60 + "\n")

print("BASIC NUMBERS:")
print(f"Total transactions: {len(df):,}")
print(f"Total revenue: ${df['revenue'].sum():,.2f}")
print(f"Average per transaction: ${df['revenue'].mean():.2f}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Time period: 181 days (6 months)")
print(f"Average daily revenue: ${df.groupby('date')['revenue'].sum().mean():,.2f}")
print(f"Busiest day: ${df.groupby('date')['revenue'].sum().max():,.2f}")
print(f"Slowest day: ${df.groupby('date')['revenue'].sum().min():,.2f}")
print()

print("SALES BY PRODUCT CATEGORY:")
by_category = df.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
total_rev = by_category.sum()
for category, amount in by_category.items():
    pct = (amount / total_rev) * 100
    print(f"  {category}: ${amount:,.2f} ({pct:.1f}% of total)")
print()

print("TOP 5 PRODUCTS:")
top_products = df.groupby('product_detail')['revenue'].sum().sort_values(ascending=False).head(5)
for product, amount in top_products.items():
    print(f"  {product}: ${amount:,.2f}")
print()

print("SALES BY HOUR:")
by_hour = df.groupby('hour')['revenue'].sum()
by_hour_count = df.groupby('hour').size()
peak_hour = by_hour.idxmax()
slowest_hour = by_hour.idxmin()
print(f"  Peak hour: {peak_hour}:00 with ${by_hour[peak_hour]:,.2f}")
print(f"  Slowest hour: {slowest_hour}:00 with ${by_hour[slowest_hour]:,.2f}")
print(f"  Morning rush (7-11am): ${by_hour[7:12].sum():,.2f}")
print(f"  Afternoon (12-5pm): ${by_hour[12:18].sum():,.2f}")
print(f"  Evening (6pm+): ${by_hour[18:].sum():,.2f}")
print()

print("WEATHER CORRELATION:")
daily = df.groupby('date').agg({
    'revenue': 'sum',
    'temperature': 'first',
    'rain': 'first',
    'transaction_id': 'count'
}).reset_index()
daily.columns = ['date', 'revenue', 'temperature', 'rain', 'transactions']

correlation_temp = daily['revenue'].corr(daily['temperature'])
correlation_rain = daily['revenue'].corr(daily['rain'])

print(f"  Temperature vs Sales Correlation: {correlation_temp:.3f}")
if correlation_temp > 0.7:
    print(f"    STRONG positive correlation - Warmer days = MUCH higher sales")
    temp_cold = daily[daily['temperature'] < 40]['revenue'].mean()
    temp_warm = daily[daily['temperature'] > 70]['revenue'].mean()
    print(f"    Cold days (<40F) average daily revenue: ${temp_cold:,.2f}")
    print(f"    Warm days (>70F) average daily revenue: ${temp_warm:,.2f}")
    print(f"    Difference: ${temp_warm - temp_cold:,.2f} ({((temp_warm/temp_cold - 1)*100):.0f}% more on warm days)")
elif correlation_temp > 0.3:
    print(f"    MODERATE positive correlation - Warmer days have somewhat higher sales")
else:
    print(f"    WEAK correlation - Temperature doesn't strongly affect sales")

print(f"\n  Rainfall vs Sales Correlation: {correlation_rain:.3f}")
if abs(correlation_rain) < 0.2:
    print(f"    WEAK correlation - Rain barely affects sales")
    rainy = daily[daily['rain'] > 0]['revenue'].mean()
    no_rain = daily[daily['rain'] == 0]['revenue'].mean()
    print(f"    Rainy days average daily revenue: ${rainy:,.2f}")
    print(f"    Clear days average daily revenue: ${no_rain:,.2f}")
    print(f"    Difference: ${abs(no_rain - rainy):,.2f} (rain has minimal impact)")
elif correlation_rain < 0:
    print(f"    NEGATIVE correlation - More rain = less sales")
else:
    print(f"    POSITIVE correlation - More rain = more sales (unusual!)")

print(f"\n  Temperature Range: {daily['temperature'].min():.1f}F to {daily['temperature'].max():.1f}F")
print(f"  Average Temperature: {daily['temperature'].mean():.1f}F")
print(f"  Rainiest Day: {daily['rain'].max():.2f} inches")
print(f"  Days with Rain: {(daily['rain'] > 0).sum()} out of 181 days")
print()

# CHARTS
fig, charts = plt.subplots(1, 3, figsize=(15, 4))

by_hour = df.groupby('hour')['revenue'].sum()
charts[0].bar(by_hour.index, by_hour.values)
charts[0].set_title('Revenue by Hour')
charts[0].set_xlabel('Hour of Day')
charts[0].set_ylabel('Revenue ($)')

by_category = df.groupby('product_category')['revenue'].sum().sort_values()
charts[1].barh(by_category.index, by_category.values)
charts[1].set_title('Revenue by Product')
charts[1].set_xlabel('Revenue ($)')

charts[2].scatter(daily['temperature'], daily['revenue'])
charts[2].set_title('Warm Days = More Sales?')
charts[2].set_xlabel('Temperature (F)')
charts[2].set_ylabel('Daily Revenue ($)')

plt.tight_layout()
plt.savefig('chart.png')

# SAVE DATA
df.to_csv('combined_data.csv', index=False)
daily.to_csv('daily_summary.csv', index=False)

print("Files saved: chart.png, combined_data.csv, daily_summary.csv")