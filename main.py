import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from textblob import TextBlob

df = pd.read_csv('file1.csv')
print(f"First five rows of data: \n{df.head()}")
print(f"Last five rows of data: \n{df.tail()}")
print(f"Info of this Data set \n {df.info()}")
print(f"Statistical Summary \n {df.describe()}")

if df.isnull().values.any():
    print(f"There are NUll values\n{df.isnull().sum()}")
else :
    print("There are no null values")

df.columns = df.columns.str.strip()
num_cols = [
    "CustomerID", "Tenure_Months", "Transaction_ID", "Quantity",
    "Avg_Price", "Delivery_Charges", "GST", "Offline_Spend",
    "Online_Spend", "Discount_pct"
]
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

cat_cols = [
    "Gender", "Location", "Product_SKU", "Product_Description",
    "Coupon_Status", "Coupon_Code"
]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

date_cols = ["Transaction_Date", "Date"]
for col in date_cols:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()

print(f"After Fill te missing values\n{df.isnull().sum()}")
if not df.duplicated().any():
    print(f"There are no duplicate values\n{df.duplicated().sum()}")
np.random.seed(42)
df["Rating"] = np.random.randint(1, 6, size=len(df))

df["Total"] = df["Avg_Price"] * df["Quantity"]

product_summary = df.groupby("Product_SKU").agg(
    Total_Revenue=('Total', 'sum'),
    Average_Product_Price=('Avg_Price', 'mean'),
    Maximum_Product_Price=('Avg_Price', 'max')
)
print(product_summary)

top_Selling = df["Product_SKU"].value_counts().idxmax()
print(f"Top Selling Product is : {top_Selling}")
city_highest_sale = df["Location"].value_counts().idxmax()
print(f"City With Highest Sale is : {city_highest_sale}")
avg_rating = df["Rating"].mean()
print(f"Average Customer Rating: {avg_rating:.2f}")

top_20_products = df.groupby("Product_SKU")["Total"].sum().reset_index()
top_20_products = top_20_products.sort_values(by="Total", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Total", y="Product_SKU", data=top_20_products, hue="Total" , palette="viridis")
plt.title("Top 10 Selling Products by Total Revenue")
plt.xlabel("Total Sales / Revenue")
plt.ylabel("Product SKU")
plt.tight_layout()
plt.show()

categories_count = df["Product_Category"].value_counts()
total_categories = categories_count.sum()
plt.figure(figsize=(12, 8))

plt.pie (
    categories_count,
    labels=None,
    autopct=None,
    startangle=140,
    colors=sns.color_palette("viridis", len(categories_count))
)


legend_labels = []
for name, count in categories_count.items():
    percentage = (count / total_categories) * 100
    legend_labels.append(f'{name} ({percentage:.1f}%)')
plt.legend(
    labels=legend_labels,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)

plt.title("Distribution of Product Categories")
plt.tight_layout()
plt.show()

sns.countplot(x = "Location" , hue = "Gender", data = df, palette = "pastel")
plt.title("Customer By City")
plt.show()

X_tenure = df[["Tenure_Months"]]
Y_tenure = df[["Total"]]

tenure_model = LinearRegression()
tenure_model.fit(X_tenure, Y_tenure)

new_Data_tenure = pd.DataFrame({"Tenure_Months" : [14]})
predict_tenure = tenure_model.predict(new_Data_tenure)
print(f"Prediction for 14 Months Tenure: {predict_tenure}")

X_rev = df[["Avg_Price" , "Quantity" , "Rating"]]
y_rev = df[["Total"]]

X_train, X_test, y_train, y_test = train_test_split(X_rev, y_rev, test_size = 0.2, random_state = 42)

revenue_model = LinearRegression()
revenue_model.fit(X_train, y_train)

new_Data_rev = pd.DataFrame({
    "Avg_Price" : [2500],
    "Quantity" : [25],
    "Rating" : [4.5]
})
predict_rev = revenue_model.predict(new_Data_rev)
print(f"Prediction for Price 2500 & Qty 25: {predict_rev}")

y_pred = revenue_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
accuracy2 = revenue_model.score(X_test, y_test)
print(f"Mean Absolute Error (MAE): {mae:.2f} Units")
print(f"Model Accuracy (R2) : {accuracy2:.2f} Units")
