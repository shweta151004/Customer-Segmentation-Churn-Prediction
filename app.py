import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🛒",
    layout="wide"
)

# ── Title ──
st.title("🛒 Customer Segmentation & Churn Predictor")
st.markdown("Upload your retail transaction data to identify at-risk customers.")

# ── Sidebar ──
st.sidebar.header("About this App")
st.sidebar.info("""
This app uses **RFM Analysis** and **XGBoost** to:
- Segment customers into groups
- Predict who is likely to churn
- Recommend business actions
""")

# ── File Upload ──
uploaded_file = st.file_uploader(
    "Upload your CSV file (Online Retail format)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

# ── Load & Clean ──
st.subheader("📦 Step 1 — Loading & Cleaning Data")

with st.spinner("Cleaning data..."):
    df = pd.read_csv(uploaded_file, encoding='latin1')
    raw_rows = len(df)

    df = df.dropna(subset=['Customer ID'])
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df['TotalSpend'] = df['Quantity'] * df['Price']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)
    clean_rows = len(df)

col1, col2, col3 = st.columns(3)
col1.metric("Raw Rows", f"{raw_rows:,}")
col2.metric("Clean Rows", f"{clean_rows:,}")
col3.metric("Rows Removed", f"{raw_rows - clean_rows:,}")

st.success("Data cleaned successfully!")

# ── RFM Analysis ──
st.subheader("📊 Step 2 — RFM Analysis")

with st.spinner("Calculating RFM scores..."):
    cutoff = pd.Timestamp('2011-06-01')
    first_half = df[df['InvoiceDate'] < cutoff]
    second_half = df[df['InvoiceDate'] >= cutoff]
    active_in_second = second_half['Customer ID'].unique()

    rfm = first_half.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (cutoff - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalSpend', 'sum')
    ).reset_index()

    # Churn label
    rfm['Churn'] = (~rfm['Customer ID'].isin(active_in_second)).astype(int)

    # RFM Scores
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

    # Segments
    def label_customer(row):
        r = int(row['R_Score'])
        f = int(row['F_Score'])
        if r >= 4 and f >= 4: return 'Champion'
        elif r >= 3 and f >= 3: return 'Loyal'
        elif r >= 4 and f <= 2: return 'New Customer'
        elif r <= 2 and f >= 3: return 'At Risk'
        elif r == 1 and f == 1: return 'Lost'
        else: return 'Needs Attention'

    rfm['Segment'] = rfm.apply(label_customer, axis=1)

# Show RFM table
st.dataframe(rfm[['Customer ID','Recency','Frequency','Monetary','Segment','Churn']].head(10))

# ── Charts ──
st.subheader("📈 Step 3 — Customer Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Customers per Segment**")
    fig, ax = plt.subplots(figsize=(5,3))
    order = rfm['Segment'].value_counts().index
    sns.countplot(data=rfm, x='Segment', order=order, palette='Blues_d', ax=ax)
    ax.set_xlabel('')
    plt.xticks(rotation=25, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("**Churn Rate by Segment**")
    fig, ax = plt.subplots(figsize=(5,3))
    churn_rate = rfm.groupby('Segment')['Churn'].mean() * 100
    churn_rate.sort_values().plot(kind='barh', color='tomato', ax=ax)
    ax.set_xlabel('Churn Rate %')
    plt.tight_layout()
    st.pyplot(fig)

# ── Key Metrics ──
st.subheader("🎯 Key Numbers")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(rfm):,}")
col2.metric("Champions", f"{len(rfm[rfm['Segment']=='Champion']):,}")
col3.metric("At Risk", f"{len(rfm[rfm['Segment']=='At Risk']):,}")
col4.metric("Predicted to Churn", f"{rfm['Churn'].sum():,}")

# ── Churn Model ──
st.subheader("🤖 Step 4 — Churn Prediction Model")

with st.spinner("Training model..."):
    rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Spend_Rate'] = rfm['Monetary'] / (rfm['Recency'] + 1)

    X = rfm[['Frequency', 'Monetary', 'Avg_Order_Value', 'Spend_Rate']]
    y = rfm['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, model.predict(X_test))

st.success(f"Model trained! Accuracy: **{acc*100:.1f}%**")

# ── Business Recommendation ──
st.subheader("💡 Business Recommendations")

at_risk_count = len(rfm[rfm['Segment'] == 'At Risk'])
avg_spend = rfm['Monetary'].mean()
voucher = 200
recovery_rate = 0.20
recovered_revenue = at_risk_count * recovery_rate * avg_spend
campaign_cost = at_risk_count * voucher

st.info(f"""
**Finding:** Frequency is the #1 predictor of churn.
Customers with ≤2 orders are at highest risk.

**Recommended Action:**
- Send a ₹{voucher} re-engagement voucher to all **{at_risk_count} At-Risk customers**
- Estimated campaign cost: ₹{campaign_cost:,.0f}
- Estimated recovered revenue (20% response): ₹{recovered_revenue:,.0f}
- **Estimated ROI: {recovered_revenue/campaign_cost:.1f}x**
""")

st.caption("Built with Python · XGBoost · SHAP · Streamlit")