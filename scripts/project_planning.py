import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def convert_categoricals(df: pd.DataFrame, extra_categoricals=None, max_unique_threshold=50) -> pd.DataFrame:
    # Step 1: Auto-detect object/string columns with limited unique values, excluding 'TransactionMonth'
    auto_categoricals = [
        col for col in df.select_dtypes(include=["object", "string"]).columns
        if col != "TransactionMonth" and df[col].nunique(dropna=False) <= max_unique_threshold
    ]

    # Step 2: Add any extra manually specified categorical columns
    if extra_categoricals is None:
        extra_categoricals = [
            "Province", "Gender", "VehicleType", "Language", "MaritalStatus",
            "CoverType", "CoverGroup", "StatutoryClass", "StatutoryRiskType",
            "Title", "LegalType", "Citizenship", "Country", "ItemType"
        ]

    # Step 3: Combine and deduplicate, and exclude 'TransactionMonth' explicitly
    all_categoricals = list(set(auto_categoricals + extra_categoricals))
    if "TransactionMonth" in all_categoricals:
        all_categoricals.remove("TransactionMonth")

    # Step 4: Convert in place
    for col in all_categoricals:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"Converted {len(all_categoricals)} columns to 'category':\n{all_categoricals}")
    return df

def plot_univariate_distributions(df, numerical_cols, categorical_cols):
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
        plt.show()


def plot_correlation_analysis(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="TotalPremium", y="TotalClaims", hue="PostalCode", alpha=0.5, legend=False)
    plt.title("TotalPremium vs TotalClaims colored by PostalCode")
    plt.show()

    corr = df[["TotalPremium", "TotalClaims"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def plot_geographic_trends(df, metric="TotalClaims"):
    province_stats = df.groupby("Province")[metric].mean().sort_values()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=province_stats.values, y=province_stats.index, palette="viridis")
    plt.title(f"Average {metric} by Province")
    plt.xlabel(metric)
    plt.ylabel("Province")
    plt.show()


def plot_outliers(df, cols):
    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Outlier Detection for {col}')
        plt.show()


def plot_temporal_trends(df):
    df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")
    monthly = df.set_index("TransactionMonth").resample("M")[["TotalPremium", "TotalClaims"]].sum()
    monthly.plot(figsize=(10, 5), marker='o', title="Monthly Trends: Premiums vs Claims")
    plt.ylabel("Amount")
    plt.xlabel("Month")
    plt.grid(True)
    plt.show()


def plot_make_vs_claims(df):
    top_makes = df.groupby("make")["TotalClaims"].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_makes.values, y=top_makes.index, palette="magma")
    plt.title("Top Vehicle Makes by Avg. Claim")
    plt.xlabel("Average Claims")
    plt.ylabel("Vehicle Make")
    plt.show()


def summarize_statistics(df, cols):
    summary = df[cols].describe().T
    summary["variance"] = df[cols].var()
    return summary
