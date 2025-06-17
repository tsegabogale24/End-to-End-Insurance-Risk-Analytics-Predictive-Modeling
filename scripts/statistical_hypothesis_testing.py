
import pandas as pd
import numpy as np
from scipy import stats



def compute_kpis(df: pd.DataFrame, avg_claim_amount: float = 1000) -> pd.DataFrame:
    # Claim occurred if TotalClaims > 0
    df['ClaimOccurred'] = df['TotalClaims'] > 0
    df['ClaimFrequency'] = df['ClaimOccurred'].astype(int)

    # Approximate Margin: subtract estimated total claim cost
    df['EstimatedClaimAmount'] = df['TotalClaims'] * avg_claim_amount
    df['Margin'] = df['TotalPremium'] - df['EstimatedClaimAmount']

    return df

def chi_square_by_province(df: pd.DataFrame):
    contingency = pd.crosstab(df['Province'], df['ClaimOccurred'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p

def chi_square_by_zip(df: pd.DataFrame, zip_a, zip_b):
    df_filtered = df[df['PostalCode'].isin([zip_a, zip_b])]
    contingency = pd.crosstab(df_filtered['PostalCode'], df_filtered['ClaimOccurred'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p

def t_test_margin_by_zip(df: pd.DataFrame, zip_a, zip_b):
    margin_a = df[df['PostalCode'] == zip_a]['Margin'].dropna()
    margin_b = df[df['PostalCode'] == zip_b]['Margin'].dropna()
    t_stat, p_value = stats.ttest_ind(margin_a, margin_b, equal_var=False)
    return t_stat, p_value

def t_test_by_gender(df: pd.DataFrame):
    freq_m = df[df['Gender'] == 'M']['ClaimFrequency']
    freq_f = df[df['Gender'] == 'F']['ClaimFrequency']
    # For frequency (binary), use proportions z-test or chi-square:
    contingency = pd.crosstab(df['Gender'], df['ClaimOccurred'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return chi2, p