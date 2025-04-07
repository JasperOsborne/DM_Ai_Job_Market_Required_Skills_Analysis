#%% Imports and Data Loading
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import process, fuzz

#%% Load Datasets
ai_df = pd.read_csv('ai_job_market_insights.csv')
job_df = pd.read_csv('job_recommendation_dataset.csv')

#%% Standardise Column Names for Merging
ai_df.rename(columns={
    'Job_Title': 'Job Title',
    'Salary_USD': 'Salary',
    'Required_Skills': 'Required Skills'
}, inplace=True)

#%% Sample and Fuzzy Matching by Job Title and Industry
job_sample_df = job_df[['Job Title', 'Industry', 'Location', 'Required Skills', 'Salary']].head(2000)
ai_jobs_by_industry = {
    industry: ai_df[ai_df['Industry'] == industry]['Job Title'].unique()
    for industry in ai_df['Industry'].unique()
}

matches = []
for _, row in job_sample_df.iterrows():
    job_title = row['Job Title']
    industry = row['Industry']
    if industry in ai_jobs_by_industry:
        best_match = process.extractOne(job_title, ai_jobs_by_industry[industry], scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= 90:
            matches.append({
                'Job Title': job_title,
                'Matched Title': best_match[0],
                'Industry': industry,
                'Location': row['Location'],
                'Required Skills': row['Required Skills'],
                'Salary': row['Salary']
            })

matched_sample_df = pd.DataFrame(matches)

#%% Merge with AI Dataset Using Fuzzy-Matched Job Titles
merged_fuzzy_sample = pd.merge(
    matched_sample_df,
    ai_df,
    left_on=['Matched Title', 'Industry'],
    right_on=['Job Title', 'Industry'],
    how='inner'
)

#%% Preprocess and Vectorise Skills
merged_fuzzy_sample['Combined Skills'] = merged_fuzzy_sample['Required Skills_x'].fillna('') + ',' + merged_fuzzy_sample['Required Skills_y'].fillna('')
merged_fuzzy_sample['Combined Skills'] = merged_fuzzy_sample['Combined Skills'].str.lower().str.split(',')
merged_fuzzy_sample['Combined Skills'] = merged_fuzzy_sample['Combined Skills'].apply(lambda x: [i.strip() for i in x if i.strip()])

mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(merged_fuzzy_sample['Combined Skills'])
skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

#%% Prepare Features and Targets
X = skills_df
y_adoption = merged_fuzzy_sample['AI_Adoption_Level']
y_risk = merged_fuzzy_sample['Automation_Risk']

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y_adoption, test_size=0.3, random_state=11)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_risk, test_size=0.3, random_state=11)

#%% Train Models
rf_adoption = RandomForestClassifier(random_state=11)
rf_adoption.fit(X_train_a, y_train_a)
y_pred_adoption = rf_adoption.predict(X_test_a)

rf_risk = RandomForestClassifier(random_state=11)
rf_risk.fit(X_train_r, y_train_r)
y_pred_risk = rf_risk.predict(X_test_r)

#%% Print Model Performance
print("AI Adoption Classification Report:")
print(classification_report(y_test_a, y_pred_adoption))

print("Automation Risk Classification Report:")
print(classification_report(y_test_r, y_pred_risk))

#%% Feature Importance Visualisations
adoption_importance_df = pd.DataFrame({
    'Skill': X.columns,
    'Importance': rf_adoption.feature_importances_
}).sort_values(by='Importance', ascending=False).head(15)

risk_importance_df = pd.DataFrame({
    'Skill': X.columns,
    'Importance': rf_risk.feature_importances_
}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Skill', data=adoption_importance_df)
plt.title('Top 15 Skills Influencing AI Adoption Level')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Skill', data=risk_importance_df)
plt.title('Top 15 Skills Influencing Automation Risk')
plt.tight_layout()
plt.show()
#%% Improved Skill Frequency by Group

def get_top_skills_by_group_clean(df, group_col, skills_column, top_n=15):
    # Flatten skills per group
    grouped = df[[group_col, skills_column]].explode(skills_column)
    
    # Count skills per group
    skill_counts = grouped.groupby([group_col, skills_column]).size().unstack(fill_value=0)
    
    # Get top N across all groups combined (for fair comparison)
    top_skills = skill_counts.sum(axis=0).sort_values(ascending=False).head(top_n).index
    
    return skill_counts[top_skills].T  # transpose for readability

# Redefine cleaned list (in case cells were re-run)
merged_fuzzy_sample['Combined Skills'] = merged_fuzzy_sample['Combined Skills'].apply(lambda x: [i.strip() for i in x if i.strip()])

# Run and print results
top_risk_skills_clean = get_top_skills_by_group_clean(merged_fuzzy_sample, 'Automation_Risk', 'Combined Skills')
print("Top Skills by Automation Risk Level (cleaned):")
print(top_risk_skills_clean)

top_adoption_skills_clean = get_top_skills_by_group_clean(merged_fuzzy_sample, 'AI_Adoption_Level', 'Combined Skills')
print("\nTop Skills by AI Adoption Level (cleaned):")
print(top_adoption_skills_clean)

#%% Export Final Merged Dataset with Skill Features
final_merged_export = pd.concat([merged_fuzzy_sample.reset_index(drop=True), skills_df.reset_index(drop=True)], axis=1)
final_merged_export.to_csv('merged_ai_job_analysis.csv', index=False)
print("Exported to 'merged_ai_job_analysis.csv'")
