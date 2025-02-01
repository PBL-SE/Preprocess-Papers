import pandas as pd

# Load the citations CSV
df_citations = pd.read_csv('C:/Homework/Second Year PICT/4th Sem/PBL/Analyze-Papers/output/citations.csv')

# Remove rows where 'cited_doi_id' is NaN or empty string
df_citations_cleaned = df_citations.dropna(subset=['cited_doi_id'])  # Drop NaN
df_citations_cleaned = df_citations_cleaned[df_citations_cleaned['cited_doi_id'].str.strip() != '']  # Drop empty strings

# Save the cleaned citations to a new CSV
df_citations_cleaned.to_csv('C:/Homework/Second Year PICT/4th Sem/PBL/Analyze-Papers/output/cleaned_citations.csv', index=False)

# Print a sample of cleaned data
print(df_citations_cleaned.head())
