import pandas as pd
import re
import numpy as np

# Load the original CSV
input_csv = "C:/Homework/Second Year PICT/4th Sem/PBL/Testing/output/updated_arxiv_dump.csv"
df = pd.read_csv(input_csv)

# Rename columns to match database schema
df.columns = ['arxiv_id', 'title', 'authors', 'year', 'pdf_link', 'category', 'DOI', 'cited_papers', 'comment', 'journal', 'summary', 'difficulty_level']

df['year'] = pd.to_datetime(df['year']).dt.year

# Extract author names and organizations
def extract_authors(authors_str):
    pattern = r"(.+?) \((.*?)\)"
    matches = re.findall(pattern, str(authors_str))
    return [(name.strip().lstrip(','), org.strip() if org != "Unknown" else None) for name, org in matches]

df['authors_parsed'] = df['authors'].apply(extract_authors)

authors_data = []
organizations_data = set()
paper_author_data = []

for _, row in df.iterrows():
    arxiv_id = row['arxiv_id']
    for author, org in row['authors_parsed']:
        authors_data.append((author, org))
        if org:
            organizations_data.add(org)
        paper_author_data.append((arxiv_id, author))

# Convert to DataFrame
df_authors = pd.DataFrame(authors_data, columns=['author_name', 'organization']).drop_duplicates().reset_index(drop=True)
df_organizations = pd.DataFrame(list(organizations_data), columns=['organization']).drop_duplicates().reset_index(drop=True)
df_organizations['organization_id'] = df_organizations.index + 1

# Assign organization IDs to authors
df_authors = df_authors.merge(df_organizations, on='organization', how='left')
df_authors['author_id'] = df_authors.index + 1

df_paper_author = pd.DataFrame(paper_author_data, columns=['arxiv_id', 'author_name'])
df_paper_author = df_paper_author.merge(df_authors[['author_name', 'author_id']], on='author_name', how='left')

# Map difficulty levels
difficulty_mapping = {"Beginner": 0, "Intermediate": 1, "Pro": 2, "Expert": 3}
df['difficulty_level'] = df['difficulty_level'].map(difficulty_mapping)

# Papers Table
df_papers = df[['arxiv_id', 'title', 'summary', 'comment', 'year', 'journal', 'difficulty_level', 'pdf_link', 'DOI']]

# Remove duplicates based on title
def remove_duplicates(df):
    # Helper function to count missing values (NaN)
    def count_missing(row):
        return row.isna().sum()
    
    # Sort by title and prioritize the row with fewer NaN values
    df_sorted = df.sort_values(by='title')
    df_sorted['missing_values_count'] = df_sorted.apply(count_missing, axis=1)
    
    # Remove duplicates based on title and keep row with least NaN values
    df_dedup = df_sorted.drop_duplicates('title', keep='first')
    
    return df_dedup.drop('missing_values_count', axis=1)

df_papers = remove_duplicates(df_papers)

# Categories Table
df_categories = df['category'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).reset_index()
df_categories.columns = ['index', 'name']
df_categories = df_categories[['name']].drop_duplicates().reset_index(drop=True)
df_categories['category_id'] = df_categories.index + 1

# Paper_Category Table
df_paper_category = df[['arxiv_id', 'category']].dropna()
df_paper_category = df_paper_category.assign(category=df_paper_category['category'].str.split(','))
df_paper_category = df_paper_category.explode('category').merge(df_categories, left_on='category', right_on='name', how='left')

# Citations Table
df_citations = df[['arxiv_id', 'cited_papers']].dropna()
df_citations = df_citations.assign(cited_papers=df_citations['cited_papers'].str.split(','))
df_citations = df_citations.explode('cited_papers')
df_citations.columns = ['citing_arxiv_id', 'cited_doi_id']

df_paper_author = df_paper_author[df_paper_author['arxiv_id'].isin(df_papers['arxiv_id'])]
df_paper_category = df_paper_category[df_paper_category['category_id'].isin(df_categories['category_id'])]
df_paper_category = df_paper_category[df_paper_category['arxiv_id'].isin(df_papers['arxiv_id'])]
df_citations = df_citations[df_citations['citing_arxiv_id'].isin(df_papers['arxiv_id'])]
df_citations = df_citations.dropna(subset=['cited_doi_id'])

# Save CSVs for SQL import
df_papers.to_csv("output/papers.csv", index=False)
df_authors[['author_id', 'author_name', 'organization_id']].to_csv("output/authors.csv", index=False)
df_organizations.to_csv("output/organizations.csv", index=False)
df_paper_author[['arxiv_id', 'author_id']].to_csv("output/paper_author.csv", index=False)
df_categories.to_csv("output/categories.csv", index=False)
df_paper_category[['arxiv_id', 'category_id']].to_csv("output/paper_category.csv", index=False)
df_citations.to_csv("output/citations.csv", index=False)

print("CSV files successfully created for SQL import!")
