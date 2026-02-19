Nexus Mods Analysis
====================

A data pipeline and analysis project that collects, cleans, and analyses
mod and modder data from Nexus Mods using their public APIs. The final
analysis computes a "Devotion Score" and applies clustering techniques
to categorise modder behaviour, supporting an accompanying research paper.

License: Apache 2.0 (see LICENSE)


Project Structure
-----------------

  Part_1_API_Data_Collection.ipynb   Collect mod and author data via the
                                     Nexus Mods REST API v1 and GraphQL
                                     API v2, writing results to Azure SQL.

  Part_2_Data_Cleaning.ipynb         Translate non-English descriptions,
                                     extract Patreon URLs, and normalise
                                     game categories with fuzzy matching.

  Part_3_Parquet_Export.ipynb         Export SQL tables and views to Parquet
                                     files and create ZIP archives for
                                     data sharing.

  Part_4_Data_Investigation.ipynb     Exploratory data analysis and
                                     descriptive statistics.

  Part_5_Final_Analysis.ipynb        Devotion Score calculation, clustering
                                     (KMeans, DBSCAN, GMM), PCA, and
                                     statistical tests for the paper.

  Archive/                            Legacy notebooks, scripts, CSV/JSON
                                     data snapshots, and Azure Functions
                                     code kept for reference.


Data Sources
------------

  - Nexus Mods REST API v1  : api.nexusmods.com/v1/games/{domain}/mods/{id}
  - Nexus Mods GraphQL API v2: api.nexusmods.com/v2/graphql


Prerequisites
-------------

  Python 3.10+

  Core libraries:
    pandas, numpy, sqlalchemy, pyodbc, requests, tqdm

  Data cleaning:
    fuzzywuzzy, python-Levenshtein, langdetect, deep-translator, beautifulsoup4

  Analysis and visualisation:
    scikit-learn, scipy, matplotlib, seaborn, plotly, ipywidgets, joblib

  Database:
    An Azure SQL Server instance with ODBC Driver 17 for SQL Server

  API key:
    A valid Nexus Mods API key stored in api_key.txt (not tracked by git)


Getting Started
---------------

  1. Clone the repository.
  2. Install dependencies:
       pip install pandas numpy sqlalchemy pyodbc requests tqdm \
                   fuzzywuzzy python-Levenshtein langdetect deep-translator \
                   beautifulsoup4 scikit-learn scipy matplotlib seaborn \
                   plotly ipywidgets joblib
  3. Create an api_key.txt file in the project root containing your
     Nexus Mods API key.
  4. Update the SQL Server connection string in each notebook to point
     to your Azure SQL instance.
  5. Run the notebooks in order (Part 1 through Part 5).
