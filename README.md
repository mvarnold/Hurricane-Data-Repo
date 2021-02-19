# Hurricane-Data-Repo

A repo containing all data used for analysis of *Hurricanes and hashtags: Characterizing online collective attention for natural disasters*

The three main data sources are seperated into directories: 
  - `wikidata` (containing general hurricane impacts)
  - `hurdat_data` (containing geospatial data of hurricane trajectories)
  - `ngram_data` (containing count and usage frequency data for ngrams associated with hurricanes in English tweets)
  
A pickled pandas dataframe object is saved as `cleaned_hurricane_updated.pkl`

## Source Code

Source code to rerun the regressions and produce the associated figures are provided in the `src` directory.

  - `category_regressions.py` -- runs our regression models for each category seperately (Figure 3).
  - `hurricane_regressions.py` -- runs regression models 1, 2, and 3, on only storms reaching Hurricane force winds.
