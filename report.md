## Entries

- datasets
  - 8 categries each 25000 data
    - children
    - comics_graphic
    - fantasy_paranormal
    - history_biography
    - mystery_thriller_crime
    - poetry
    - romance
    - young_adult
  - train / test split -> 20000 / 5000
- statistics
  - for entire dataset/each category
    - wordcloud
    - average words
    - grams
      - size
        - uni
        - bi
        - tri
      - feature
        - ngram top N=? words
        - tfidf top N=? words
        - fasttext top N=? words
- models
  - baseline -> Random
  - NB
  - Logistic R
  - SVM
  - MLP
- training loss
  - cross entropy
- evaluation
  - classification error