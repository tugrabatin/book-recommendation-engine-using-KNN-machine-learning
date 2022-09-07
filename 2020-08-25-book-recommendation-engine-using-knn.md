---
title: "Book Recommendation Engine Using KNN"
date: 2020-08-25
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Book Recommendation Engine Using KNN

This project entailed creating a book recommendation algorithm using K-Nearest Neighbors.

I was able to use the Book-Crossings dataset which was provided and created a impressively accurate book recommendation engine. This dataset contains 1.1 million ratings (scale of 1-10) of 270,000 books by 90,000 users.

## Project Instructions

*Note: You are currently reading this using Google Colaboratory which is a cloud-hosted version of Jupyter Notebook. This is a document containing both text cells for documentation and runnable code cells. If you are unfamiliar with Jupyter Notebook, watch this 3-minute introduction before starting this challenge: https://www.youtube.com/watch?v=inN8seMm7UI*

---

In this challenge, you will create a book recommendation algorithm using **K-Nearest Neighbors**.

You will use the [Book-Crossings dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/). This dataset contains 1.1 million ratings (scale of 1-10) of 270,000 books by 90,000 users. 

After importing and cleaning the data, use `NearestNeighbors` from `sklearn.neighbors` to develop a model that shows books that are similar to a given book. The Nearest Neighbors algorithm measures distance to determine the “closeness” of instances.

Create a function named `get_recommends` that takes a book title (from the dataset) as an argument and returns a list of 5 similar books with their distances from the book argument.

This code:

`get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")`

should return:

```
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301], 
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]
```

Notice that the data returned from `get_recommends()` is a list. The first element in the list is the book title passed in to the function. The second element in the list is a list of five more lists. Each of the five lists contains a recommended book and the distance from the recommended book to the book passed in to the function.

If you graph the dataset (optional), you will notice that most books are not rated frequently. To ensure statistical significance, remove from the dataset users with less than 200 ratings and books with less than 100 ratings.

The first three cells import libraries you may need and the data to use. The final cell is for testing. Write all your code in between those cells.


```python
# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
```


```python
# get data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'
```

    --2020-08-25 11:04:20--  https://cdn.freecodecamp.org/project-data/books/book-crossings.zip
    Resolving cdn.freecodecamp.org (cdn.freecodecamp.org)... 157.230.103.136, 167.99.137.12, 2a03:b0c0:3:e0::26f:c001, ...
    Connecting to cdn.freecodecamp.org (cdn.freecodecamp.org)|157.230.103.136|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 26085508 (25M) [application/zip]
    Saving to: ‘book-crossings.zip’
    
    book-crossings.zip  100%[===================>]  24.88M  8.20MB/s    in 3.0s    
    
    2020-08-25 11:04:23 (8.20 MB/s) - ‘book-crossings.zip’ saved [26085508/26085508]
    
    Archive:  book-crossings.zip
      inflating: BX-Book-Ratings.csv     
      inflating: BX-Books.csv            
      inflating: BX-Users.csv            
    


```python
# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})
```


```python
# add your code here - consider creating a new cell for each section of code
print(df_books)
```

                  isbn  ...                author
    0       0195153448  ...    Mark P. O. Morford
    1       0002005018  ...  Richard Bruce Wright
    2       0060973129  ...          Carlo D'Este
    3       0374157065  ...      Gina Bari Kolata
    4       0393045218  ...       E. J. W. Barber
    ...            ...  ...                   ...
    271374  0440400988  ...        Paula Danziger
    271375  0525447644  ...            Teri Sloat
    271376  006008667X  ...      Christine Wicker
    271377  0192126040  ...                 Plato
    271378  0767409752  ...   Christopher  Biffle
    
    [271379 rows x 3 columns]
    


```python
print(df_ratings)
```

               user         isbn  rating
    0        276725   034545104X     0.0
    1        276726   0155061224     5.0
    2        276727   0446520802     0.0
    3        276729   052165615X     3.0
    4        276729   0521795028     6.0
    ...         ...          ...     ...
    1149775  276704   1563526298     9.0
    1149776  276706   0679447156     0.0
    1149777  276709   0515107662    10.0
    1149778  276721   0590442449    10.0
    1149779  276723  05162443314     8.0
    
    [1149780 rows x 3 columns]
    


```python
# Calculate user and book rating counts
user_RatingCount = df_ratings.groupby('user')['rating'].count().reset_index().rename(columns = {'rating':'userTotalRatingCount'})
book_RatingCount = df_ratings.groupby('isbn')['rating'].count().reset_index().rename(columns = {'rating':'bookTotalRatingCount'})

# Add to df_ratings
df_ratings = df_ratings.merge(user_RatingCount,how='left', left_on='user', right_on='user')
df_ratings = df_ratings.merge(book_RatingCount, how='left', left_on='isbn', right_on='isbn')

# Filter data for statistical significance
df_ratings_2 =df_ratings.loc[(df_ratings['userTotalRatingCount']>=200) & (df_ratings['bookTotalRatingCount']>=100)]
```


```python
# merge data sets
books_with_ratings = pd.merge(df_ratings_2, df_books, on='isbn')
print(books_with_ratings)
```

             user  ...             author
    0      277427  ...  James Finn Garner
    1        3363  ...  James Finn Garner
    2       11676  ...  James Finn Garner
    3       12538  ...  James Finn Garner
    4       13552  ...  James Finn Garner
    ...       ...  ...                ...
    49512  238864  ...  Patricia Cornwell
    49513  251843  ...  Patricia Cornwell
    49514  253821  ...  Patricia Cornwell
    49515  265115  ...  Patricia Cornwell
    49516  266226  ...  Patricia Cornwell
    
    [49517 rows x 7 columns]
    


```python
# Remove duplicates
books_with_ratings_2 = books_with_ratings.drop_duplicates(['title', 'user'])
```


```python
# Preparing data table for analysis
books_with_ratings_pivot = pd.pivot_table(data=books_with_ratings_2, values='rating', index='title', columns='user').fillna(0)
print(books_with_ratings_pivot)
```

    user                                                254     ...  278418
    title                                                       ...        
    1984                                                   9.0  ...     0.0
    1st to Die: A Novel                                    0.0  ...     0.0
    2nd Chance                                             0.0  ...     0.0
    4 Blondes                                              0.0  ...     0.0
    A Beautiful Mind: The Life of Mathematical Geni...     0.0  ...     0.0
    ...                                                    ...  ...     ...
    Without Remorse                                        0.0  ...     0.0
    Year of Wonders                                        0.0  ...     0.0
    You Belong To Me                                       0.0  ...     0.0
    Zen and the Art of Motorcycle Maintenance: An I...     0.0  ...     0.0
    \O\" Is for Outlaw"                                    0.0  ...     0.0
    
    [673 rows x 888 columns]
    


```python
# Convert to 2D matrıx
books_with_ratings_matrix = csr_matrix(books_with_ratings_pivot.values)
```


```python
# Train Model
model_knn = NearestNeighbors(algorithm='auto', metric='cosine')
model_knn.fit(books_with_ratings_matrix)
```




    NearestNeighbors(algorithm='auto', leaf_size=30, metric='cosine',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     radius=1.0)




```python
# function to return recommended books - this will be tested
def get_recommends(book = ""):

  X = books_with_ratings_pivot[books_with_ratings_pivot.index == book]
  X = X.to_numpy().reshape(1,-1)
  distances, indices = model_knn.kneighbors(X,n_neighbors=8)
  recommended_books = []
  for x in reversed(range(1,6)):
      bookrecommended = [books_with_ratings_pivot.index[indices.flatten()[x]], distances.flatten()[x]]
      recommended_books.append(bookrecommended)
  recommended_books = [book, recommended_books]
  
  return recommended_books
```


```python
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)
```

    ["Where the Heart Is (Oprah's Book Club (Paperback))", [["I'll Be Seeing You", 0.8016211], ['The Weight of Water', 0.77085835], ['The Surgeon', 0.7699411], ['I Know This Much Is True', 0.7677075], ['The Lovely Bones: A Novel', 0.7234864]]]
    

Use the cell below to test your function. The `test_book_recommendation()` function will inform you if you passed the challenge or need to keep trying.


```python
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2): 
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You havn't passed yet. Keep trying!")

test_book_recommendation()
```

    ["Where the Heart Is (Oprah's Book Club (Paperback))", [["I'll Be Seeing You", 0.8016211], ['The Weight of Water', 0.77085835], ['The Surgeon', 0.7699411], ['I Know This Much Is True', 0.7677075], ['The Lovely Bones: A Novel', 0.7234864]]]
    You passed the challenge! 🎉🎉🎉🎉🎉
    
