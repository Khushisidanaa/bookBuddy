import sys
print("Python Executable:", sys.executable)
print("Python PATH:", sys.path)


from surprise import Dataset, Reader, SVD 
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from shiny import ui, render, App, experimental
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Load your dataset
df = pd.read_csv('data/books-ratings.csv', nrows=50000)

df2=pd.read_csv('data/users-ratings.csv', nrows=50000)

unique_cities = df2['Location'].str.split(',').str[0].unique().tolist()

# Assuming 'Book-Title' and 'Image-URL-L' are columns in your DataFrame
books = df['Book-Title'].unique().tolist()

def popular_books(df, n=100):
    rating_count = df.groupby("Book-Title").count()["Book-Rating"].reset_index()
    rating_count.rename(columns={"Book-Rating": "NumberOfVotes"}, inplace=True)

    rating_average = df.groupby("Book-Title")["Book-Rating"].mean().reset_index()
    rating_average.rename(columns={"Book-Rating": "AverageRatings"}, inplace=True)

    popular_books = rating_count.merge(rating_average, on="Book-Title")
    image_data = df[['Book-Title', 'Image-URL-L']].drop_duplicates(subset=['Book-Title'])
    popular_books = popular_books.merge(image_data, on="Book-Title", how='left')

    C = popular_books["AverageRatings"].mean()
    m = popular_books["NumberOfVotes"].quantile(0.90)

    def weighted_rate(x):
        v = x["NumberOfVotes"]
        R = x["AverageRatings"]
        return ((v * R) + (m * C)) / (v + m)

    popular_books['Popularity'] = popular_books.apply(weighted_rate, axis=1)
    popular_books = popular_books[popular_books["NumberOfVotes"] >= 250].sort_values(by="Popularity", ascending=False)
    return popular_books[["Book-Title", "NumberOfVotes", "AverageRatings", "Popularity", "Image-URL-L"]].head(n)
    
top_books_df = popular_books(df, 8)

# Function to train SVD model
def train_svd_model(df):
    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(df[['User-ID', 'Book-Title', 'Book-Rating']], reader)
    train_set, test_set = train_test_split(data, test_size=0.20, random_state=42)
    model = SVD()
    model.fit(train_set)
    return model

# Function to recommend books using SVD
def recommend_books_svd(model, user_id, n=10):
    # Fetch user data
    user_data = df[df['User-ID'] == user_id]

    # Get unique books from the dataset and books rated by the user
    all_books = set(df['Book-Title'].unique())
    rated_books = set(user_data['Book-Title'].values)
    books_to_predict = list(all_books - rated_books)

    predictions = []
    seen_books = set()  # To ensure no duplicates in recommendations

    for book in books_to_predict:
        if book not in seen_books:
            pred = model.predict(user_id, book)
            predictions.append((book, pred.est))
            seen_books.add(book)  # Preventing recommending the same book again

    # Sorting predictions by estimated ratings
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    return top_n



# Function to train KNN model
def train_knn_model(df):
    # Assuming 'Book-Title' is a string that needs to be encoded
    le = LabelEncoder()
    df['Encoded-Book-Title'] = le.fit_transform(df['Book-Title'])
    
    # Now fitting model on numeric data only
    model = NearestNeighbors(metric='cosine')
    model.fit(df[['User-ID', 'Encoded-Book-Title', 'Book-Rating']].values)
    return model, le

#
# Function to recommend books using KNN
def recommend_books_knn(model, book_title, le, n=10):
    try:
        # Transforming the book title using the captured LabelEncoder
        book_index = le.transform([book_title])[0]
        book_data = df[df['Encoded-Book-Title'] == book_index]

        # Fetch ingmore neighbors initially to ensure enough remain after filtering
        distance, indices = model.kneighbors(book_data[['User-ID', 'Encoded-Book-Title', 'Book-Rating']].values, n_neighbors=n + 20)

        # Preparing the recommended books DataFrame
        recommended_books = pd.DataFrame({
            'title': le.inverse_transform(df.iloc[indices.flatten()]['Encoded-Book-Title']),
            'distance': distance.flatten()
        })

        # Filtering out the exact book and those with distance of 0
        recommended_books = recommended_books[(recommended_books['title'] != book_title) & (recommended_books['distance'] > 0)]

        # Droping duplicates and sorting by distance
        recommended_books = recommended_books.drop_duplicates(subset='title').sort_values(by='distance', ascending=True).head(n)
        
        return recommended_books
    except Exception as e:
        print(f'Error: {e}')
        return []


def generate_book_cards(recommendations):
    return ui.layout_column_wrap(
        *[experimental.ui.card(
                          experimental.ui.card_image(
                            file=None, 
                            src=row['Image-URL-L'],  # Ensure this column is correctly named and merged
                            fill=True,
                            border_radius='all',
                            height='200px',
                            width='200px',
                        ),
                         experimental.ui.card_footer(row['Book-Title']),
                        
                    ) for _, row in recommendations.iterrows()],
        width="200px"
)


# Defining UI layout
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_numeric("user_id", "User ID", value=11676, min=1),  
        ui.input_selectize(
                "book", "Enter your current favourite book!", books, selected=top_books_df.iloc[0]['Book-Title'] if not top_books_df.empty else None
            ),
            ui.output_text("bookdetails"),
          
        ui.input_selectize("recommendation_type", "Recommendation Type", ['Based on Your Previous Ratings', 'Based on the Book You Have Chosen']),
        ui.markdown("<p4 style='color: navy; margin-bottom: 20px; align: center'>Recommendations based on demographics  </p4>"),
        ui.input_numeric("age", "Enter your age", min=0, value=15),
        ui.input_selectize("location", "Enter your location", unique_cities, selected="barcelona"),
        bg="#f8f8f8",
        open="always",
        width=334,
    ),
    ui.page_navbar(
        ui.nav_panel("Home",
            ui.markdown("<h1 style='color: navy; margin-bottom: 20px; align: center'>Popular Books Right Now</h1>"),
         ui.layout_column_wrap(
                    *[experimental.ui.card(
                        
                        experimental.ui.card_image(
                            file=None, 
                            src=row['Image-URL-L'],  
                            fill=True,
                            border_radius='all',
                            height='200px',
                            width='200px',
                        ),
                         experimental.ui.card_footer(row['Book-Title']),
                        
                    ) for index, row in top_books_df.iterrows()],
                    width="200px",
            ),  ),
            ui.nav_panel("Recommendations",
             ui.markdown("<h1 style='color: navy; margin-bottom: 20px; align: center'>Personalised Recommendations for you </h1>"),
            ui.output_ui("output_recommendations"),  # This will display the output from recommended_books
        ),
        ui.nav_panel("Demographic Recommendations",
             ui.markdown("<h1 style='color: navy; margin-bottom: 20px; align: center'>Recommendations for you based on Demographics- age and location!   </h1>"),
            ui.output_ui("output_recommendations_demographics"),  # This will display the output from recommended_books
        ),
        ui.nav_panel("About",
             ui.markdown("""
### Welcome to BookBuddy!

**BookBuddy** is your go-to destination for discovering your next favorite book. With a vast library and intuitive features, we make finding your next read easy and enjoyable.

#### Discover New Favorites

BookBuddy offers personalized recommendations tailored to your reading tastes. Whether you're looking for something similar to your recent favorites or wanting to explore a new genre, BookBuddy has you covered.

#### Tailored to You

Our recommendations are customized just for you, based on your preferences and reading history. We also offer suggestions based on popular trends in your area and among readers of your age, helping you stay in sync with the best and most talked-about books.

#### Easy and Interactive

BookBuddy's user-friendly interface ensures that finding your next book is a breeze. Browse through our selections, check out what's popular, or search for books by title, author, or genre—all in a few clicks.

#### Always Up-to-Date

Stay ahead of the curve with our 'Popular Books Right Now' feature, which showcases the top books that are currently trending.

#### Join Our Community

Become a part of a community of readers who share your passion. BookBuddy is not just about book recommendations; it's a platform to discover, discuss, and dive deeper into the world of books.

**Dive into your next reading adventure with BookBuddy—where your next book finds you!**
""")
,), 
            
        id="page",
    ),
    title="BookBuddy- Your Book Recommendation System",
    id="page_sidebar",
    )


def server(input, output, session):
    @output
    @render.ui('output_recommendations')
    def output_recommendations():
        user_id = input.user_id()
        recommendation_type = input.recommendation_type()

        
        if recommendation_type == 'Based on Your Previous Ratings':

            if user_id not in df['User-ID'].unique():
                return ui.markdown("<h4 style='color: navy; margin-bottom: 20px; align: center'>User does not exist.Try making recommendations based on your favourite book instead!  </p4>"),

            user_ratings = df[df['User-ID'] == user_id]
            if len(user_ratings) <= 5:
                return ui.markdown("<h4 style='color: navy; margin-bottom: 20px; align: center'>You have less than 5 votes, not enough to make strong recommendations! Try making recommendations based on your favourite book instead!  </p4>"),

            if len(user_ratings) > 5:
                svd_model = train_svd_model(df)
                recommendations = recommend_books_svd(svd_model, user_id, n=8)
                print(recommendations)
                recommended_df = df[df['Book-Title'].isin([title for title, _ in recommendations])].drop_duplicates(subset='Book-Title')
              
                return generate_book_cards(recommended_df)

        elif recommendation_type == 'Based on the Book You Have Chosen':
            chosen_book = input.book()
            knn_model, le = train_knn_model(df)
            recommendations = recommend_books_knn(knn_model, chosen_book,le, n=8)
            print(recommendations)
            # Assuming recommendations is the DataFrame you showed
            recommended_df = df[df['Book-Title'].isin(recommendations['title'])].drop_duplicates(subset='Book-Title')


            return generate_book_cards(recommended_df)
    
    @output
    @render.ui('output_recommendations_demographics')
    def output_recommendations_demographics():
        age = input.age()
        location = input.location()

        # Filtering users based on location and age range
        filtered_users = df2[(df2['Location'].str.contains(location)) & (df2['Age'] >= (age - 5)) & (df2['Age'] <= (age + 5))]
        filtered_user_ids = filtered_users['User-ID'].unique()

        # Filtering the ratings DataFrame to get ratings only from these users
        filtered_ratings = df[df['User-ID'].isin(filtered_user_ids)]

        # Calculating popularity among filtered ratings
        if not filtered_ratings.empty:
            return generate_popular_books(filtered_ratings)
        else:
            return ui.markdown("No available data for the specified demographics.")

    def generate_popular_books(filtered_df):
        rating_count = filtered_df.groupby('Book-Title').size().reset_index(name='NumberOfVotes')
        rating_average = filtered_df.groupby('Book-Title')['Book-Rating'].mean().reset_index(name='AverageRatings')
        
        popular_books = pd.merge(rating_count, rating_average, on='Book-Title')

        popular_books = popular_books.sort_values(by='AverageRatings', ascending=False)
        
        image_data = filtered_df[['Book-Title', 'Image-URL-L']].drop_duplicates('Book-Title')
        popular_books = pd.merge(popular_books, image_data, on='Book-Title', how='left')

        return generate_book_cards(popular_books.head(8))




app = App(app_ui, server)
