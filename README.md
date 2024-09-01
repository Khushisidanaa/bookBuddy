# BookBuddy: Your Personalized Book

## Statement of Purpose

BookBuddy is designed to revolutionize the way readers discover books they love. Leveraging advanced machine learning algorithms, BookBuddy provides personalized book recommendations based on user preferences and demographic data. Our goal is to create an intuitive and engaging platform that enhances reading experiences by accurately suggesting books that align with individual tastes and trends.

## Domain/Context of Application

BookBuddy serves the domain of digital libraries and e-commerce platforms where users seek guidance on what books to read next. BookBuddy addresses the real-world problem of choice overload in literature by providing personalized book recommendations, helping users efficiently discover books that match their preferences and reading habits. It is particularly useful for:

- Readers looking to discover new books based on their reading history.
- New readers seeking guidance on which books might suit their tastes.
- Libraries and bookstores aiming to provide better recommendations to their patrons.
- Readers looking to discover new books based on a book they really liked.
- Readers looking for books people their age and in their locality are reading.

## Data Preprocessing

Before providing recommendations, BookBuddy performs several preprocessing steps on the data:

- **Book Titles:** Special characters are removed, and text is standardized to ensure consistency.
- **User Data:** Ages are normalized, and user locations are parsed to extract meaningful geographic data.
- **Ratings:** We filter out anomalies and ensure that the ratings data used in our models is robust and reliable.

## BookBuddy's Functionality

### 1. Recommendation Based on Book - Using KNN:

BookBuddy employs the K-Nearest Neighbors (KNN) algorithm to provide recommendations based on book similarity. This method works by encoding each book title into a numeric vector using label encoding, which translates the book titles into a format that can be processed by the algorithm. When a user selects a book, the KNN algorithm identifies the closest neighboring books based on content similarity, measured through user ratings. This ensures that the recommendations are closely related to the user's selected book, providing a personalized experience that focuses on content relevance.

### 2. User-Based Recommendations - Using SVD:

For recommendations based on user history, BookBuddy utilizes Singular Value Decomposition (SVD), a type of collaborative filtering. This method predicts a user's book preferences by analyzing patterns from similar users' ratings. The system handles various scenarios gracefully:

- **No Votes:** If a user has no voting history (i.e., no ratings given), the system informs the user that no personalized recommendations can be made due to the lack of data.
- **Insufficient Votes:** If a user has given fewer than 5 ratings, BookBuddy communicates that there are too few data points to generate reliable recommendations, encouraging further interaction with the platform.

### 3. Trending Books - Popularity Based Recommendations:

The 'Popular Books Right Now' feature showcases books that are currently trending across the platform. This section is dynamically generated by calculating a weighted popularity score for each book, which considers both the average rating and the number of votes a book has received. Books that meet a minimum threshold of votes are displayed, ensuring that the books listed are not only popular but also broadly validated by a larger audience.

### 4. Demographic-Based Recommendations:

BookBuddy also offers recommendations based on demographic data. It filters user ratings based on age and location, allowing the system to suggest books that are popular within specific demographic groups. This feature helps users discover books that are particularly appreciated by peers in similar age groups or geographic locations, adding another layer of personalization to the recommendations.

## Setup Instructions

To set up BookBuddy on your local machine, follow these steps:

**Clone the Repository:**

```bash
git clone git@github.com:Khushisidanaa/bookBuddy.git
cd bookBuddy
```

## Set Up a Python Virtual Environment

```bash
python3.8 -m venv myenv
source myenv/bin/activate  # On Unix or MacOS
myenv\Scripts\activate     # On Windows
```

##Install Required Packages

```bash
pip install -r requirements.txt
```

If you encounter any errors after installing requirements while running the app like surprise doesnt exist :
Sometimes, package installations or updates require reinitializing the environment to properly load new configurations. If you encounter errors immediately after installation, try deactivating and then reactivating your virtual environment:

Deactivate the Virtual Environment

```
deactivate
```

Reactivate the Virtual Environment

```
source myenv/bin/activate  # On Unix or MacOS
myenv\Scripts\activate     # On Windows
```

Run the Application again/first time

```bash
shiny run app.py
```

## Usage Examples

### Starting the Application

Navigate to the project directory and run:

```bash
shiny run app.py
```

This command starts a local server where BookBuddy is accessible via a web browser.

## Getting Recommendations

### By User Preferences

Enter your user ID to get recommendations based on your past ratings.

- **Default user ID** is given, you can also try another existing one which is is 8.
- **Try with user ID 0**, an ID that does not exist, to see how it handles.
- **Try with user ID 67544**, an ID with fewer than 5 votes, to understand its response.

## Explore Popular Books

Visit the 'Popular Books Right Now' section on the Home page to see trending books based on user ratings.

## Explore Popular Books Based on Demographic Data

Check out books popular among users in your age group and locality to find culturally and regionally relevant reads. Try changing age and also location, if there isn't enough location data, it will let you know.

## About Page

The About page provides a guide on how to navigate and utilize the web application effectively.

Note : Due to size limitations on direct uploads, I have uploaded large files as zip files in data-raw. If you need to check the preprocessing steps, you can unzip those files.
