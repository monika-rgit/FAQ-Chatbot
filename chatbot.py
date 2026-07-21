import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ Questions and Answers
questions = [
    "What is AI?",
    "What is Python?",
    "What is Machine Learning?"
]

answers = [
    "AI stands for Artificial Intelligence.",
    "Python is a programming language.",
    "Machine Learning is a branch of Artificial Intelligence."
]

user_question = input("Ask a question: ")

vectorizer = TfidfVectorizer()

all_questions = questions + [user_question]

tfidf = vectorizer.fit_transform(all_questions)

similarity = cosine_similarity(tfidf[-1], tfidf[:-1])

index = similarity.argmax()

print("\nChatbot Answer:")
print(answers[index])