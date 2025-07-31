import pickle
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")



with open("./model.pkl", "rb") as f:
    clf = pickle.load(f)


model = SentenceTransformer("all-MiniLM-L6-v2")


def predict(text: str):
    vectorized_text = model.encode(text)
    prediction = clf.predict([vectorized_text])
    return prediction[0]


feedback = "I like to wear this color dress"
sentiment = predict(feedback)
print(sentiment)


# if __name__ == "__main__":
#     while True:
#         feedback = input("Enter your feedback(or type end to stop):  ")
#         if feedback.strip().lower() == "end":
#             break
#         sentiment = predict(feedback)

#         print(f"Sentiment: {sentiment}")