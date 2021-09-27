from models.predict_model import PredictSentiment
import streamlit as st

pred_sent = PredictSentiment()

st.title('UVU Data Science Club Sentiment Classifier')
st.subheader('Built By Drew Ipson')
st.write("This application predicts whether or not a block of text is positive or negative by using a Multinominal Naive Bayes model trained off of 25,000 movie reviews obtained from Kaggle.com. Please enter the text you want analyzed and the model will predict whether or not the text is potivie or negative.")
default_movie_review = "I went to see Shang Chi and the legend of the ten rings by marvel studios in theaters, and this movie has blown me away meaning it was incredible. It has been my dream to see my very first MCU movie in theaters. I’m not going to lie, but this movie truly brought me to see it since I feel a real inspiration for martial arts. But as soon as I saw the movie, the human acting was so strong, incredible, and speechless. This had one of the most definable and meaningful origins in the movie which is dark and emotional. Marvel decided to come up with another flavor that has never existed before. First MCU started with nordic culture with thor, African culture for black panther, and they finally come up with a stunning martial arts/Asian culture that is so satisfying. The mandarin (Tony Leung Chiu-wai) was an incredible character in the movie. It made me feel that I want to see this movie again, he’s my favorite character in the movie. Awkwafina was the funniest actor and there isn’t a problem with it since MCU needs humor. Lastly, Simu Liu as Shang chi’s dream of becoming a superhero became true and he did very excellently. This movie has various easter eggs you must know, some of them are teasers while some of them are returning MCU characters. This movie isn’t only a must-watch, but a movie to give a tribute to martial arts culture for those who are true fans of it. I would give this a 9.6/10 perfect score. Don’t forget, there are post-credits!"
user_input = st.text_area("Enter text here:", default_movie_review)
run_text = st.button("Predict Sentiment")
if run_text and user_input != '':
    results = pred_sent.get_sentiment(user_input)
    if results['positive'] == True:
        st.success('This is a positive review!')
    else:
        st.error("This is probably not a good movie to see...")

see_confidence = st.checkbox("See sentiment score.")
if see_confidence == True and run_text:
    st.progress(int(results['confidence']*100))
    st.info(f"Our sentiment score for this text is {round(results['confidence']*100,2)}%.")