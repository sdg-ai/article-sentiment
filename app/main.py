from pydantic import BaseModel
from fastapi import FastAPI, Body
from sentiment_classifier import SentimentInterface

#TODO: Add validation to input parameters

class InputText(BaseModel):
    text: str

class InputTrends(BaseModel):
    trends: list = []

x = SentimentInterface()
app = FastAPI()

@app.get("/sentiment-example")
def get_sentiment_example():

    input_from_trend_classifier = [{'ID': 1545},
		{'string_indices': (0, 121),
            'text': 'Three-dimensional printing has changed the way we make everything from prosthetic limbs to aircraft parts and even homes.',
            'string_prediction': ['building', '3-d printing'], 'string_prob': [0.9, 0.5]},
        {'string_indices': (122, 181),
            'text': 'This is not a great start for the industry!.',
            'string_prediction': ['building', '3-d printing'], 'string_prob': [0.9, 0.5]},
        {'string_indices': (123, 154),
            'text': 'windmills',
            'string_prediction': ['building', '3-d printing'], 'string_prob': [0.9, 0.5]}]
    #Now it may be poised to upend the apparel industry as well.
    output = x.input_to_sentiment(input_from_trend_classifier=input_from_trend_classifier)
    #print("TEST:\n\n")
    #for i in range(1,len(output)):
    #    print('TEXT: '+output[i]['text']+ '\nsentiment_class:  '+output[i]['sentiment_class'] + '\nProba: ' + str(round(output[i]['sentiment_proba'],6))+'\n\n')

    return {"article_sentiment": output}

@app.post("/sentiment")
def process_sentiment(input_text: InputText):
    sentiment = x.text_to_sentiment(input_text.text)
    return {"article-sentiment": sentiment}

@app.post("/batch-sentiment")
def batch_process_sentiment(trend_results: InputTrends):
    for item in trend_results.trends:
        item['extract_sentiment'] = x.text_to_sentiment(item['text'])

    return {"article-sentiment-trends": trend_results}



