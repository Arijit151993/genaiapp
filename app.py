from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json

app = Flask(__name__)


genai.configure(api_key="AIzaSyB-n9nVWlbeN9Ey0bGkRNfQSIhE9hDi0FE")



# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
 

  # See https://ai.google.dev/gemini-api/docs/safety-settings
)


@app.route('/')
def index():
  return render_template('index.html')  # Assuming you have an index.html file

def toxicity_check(query):


  model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    system_instruction="You are a Content Moderation Agent and your task it identify with your intelligence that weather you the content is Toxic in nature of not if it is then you have to give the degree of its toxicity percentage, identify its language , then try to identify the toxic words in that sentiment  and also identify its sentiment score ",
  )

  chat_session = model.start_chat(
    history=[
      {
        "role": "user",
        "parts": [
          "You are bullshit",
        ],
      },
      {
        "role": "model",
        "parts": [
          "## Content Analysis:\n\n**Toxicity:** 90% \n\n**Language:** English\n\n**Toxic_Words:** \"bullshit\"\n\n**Sentiment_Score:** Extremely",
        ],
        },
    ]
  )

  response = chat_session.send_message(query)

  return(response.text)


def RAG_prompt(query):
  response = model.generate_content([
    "Your Task is to convert the below mentioned content to a Non-Toxic content and give 3 best Answers.\nPoints to consider :- \n1. I only need answers I don't need its explanation.\n2. Give Answer in the format it has been shown in examples.\n3. Do not generate any answer from your sideOutput should be in below mentioned format :- \n{ \"data\" = [   \n\"I understand that you have strong feelings about this individual. Perhaps we can focus on the facts of the situation instead?\",\n\"It's important to remember that everyone deserves to be treated with respect, even if we disagree with them.\",\n\"Let's try to have a constructive discussion about this topic without using inflammatory language.\" \n]\n}",
    "input: You all are some dumb as people",
    "output: 1. There seems to be some confusion about this topic. Perhaps we can clarify it together?\n2. I'm happy to explain this further if you'd like.\n3. I see things differently. Can you explain your perspective?",
    "input: This guy is a moron",
    "output: 1. I don't understand that approach. Can someone explain the reasoning here?\n2. I disagree with that statement. Here's why...\n3. There might be a better way to approach this. Perhaps we can...",
    "input: Why the hell are they even interviewing him? This guy's such an idiot",
    "output: 1. I'm curious, what makes this interview relevant to the discussion?\n2. I don't agree with everything this person says, but I'm interested in hearing their perspective\n3. I'm more interested in hearing from the other interviewee on this topic.",
    "input: This guy an idiot, if that's the best you can do, you have lost already!! WHAT A BOZO!!\" into non-toxic content",
    "output: 1. There might be room for improvement here. Have you considered...?\n2. The outcome might not be ideal, but let's see if we can brainstorm some better solutions.\n3. Everyone makes mistakes. Let's learn from this and try again.",
    "input: "+query,
    "output: ",
  ])
  json_output = json.loads(response.text)
  return json_output['data']
  
@app.route('/submit', methods=['GET', 'POST'])
def submit():
  print("Submit Form Router called")
  if request.method == 'POST':
    query = request.json.get('code')
    toxicity = toxicity_check(query)
    json_output = json.loads(toxicity)
    user_input = RAG_prompt(query)
    if(float(json_output["Toxicity"].replace("%", ""))/100)>(0.4):
      user_input = user_input  
    else :
      user_input = "This is not a Toxic Content"
    print(user_input),
    print(json_output["Toxicity"])
    print(json_output["Language"])
    print(json_output["Toxic_Words"])
    print(json_output["Sentiment_Score"])
    data = {
        'user_input': (user_input),
        'toxicity_score': json_output["Toxicity"],
        'sentiment_score': json_output["Sentiment_Score"],
        'language':json_output["Language"] ,
        'toxic_words': json_output["Toxic_Words"]
    }
    return jsonify(data)
    
if __name__ == '__main__':
  app.run()

# query = "This guy is a moron."
# toxicity_check(query)
