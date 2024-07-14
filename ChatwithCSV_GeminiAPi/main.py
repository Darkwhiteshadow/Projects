import utility
import pandas as pd
import matplotlib as plt
import seaborn as sns

import google.generativeai as genai



apikey = ""

genai.configure(api_key=apikey)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"temperature": 0.3,
                       "top_p": 0.95,
                       "top_k": 64,
                       "max_output_tokens": 8192,
                       "response_mime_type": "text/plain", },
)
path = "Sample2.csv"
files = [
  utility.upload_file(path, mime_type="text/csv"),
]
utility.check_activation_of_file(files)

chat_session = model.start_chat(
  history=[]
)

with open('role.txt','r')as fp:
    role = fp.read()

response = chat_session.send_message(role)
print(response.text)

df = pd.read_csv(path)

addon = f"addition info > file path : {path}, dataframe name : df,dataframe is already readed,if asked to plot save plots as image instead of showing, also if something is irrelavant to data just say irrelavant question"
debug = False
while(True):

    if(debug==False):
        retry =0
        question = input("Enter you question here:  ")

    if(question=='Exit'): break

    query=question+addon

    answer =  chat_session.send_message([query,files[0]])
    print("...processing_response...")
    codes=utility.get_code(answer.text)

    if(len(codes)==0):
        print(answer.text)
    else:
        try:

            for code in codes:
                exec(code)
                exec("plt.close('all')")
            debug = False
        except Exception as e:
            question=question+str(e)+"resolve"
            print(answer.text)
            retry+=1
            debug = True
            if(retry>2):
                debug=False
                print("Sorry couldn't perform the task")










