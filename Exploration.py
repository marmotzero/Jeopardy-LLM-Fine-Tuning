import pandas as pd
import json

jeopardy = pd.read_csv('JEOPARDY_CSV.csv')

print(jeopardy.head(10))
print(jeopardy.columns)
print(jeopardy.dtypes)

jeopardy = jeopardy.astype(str)
print(jeopardy.columns)
print(jeopardy.dtypes)

# Optional code for shortening the number of rows if training is expensive

jeopardy = jeopardy.iloc[:3000]

# Formatting for OpenAI API fine tuning


print(jeopardy.head(10))

messages = []
for _, row in jeopardy.iterrows():
    system_message = {"role": "system", "content": "You are an expert Jeopardy coach. Please quiz me on lots of questions so that I can train to be a contestant on the show!"}
    user_message = {"role": "user", "content": "Show " + row['Show Number'] + " " + row[' Air Date'] + " Round: " + row[' Round'] + " Category: " + row[' Category'] + " Value: " + row[' Value'] + " " + row[' Question']}
    assistant_message = {"role": "assistant", "content": row[' Answer']}
    messages.append([system_message, user_message, assistant_message])

jeopardy = jeopardy.drop(columns=['Show Number', ' Air Date', ' Round', ' Category', ' Value', ' Question', ' Answer'], axis=1)

print(jeopardy.head(10))


jeopardy_out = 'jeopardy_fine_tune_OpenAI.jsonl'

jeopardy_out = 'jeopardy_fine_tune_OpenAI.jsonl'
with open(jeopardy_out, 'w') as f:
    for msg in messages:
        f.write(json.dumps({"messages": msg}) + '\n')
        