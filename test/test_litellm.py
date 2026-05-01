import litellm
import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyDzqqzfqt9mfMp0fOhaa1Pczvvir6vOSKQ'
try:
    response = litellm.completion(model='gemini/gemini-2.5-flash', messages=[{'role': 'user', 'content': 'Hello'}])
    print(response)
except Exception as e:
    print('ERROR:', e)
