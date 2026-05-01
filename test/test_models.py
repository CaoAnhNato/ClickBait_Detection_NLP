import sys
import sys; sys.path.insert(0, 'GUI/application/backend'); import litellm_compat; sys.modules['litellm'] = litellm_compat; import litellm

litellm.set_verbose = False
api_key = 'sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v'
base_url = 'https://api-v2.shopaikey.com/v1'

for m in ['gpt-4o', 'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-pro', 'gemini-1.5-pro', 'gpt-4o-mini']:
    try:
        response = litellm.completion(
            model=m,
            messages=[{'role': 'user', 'content': 'hi'}],
            api_key=api_key,
            api_base=base_url,
            custom_llm_provider='openai',
            max_tokens=5
        )
        print(f'SUCCESS: {m}')
        break
    except Exception as e:
        print(f'FAIL: {m} - {e}')
