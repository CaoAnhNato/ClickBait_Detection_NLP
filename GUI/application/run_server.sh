#!/bin/bash
cd "/mnt/c/Users/Admin/HUIT - Học Tập/Năm 3/Semester_2/Class/NLP/GUI/application"
export PYTHONPATH="$(pwd):$(pwd)/tests:$(pwd)/../tests:/mnt/c/Users/Admin/HUIT - Học Tập/Năm 3/Semester_2/Class/NLP/GUI/test"
export ORCD_MODEL_KEY=generate-and-predict
export ORCD_API_KEY=sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v
export ORCD_API_MODEL_OVERRIDE=gpt-3.5-turbo-1106
export ORCD_API_BASE_OVERRIDE=https://api-v2.shopaikey.com/v1
export ORCD_API_PROVIDER_OVERRIDE=openai
/home/nato/ENTER/envs/MLE/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
