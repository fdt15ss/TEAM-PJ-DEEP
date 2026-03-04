fastapi 실행
cd c:\team2\TEAM-PJ-DEEP\fastapi ; ..\\.venv\Scripts\uvicorn.exe fastapi_main:app --reload --host 127.0.0.1 --port 7394

swagger
http://localhost:7394/docs

streamlit 실행
.venv\Scripts\streamlit.exe run streamlit\face_streamlit_main.py --server.port 8501

http://localhost:8501/
