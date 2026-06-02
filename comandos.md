
uv venv
.venv\Scripts\activate
uv sync --no-install-project
crewai install
cd src/content_marketing_project_manager
uv run main.py
uv run uvicorn main:app --reload
uvicorn main:app --reload --port 8000
crewai run


poetry lock --no-update
poetry install
poetry run python mi_script.py

poetry add langchain-ollama


DuckDuckGo

pip install -U ddgs

uv run streamlit run .\main.py

