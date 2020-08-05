FROM python:3.6
EXPOSE 8501
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY resources ./resources
RUN pip install resources/oats-0.0.1-py3-none-any.whl

COPY . .
CMD streamlit run app.py