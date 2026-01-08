from sqlalchemy import text

def create_predictions_table(engine):
    query = """
    CREATE TABLE IF NOT EXISTS text (
        input TEXT,
        output TEXT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(query))
    print("Table created or already exists.")

def insert_prediction(engine, input, output):
    query = text("""
        INSERT INTO text (input, output)
        VALUES (:input, :output)
    """)
    with engine.begin() as conn:
        conn.execute(query, {
            "input": input,
            "output": output
        })
    print("Prediction inserted successfully.")

