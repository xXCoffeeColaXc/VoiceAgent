from sqlalchemy import create_engine, text
import os


def get_db_connection_str() -> str:
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return connection_string


# if __name__ == "__main__":
#     engine = get_db_engine()
#     with engine.connect() as connection:
#         result = connection.execute(text("SELECT * FROM documents"))
#         print(result.all())
