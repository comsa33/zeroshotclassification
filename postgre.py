from sqlalchemy import create_engine
import settings

postgre_host = settings.POSTGRE_HOST
postgre_port = settings.POSTGRE_PORT
postgre_username = settings.POSTGRE_USERNAME
postgre_password = settings.POSTGRE_PASSWORD
postgre_database = settings.POSTGRE_DATABASE

database_url = f"postgresql://{postgre_username}:{postgre_password}@{postgre_host}:{postgre_port}/{postgre_database}"

postgre_engine = create_engine(database_url)
