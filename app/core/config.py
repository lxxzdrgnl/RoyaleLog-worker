from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # DB
    db_host: str = "localhost"
    db_port: int = 5434
    db_name: str = "royalelog"
    db_user: str = "royale"
    db_password: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Model
    model_dir: str = "./models"
    state_file_path: str = "./training_state.json"
    window_days: int = 3
    val_days: int = 1
    min_child_samples: int = 50
    min_train_rows: int = 10000
    min_post_patch_days: int = 3
    early_stopping_rounds: int = 50
    accuracy_margin: float = 0.005
    train_battle_types: list[str] = ["pathOfLegend", "ladder"]

    # Server
    worker_port: int = 8082

    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
