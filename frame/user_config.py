from frame.config_handle import Config


class UserConfig(Config):
    user: str
    out_dir: str
    scripts_dir: str
