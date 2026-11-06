class OmniFactoryError(ValueError):
    """Actionable user-input or contract error, safe to display in the CLI."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")
