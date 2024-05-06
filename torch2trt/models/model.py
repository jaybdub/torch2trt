from typing import Any


class Model:

    @classmethod
    def build(self, model_path: str):
        raise NotImplementedError
    
    @classmethod
    def load(self, model_path: str):
        raise NotImplementedError
    