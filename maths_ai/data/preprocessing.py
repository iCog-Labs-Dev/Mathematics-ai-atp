class Preprocessor:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        # Implement data cleaning logic here
        pass

    def normalize_data(self):
        # Implement data normalization logic here
        pass

    def preprocess(self):
        self.clean_data()
        self.normalize_data()
        return self.data
    

class AstBuilder:
    def __init__(self) -> None:
        pass


class GraphExpressionBuilder:
    def __init__(self) -> None:
        pass


class AbstractionLayerBuilder:
    def __init__(self) -> None:
        pass