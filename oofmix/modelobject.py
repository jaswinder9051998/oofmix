


class SplitModelObjectHolder():
    def __init__(self, models:list=[]):
        self.models = models

    def add_model(self, model):
        self.models = self.models.appned(model)

    def predict(self, dataframe):
        result = 0
        for model in self.models:
            result += model.predict(dataframe)
        return result/len(self.models)


