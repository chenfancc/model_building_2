class BalancedData:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        pass

    def oversample(self):
        data_balanced = self.data
        labels_balanced = self.labels
        return data_balanced, labels_balanced

    def undersample(self):
        data_balanced = self.data
        labels_balanced = self.labels
        return data_balanced, labels_balanced

    def smote(self):
        data_balanced = self.data
        labels_balanced = self.labels
        return data_balanced, labels_balanced

    def sample(self, method='oversample'):
        """
        根据指定方法进行采样
        """
        if method == 'oversample':
            return self.oversample()
        elif method == 'undersample':
            return self.undersample()
        elif method == 'smote':
            return self.smote()
        else:
            raise ValueError(f"Unknown sampling method: {method}")
