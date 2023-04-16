class TodoError(Exception):
    def __init__(self, input_message = ''):
        if input_message == '':
            self.message = "Some extra work need to be done."
        else:
            self.message = "Some extra work need to be done: " + input_message
        super().__init__(self.message)