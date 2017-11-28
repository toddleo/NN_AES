class Configuration():
    def __init__(self):
        self.batch_size = 10
        self.test_batch_size = 1
        self.hidden_size = 50
        self.embedding_output = 50
        self.needCreateEM = True
        self.localGlove = 'glove.6B.50d.txt'
        self.window_size = 5
        self.num_of_filters = 50
        self.clip_grad = 0.25
        self.print_every_batch = 50

        self.dropout = 0.5
        self.output_size = 1
        self.epochs = 100
        self.init_lr = 0.5

        self.max_num_sent = 0
        self.max_length_sent = 0
        self.vocab_size = 0
        self.emb_mat = []







