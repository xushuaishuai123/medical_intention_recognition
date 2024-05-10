class Args():
    train_path = './data/IMCS-DAC_train.json'
    eval_test = './data/IMCS-DAC_test.json'
    eval_path='./data/IMCS-DAC_dev.json'
    seq_labels_path = './data/dialogue_acts.txt'
    bert_dir = r'C:\Users\xss\Desktop\pretrain_models\bert-base-chinese'
    save_dir = './checkpoints/'
    load_dir = './checkpoints/'
    do_train = True
    do_eval = True
    do_test = True
    do_save = True
    do_predict = True
    load_model = False
    device = None
    eval_steps=0.2
    seqlabel2id = {}
    id2seqlabel = {}
    with open(seq_labels_path, 'r') as fp:
        seq_labels = fp.read().split('\n')
        for i, label in enumerate(seq_labels):
            seqlabel2id[label] = i
            id2seqlabel[i] = label

    hidden_size = 768
    seq_num_labels = len(seq_labels)
    max_len = 32
    batchsize = 64
    lr = 2e-5
    weight_decay=0.01
    epoch = 10
    hidden_dropout_prob = 0.1
    show_progress_bar=True


if __name__ == '__main__':
    args = Args()
    print(args.seq_labels)
    print(args.seqlabel2id)
    print(args.id2seqlabel)
